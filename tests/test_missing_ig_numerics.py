"""Numerical and checkpoint contract tests; no pretrained weights or GPU."""
import io,sys,copy
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile
import numpy as np
import pytest
import torch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import missing_ig_numerics as num
import run_missing_ig_numerics as run

def test_linear_quadrature_and_precision():
    torch.set_num_threads(1)
    x=torch.linspace(-2,3,48).reshape(1,3,4,4);b=torch.ones_like(x)*.1
    coef=torch.linspace(-1,2,48).reshape_as(x)
    for method in ('midpoint','gausslegendre'):
        a32,a64=num.integrate_pair(lambda t:(t*coef).double().sum()+4,x,b,method,32)
        truth=(x-b).double()*coef.double()
        assert torch.allclose(a64,truth,rtol=1e-12,atol=1e-12)
        assert torch.allclose(a32.double(),truth,rtol=1e-5,atol=1e-5)
        assert not a64.requires_grad

def test_cubic_gauss_and_midpoint_convergence():
    x=torch.tensor([[.3,.7,1.2]]);b=torch.zeros_like(x);forward=lambda t:(t.double()**3).sum()
    truth=x.double()**3
    coarse=num.integrate_pair(forward,x,b,'midpoint',4)[1]
    fine=num.integrate_pair(forward,x,b,'midpoint',128)[1]
    gauss=num.integrate_pair(forward,x,b,'gausslegendre',8)[1]
    assert (fine-truth).abs().max()<(coarse-truth).abs().max()
    assert torch.allclose(gauss,truth,atol=1e-6,rtol=1e-6)

def test_midpoint32_reproduces_prior_fp32_implementation_exactly():
    import missing_ig_location as prior
    x=torch.linspace(-.8,1.1,48).reshape(1,3,4,4);b=torch.ones_like(x)*.1
    forward=lambda t:(torch.sin(t)*t).sum()
    want=prior.integrate(forward,x,b,32)
    actual,_=num.integrate_pair(forward,x,b,'midpoint',32)
    assert torch.equal(actual,want)

def test_directional_derivative_epsilon_ladder_and_mismatch():
    x=torch.tensor([[.3,.7,1.2]]);b=torch.zeros_like(x)
    points=num.directional(lambda t:(t.double()**3).sum(),x,b)
    for p in points:
        truth=3*p['alpha']**2*float((x.double()**3).sum())
        assert p['autograd']==pytest.approx(truth,rel=1e-5)
        assert all(v['within_tolerance'] for v in p['finite_differences'])
    class Bad(torch.autograd.Function):
        @staticmethod
        def forward(ctx,t):return t.sum()
        @staticmethod
        def backward(ctx,g):return g.expand_as(x)*2
    wrong=num.directional(Bad.apply,x,b)
    assert all(not v['within_tolerance'] for p in wrong for v in p['finite_differences'])

@pytest.mark.parametrize('kind',['method','steps','dtype','nonfinite','output'])
def test_invalid_numerical_inputs(kind):
    x=torch.ones(1,3,4,4);b=torch.zeros_like(x);method='midpoint';n=2;f=lambda t:t.sum()
    if kind=='method':method='wrong'
    if kind=='steps':n=0
    if kind=='dtype':x=x.double();b=b.double()
    if kind=='nonfinite':x[0,0,0,0]=float('nan')
    if kind=='output':f=lambda t:t.sum()*float('nan')
    with pytest.raises(ValueError):num.integrate_pair(f,x,b,method,n)

def test_completeness_warning_and_zero_map_similarity_preserved():
    assert num.completeness(np.ones((2,2)),1,0)['warning']
    assert not num.completeness(np.ones((2,2))*.25,1,0)['warning']
    assert num.similarity(np.zeros(4),np.zeros(4))['cosine'] is None
    assert num.choose(['b','a','c','d','e'])==num.choose(['e','d','c','a','b'])

def test_tiny_random_hf_dino_actual_backward_and_backend_context_restores():
    from transformers import Dinov2Config,Dinov2Model
    torch.manual_seed(42);torch.set_num_threads(1)
    model=Dinov2Model(Dinov2Config(hidden_size=12,num_hidden_layers=1,num_attention_heads=3,
        intermediate_size=24,image_size=28,patch_size=14)).eval();model.requires_grad_(False)
    native=num.NativeNumerics.__new__(num.NativeNumerics);native.torch=torch;native.engine=SimpleNamespace(model=model)
    native.head=dict(intercept=.2,mean=[0.]*12,scale=[1.]*12,coef=np.linspace(-1,1,12).tolist())
    native.mean=torch.zeros(12,dtype=torch.float64);native.scale=torch.ones(12,dtype=torch.float64)
    native.coef=torch.linspace(-1,1,12,dtype=torch.float64);native.stats=Counter()
    def tensor(im):return torch.from_numpy(np.array(im).copy()).permute(2,0,1).unsqueeze(0).float()/255,dict(original_size=[28,28],side=28,left=0,top=0,tensor_size=28,patch_size=14)
    native.tensor=tensor;image=Image.fromarray(np.arange(28*28*3,dtype=np.uint8).reshape(28,28,3))
    expected=native.feature(image);before={n:p.clone() for n,p in model.named_parameters()}
    result=native.execute(image,expected,'blur','midpoint',2)
    assert result['forward_passes']==4 and result['backward_passes']==2
    assert result['raw_maps']['fp64'].dtype==np.float64
    original=torch.backends.cuda.flash_sdp_enabled()
    result=native.execute(image,expected,'mean','derivatives',0)
    assert result['forward_passes']==43 and result['backward_passes']==6
    assert set(result['checks'])=={'default','sdpa_math'} and torch.backends.cuda.flash_sdp_enabled()==original
    assert all(torch.equal(p,before[n]) and p.grad is None and not p.requires_grad for n,p in model.named_parameters())

def seal(root):
    files={n:(run.ROOT/n).read_bytes() for n in run.old.runtime_hashes()}
    for n,b in files.items():p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    run.write(root/'source_manifest.json',{p.relative_to(root).as_posix():run.sha(p.read_bytes()) for p in root.rglob('*') if p.is_file() and p.name!='source_manifest.json'})

def feature(im):
    x=np.zeros(768,np.float32);x[0]=1.;return x

def setup(tmp):
    root=tmp/'source';inp=root/'inputs';inp.mkdir(parents=True)
    image=Image.new('RGB',(28,42),(20,30,40));buf=io.BytesIO();image.save(buf,format='PNG');b=buf.getvalue();h=run.sha(b)
    member='defect_supervised/glass-insulator/train/c0.png'
    with ZipFile(tmp/'sup.zip','w') as z:z.writestr(member,b)
    head=dict(mean=[0.]*768,scale=[1.]*768,coef=[1.]+[0.]*767,intercept=0.)
    run.write(inp/'head.json',head);run.write(inp/'requests.json',[dict(core_id='c0',image=member,image_sha256=h)])
    run.write(inp/'selection.json',dict(positive_cores=['c0']))
    seeds={}
    for kind,cell in [('original',0),('blur',1)]:
        key='c0/'+kind+'/'+str(cell);im=image if kind=='original' else run.old.old.oc.masked(image,cell,kind)
        p=inp/'features'/(kind+'.npy');p.parent.mkdir(exist_ok=True);np.save(p,feature(im),allow_pickle=False)
        seeds[key]=dict(id=key,core_id='c0',kind=kind,cell=cell,image=member,image_sha256=h,pixel_sha256=run.sha(im.tobytes()),
            feature_path=p.relative_to(root).as_posix(),feature_sha256=run.sha(p.read_bytes()))
    run.write(inp/'seeds.json',seeds)
    patch=inp/'patch.npy';np.save(patch,np.ones((37,37)),allow_pickle=False)
    run.write(inp/'previous32.json',{'c0':{k:dict(patch_path='inputs/patch.npy',original_input_sha256='fixture',baseline_input_sha256='fixture',signed_map_sha256='fixture') for k in ('blur','mean')}})
    seal(root);out=tmp/'missing_ig_numerics_fixture'
    return root,out,['--capsule-dir',str(root),'--sup-zip',str(tmp/'sup.zip'),'--out-dir',str(out)]

class Fixture:
    def __init__(self,fail=False,interrupt=None,bad_canary=False):
        self.stats=Counter(forward_passes=0,backward_passes=0,model_loads=0);self.calls=0;self.fail=fail;self.interrupt=interrupt;self.bad_canary=bad_canary
    def feature(self,im):
        self.stats['forward_passes']+=1;x=feature(im)
        if self.bad_canary:x=np.roll(x,1)
        return x
    def execute(self,im,expected,kind,method,steps):
        self.calls+=1
        if self.interrupt==self.calls:raise KeyboardInterrupt()
        if self.fail:raise RuntimeError('intentional fixture failure')
        forwards=43 if method=='derivatives' else steps+2;backwards=6 if method=='derivatives' else steps
        self.stats['forward_passes']+=forwards;self.stats['backward_passes']+=backwards
        result=dict(target_feature=expected.copy(),target_logit=1.,geometry=run.old.ig.geometry(im.size),
            original_input_sha256='fixture',baseline_input_sha256='fixture',forward_passes=forwards,backward_passes=backwards,weights_unchanged=True)
        if method=='derivatives':
            checks=[dict(alpha=a,autograd=1.,finite_differences=[dict(within_tolerance=True,relative_error=0.) for _ in range(3)]) for a in num.CONFIG['alphas']]
            result.update(checks={b:copy.deepcopy(checks) for b in num.CONFIG['backends']},attention={'fixture':True})
        else:
            raw=np.zeros((518,518),np.float32);raw[200:214,200:214]=1/196
            maps=dict(fp32=raw,fp64=raw.astype(float));result.update(raw_maps=maps,baseline_logit=0.,
                diagnostics={k:num.completeness(a,1.,0.) for k,a in maps.items()},accumulation_comparison=num.similarity(maps['fp32'],maps['fp64']))
        return result

def test_checkpoint_resume_no_model_and_warning_status_separate(tmp_path):
    root,out,args=setup(tmp_path);f=Fixture();assert run.main(args,engine=f)==0 and f.calls==10
    m=run.read(out/'run_metadata.json');assert m['forward_passes']==1704 and m['backward_passes']==1612
    again=Fixture();again.feature=lambda _:(_ for _ in ()).throw(AssertionError('resume should not load'))
    assert run.main(args,engine=again)==0 and again.calls==0
    assert run.read(out/'run_metadata.json')['resumed_conditions']==10 and again.stats['forward_passes']==0
    assert run.read(out/'run_status.json')['numerical_validation'].startswith('inspect')

def test_interruption_resumes_completed_condition(tmp_path):
    root,out,args=setup(tmp_path)
    with pytest.raises(KeyboardInterrupt):run.main(args,engine=Fixture(interrupt=2))
    saved=run.read(out/'records.json');assert len(saved)==1
    f=Fixture();assert run.main(args,engine=f)==0 and f.calls==9
    assert next(iter(saved.values()))==next(iter(run.read(out/'records.json').values()))

def test_canary_failure_blocks_then_retry_and_preserves_history(tmp_path):
    root,out,args=setup(tmp_path);f=Fixture(bad_canary=True)
    assert run.main(args,engine=f)==1 and f.calls==0
    assert run.read(out/'run_status.json')['condition_errors']==10
    assert run.main(args,engine=Fixture(fail=True))==1
    assert run.main(args,engine=Fixture())==0 and list((out/'attempt_history').glob('*.json'))

def test_numerical_warnings_and_fd_mismatch_do_not_become_execution_success_claims(tmp_path):
    root,out,args=setup(tmp_path)
    class WarningFixture(Fixture):
        def execute(self,*args):
            r=super().execute(*args)
            if 'raw_maps' in r:
                r['raw_maps']={k:a*4 for k,a in r['raw_maps'].items()}
                r['diagnostics']={k:num.completeness(a,1.,0.) for k,a in r['raw_maps'].items()}
            else:
                for checks in r['checks'].values():
                    for point in checks:
                        for fd in point['finite_differences']:fd.update(within_tolerance=False,relative_error=.8)
            return r
    assert run.main(args,engine=WarningFixture())==0
    status=run.read(out/'run_status.json')
    assert status['complete'] and status['warnings_fp64']==8 and status['derivative_points_without_matching_epsilon']==12
    assert len(run.read(out/'integration_summary.json'))==8

@pytest.mark.parametrize('kind',['source','map','record','canary','request','software'])
def test_corruption_or_identity_change_rejected(tmp_path,kind):
    root,out,args=setup(tmp_path);assert run.main(args,engine=Fixture())==0
    if kind=='source':(root/'inputs/head.json').write_text('{}')
    if kind=='map':next((out/'maps').glob('*.npy')).write_bytes(b'corrupt')
    if kind=='canary':next((out/'canary_features').glob('*.npy')).write_bytes(b'corrupt')
    if kind in ('record','request'):
        r=run.read(out/'records.json');v=next(iter(r.values()));v['target_logit' if kind=='record' else 'image_sha256']='changed';run.write(out/'records.json',r)
    if kind=='software':
        m=run.read(out/'run_metadata.json');m['identity']='changed';run.write(out/'run_metadata.json',m)
    with pytest.raises((ValueError,FileNotFoundError)):run.main(args,engine=Fixture())

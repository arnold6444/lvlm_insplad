import copy
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile
import numpy as np
import pytest
import torch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_missing_ig_location_dev as run
import build_missing_ig_location_notebook as build
ig=run.ig


def test_fast_runtime_hash_inventory_matches_existing_contract():
    assert run.runtime_hashes()==run.old.prior.shared.code_hashes()


def fixture_feature(im):
    return run.old.oc.vector([np.asarray(im,dtype=float)[:,:,0].std()/10,1],2)


def seal(root):
    manifest=run.runtime_hashes()
    for name in manifest:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((run.ROOT/name).read_bytes())
    manifest.update({p.relative_to(root).as_posix():run.old.probe.sha_file(p) for p in (root/'inputs').rglob('*') if p.is_file()})
    run.write(root/'source_manifest.json',manifest)


def setup(tmp):
    root=tmp/'capsule';inp=root/'inputs';inp.mkdir(parents=True)
    pixels=np.full((90,90,3),128,dtype=np.uint8)
    for y in range(60,90):
        for x in range(30,60):pixels[y,x]=[20 if (x+y)%2 else 240]*3
    image=Image.fromarray(pixels);content=run.old.oc.png(image);h=run.sha(content)
    head=dict(model_id=run.old.probe.MODEL,model_revision=run.old.probe.REVISION,feature=run.old.probe.FEATURE,
        gold_scored=False,selection_uses=['fit','calibration'],mean=[0,0],scale=[1,1],coef=[10,0],intercept=-2,threshold=.5)
    run.write(inp/'head.json',head)
    rows=[];identity=[];refs={};labels=[];selected={};ref='glass-insulator/train/good/ref.png'
    identity.append(dict(role='reference',member=ref,sha256=h))
    with ZipFile(tmp/'sup.zip','w') as z:
        for i in range(4):
            core='c'+str(i);member='defect_supervised/glass-insulator/train/'+core+'.png';z.writestr(member,content)
            rows.append(dict(core_id=core,id=core+'-q3',image=member,asset_category='glass-insulator',
                protocol=run.old.prior.PROTOCOL,question_type=run.old.prior.agent.Q3,label_basis='dev_image_label_only',
                anomaly_gt='normal' if i==1 else 'anomalous',defect_gt='normal' if i==1 else 'missing-cap',
                regions_gt=[] if i==1 else None))
            identity.append(dict(role='target',member=member,sha256=h));refs[member]=ref
            if i!=1:labels.append(dict(core_id=core,image=member,image_sha256=h,regions_gt=None if i==3 else [8],
                normalized_bboxes_xywh=[] if i==3 else [[.4,.8,.1,.1]]))
            requests=[('original',0)]+([('blur',1)] if i==0 else [])
            for kind,cell in requests:
                im=image if kind=='original' else run.old.oc.masked(image,cell,kind)
                key=core+'/'+kind+'/'+str(cell);p=inp/'features'/(run.digest(key)+'.npy');p.parent.mkdir(exist_ok=True)
                np.save(p,fixture_feature(im),allow_pickle=False)
                selected[key]=dict(id=key,core_id=core,kind=kind,cell=cell,image_sha256=h,pixel_sha256=run.sha(im.tobytes()),
                    feature_path=p.relative_to(root).as_posix(),feature_sha256=run.old.probe.sha_file(p))
    data=dict(protocol=run.old.oc.PROTOCOL,source_run_sha256='fixture',rows=rows,cached_direct={r['core_id']:[] for r in rows},
        cached_cue_review={r['core_id']:[] for r in rows},experts={},references=refs,image_manifest=identity,
        detector_sha256=run.old.probe.sha_file(inp/'head.json'))
    run.write(inp/'dataset.json',data);run.write(inp/'evaluation_gt.json',labels);run.write(inp/'seeds.json',selected)
    x=fixture_feature(image)
    tools={c:dict(score=run.old.score_feature(head,x)['score'],positive=True,error='') for c in ['c0','c1','c2','c3']}
    baseline={c:dict(present=True,regions=[8],source='frozen_missing_expert',location_abstained=False,error='') for c in tools}
    maps={c:dict(proposal=dict(agreement_rank=[0.,1.,2.,3.,4.,5.,6.,7.,8.])) for c in tools}
    run.write(inp/'tools.json',tools);run.write(inp/'baseline.json',baseline);run.write(inp/'old_maps.json',maps)
    seal(root)
    out=tmp/(run.PREFIX+'_fixture')
    args=['--capsule-dir',str(root),'--sup-zip',str(tmp/'sup.zip'),'--out-dir',str(out),'--n-boot','30']
    return root,out,args,head


class FixtureAttributor:
    def __init__(self,head,*,fail=False,interrupt=None,bad_canary=False):
        self.head=head;self.fail=fail;self.interrupt=interrupt;self.bad_canary=bad_canary
        self.stats=dict(forward_passes=0,backward_passes=0,model_loads=0);self.calls=0
    def feature(self,im):
        self.stats['forward_passes']+=1
        return run.old.oc.vector([1,5],2) if self.bad_canary else fixture_feature(im)
    def attribute(self,im,expected):
        self.calls+=1
        if self.interrupt==self.calls:raise KeyboardInterrupt()
        if self.fail:raise RuntimeError('fixture attribution failure')
        self.stats['forward_passes']+=67;self.stats['backward_passes']+=64
        raw={};diag={};target=run.old.score_feature(self.head,expected)['logit']
        for kind in ('blur','mean'):
            a=np.zeros((518,518),np.float32);a[28:42,28:42]=3;a[420:434,252:266]=4
            total=float(a.astype(np.float64).sum());raw[kind]=a
            diag[kind]=dict(baseline_logit=target-total,target_minus_baseline=total,attribution_sum=total,
                completeness_residual=0.,relative_residual=0.,completeness_warning=False,baseline_input_sha256='fixture')
        return dict(raw_maps=raw,target_feature=expected.copy(),geometry=ig.geometry(im.size),target_logit=target,
                    diagnostics=diag,original_input_sha256=run.sha(im.tobytes()),forward_passes=67,backward_passes=64)


def test_ig_linear_closed_form_and_completeness():
    x=torch.arange(48,dtype=torch.float64).reshape(1,3,4,4)/10;b=torch.ones_like(x)*.3
    coef=torch.linspace(-2,3,48,dtype=torch.float64).reshape_as(x)
    forward=lambda p:(p*coef).sum()+5
    a=ig.integrate(forward,x,b,32)
    assert torch.allclose(a,(x-b)*coef,atol=1e-12,rtol=1e-12)
    assert abs(float(a.sum()-(forward(x)-forward(b))))<1e-11
    assert not a.requires_grad


def test_ig_cubic_midpoint_converges_and_model_weights_stay_frozen():
    x=torch.tensor([[.3,.7,1.2]],dtype=torch.float64);b=torch.zeros_like(x)
    forward=lambda p:(p**3).sum()
    coarse=ig.integrate(forward,x,b,4);fine=ig.integrate(forward,x,b,32)
    truth=x**3
    assert torch.max(abs(fine-truth))<torch.max(abs(coarse-truth))
    assert torch.allclose(fine,truth,atol=.001,rtol=.001)
    with pytest.raises(ValueError):ig.integrate(forward,x,b,0)
    with pytest.raises(ValueError):ig.integrate(lambda p:p.sum()*float('nan'),x,b,32)


def test_native_attribute_math_with_tiny_random_dinov2_no_pretrained_weights(monkeypatch):
    from transformers import Dinov2Config,Dinov2Model
    torch.manual_seed(42);torch.set_num_threads(1)
    model=Dinov2Model(Dinov2Config(hidden_size=12,num_hidden_layers=1,num_attention_heads=3,
        intermediate_size=24,image_size=28,patch_size=14)).eval();model.requires_grad_(False)
    native=ig.NativeAttributor.__new__(ig.NativeAttributor);native.torch=torch
    native.engine=SimpleNamespace(model=model);native.head=dict(intercept=.2)
    native.mean=torch.zeros(12,dtype=torch.float64);native.scale=torch.ones(12,dtype=torch.float64)
    native.coef=torch.linspace(-1,1,12,dtype=torch.float64);native.stats=__import__('collections').Counter()
    def tensor(im):
        t=torch.from_numpy(np.array(im).copy()).permute(2,0,1).unsqueeze(0).float()/255
        return t,dict(original_size=list(im.size),side=28,left=0,top=0,tensor_size=28,patch_size=14)
    native.tensor=tensor
    image=Image.fromarray(np.arange(28*28*3,dtype=np.uint8).reshape(28,28,3))
    expected=native.feature(image);before={n:p.clone() for n,p in model.named_parameters()}
    monkeypatch.setitem(ig.CONFIG,'steps',2)
    result=native.attribute(image,expected)
    assert result['forward_passes']==7 and result['backward_passes']==4
    assert all(a.shape==(28,28) and a.dtype==np.float32 and np.isfinite(a).all() for a in result['raw_maps'].values())
    assert all(torch.equal(before[n],p) and p.grad is None and not p.requires_grad for n,p in model.named_parameters())
    for kind,d in result['diagnostics'].items():
        assert np.isclose(result['raw_maps'][kind].astype(np.float64).sum()-d['target_minus_baseline'],d['completeness_residual'])


def test_padding_is_removed_and_coordinates_restore_to_original_frame():
    g=ig.geometry((100,300));a=np.ones((518,518),np.float32)
    score,x,y=ig.patch_map(a,g)
    # Narrow portrait has 100/300 of the square; padded columns contribute no mass.
    valid_columns=((np.arange(518)+.5)*300/518-100>=0)&((np.arange(518)+.5)*300/518-100<100)
    assert score.sum()==valid_columns.sum()*518
    assert ((0<=x)&(x<1)).all() and ((0<=y)&(y<1)).all()
    assert np.count_nonzero(score.sum(0))<37
    for p in [(1/3,1/3),(2/3,2/3),(.5,.5)]:assert ig.cell_of(*p)==5
    assert ig.cell_of(0,0)==1 and ig.cell_of(1,1)==9


def test_components_find_separate_sites_and_deduplicate_cells():
    a=np.zeros((37,37));a[2,2]=3;a[30,18]=4;a[31,18]=4
    xs=ys=(np.arange(37)+.5)/37
    cs=ig.components(a,xs,ys);assert [c['cell'] for c in cs]==[8,1]
    raw={k:np.repeat(np.repeat(a.astype(np.float32),14,0),14,1) for k in ('blur','mean')}
    p,_=ig.proposals(raw,ig.geometry((90,90)))
    assert p['consensus_top1']==[8] and p['consensus_multi']==[1,8]
    assert ig.components(np.zeros((37,37)),xs,ys)==[]
    disjoint=copy.deepcopy(raw);disjoint['mean']=np.zeros((518,518),np.float32);disjoint['mean'][100:114,400:414]=1
    assert ig.proposals(disjoint,ig.geometry((90,90)))[0]['consensus_multi'] is None


def test_count_controls_and_selector_are_label_blind():
    old_map=dict(proposal=dict(agreement_rank=list(range(9))))
    for k in range(4):
        out=ig.matched('core',k,old_map)
        assert all(len(v or [])==k for v in out.values())
        assert out==ig.matched('core',k,old_map)
    source=inspect.getsource(run.assemble)
    assert all(word not in source for word in ('anomaly_gt','defect_gt','regions_gt','boxes_xywh'))


def test_complete_presence_frozen_resume_has_no_model_load(tmp_path):
    root,out,args,head=setup(tmp_path);fixture=FixtureAttributor(head)
    assert run.main(args,attributor=fixture)==0 and fixture.calls==4
    p=run.read(out/'predictions.json');baseline=run.read(root/'inputs/baseline.json')
    assert all(p[policy][c]['present']==baseline[c]['present'] for policy in run.POLICIES for c in baseline)
    assert p['ig_consensus_multi']['c0']['regions']==[1,8]
    assert len(p['occlusion_matched_k']['c0']['regions'])==2
    assert run.read(out/'run_metadata.json')['forward_passes']==270
    again=FixtureAttributor(head);again.feature=lambda _:(_ for _ in ()).throw(AssertionError('resume model load'))
    assert run.main(args,attributor=again)==0 and again.calls==0
    m=run.read(out/'run_metadata.json');assert m['forward_passes']==m['backward_passes']==m['new_attributions']==0
    assert m['resumed_attributions']==4 and m['resumed_canaries']==2


def test_gt_change_does_not_change_routes_or_predictions(tmp_path):
    root,out,args,head=setup(tmp_path);assert run.main(args,attributor=FixtureAttributor(head))==0
    before=run.read(out/'predictions.json');gt=run.read(root/'inputs/evaluation_gt.json')
    for g in gt:
        if g['regions_gt'] is not None:g.update(regions_gt=[1],normalized_bboxes_xywh=[[.1,.1,.1,.1]])
    run.write(root/'inputs/evaluation_gt.json',gt);seal(root)
    args[args.index('--out-dir')+1]=str(tmp_path/(run.PREFIX+'_newlabels'))
    assert run.main(args,attributor=FixtureAttributor(head))==0
    assert run.read(Path(args[args.index('--out-dir')+1])/'predictions.json')==before


def test_failure_and_canary_blocking_preserve_positive_presence(tmp_path):
    root,out,args,head=setup(tmp_path);failed=FixtureAttributor(head,bad_canary=True)
    assert run.main(args,attributor=failed)==1 and failed.calls==0
    assert run.read(out/'paired.json')==[]
    p=run.read(out/'predictions.json');assert p['ig_consensus_multi']['c0']['present'] is True
    assert p['ig_consensus_multi']['c0']['regions'] is None
    assert run.main(args,attributor=FixtureAttributor(head,fail=True))==1
    assert not run.read(out/'run_status.json')['complete']
    assert run.main(args,attributor=FixtureAttributor(head))==0
    assert list((out/'attempt_history').glob('*.json'))


def test_interrupt_resume_keeps_successful_maps(tmp_path):
    root,out,args,head=setup(tmp_path)
    with pytest.raises(KeyboardInterrupt):run.main(args,attributor=FixtureAttributor(head,interrupt=2))
    before=run.read(out/'attributions.json');assert len(before)==1
    assert run.main(args,attributor=FixtureAttributor(head))==0
    assert run.read(out/'attributions.json')['c0']==before['c0']
    assert run.read(out/'run_metadata.json')['new_attributions']==3


@pytest.mark.parametrize('kind',['source','request','map','target','proposal','diagnostic','canary'])
def test_cache_corruption_rejected(tmp_path,kind):
    root,out,args,head=setup(tmp_path);assert run.main(args,attributor=FixtureAttributor(head))==0
    if kind=='source':(root/'scripts/missing_ig_location.py').write_text('changed')
    elif kind in ('map','target','canary'):
        directory={'map':'signed_maps','target':'target_features','canary':'canary_features'}[kind]
        next((out/directory).glob('*.npy')).write_bytes(b'bad')
    else:
        p=out/'attributions.json';d=run.read(p);rec=next(iter(d.values()))
        if kind=='request':rec['image_sha256']='bad'
        elif kind=='proposal':rec['proposals']['consensus_multi']=[9]
        else:rec['diagnostics']['blur']['attribution_sum']+=1
        run.write(p,d)
    fixture=FixtureAttributor(head)
    with pytest.raises((ValueError,FileNotFoundError)):run.main(args,attributor=fixture)
    assert fixture.calls==0


def test_zero_consensus_abstains_with_presence_and_no_forced_extra_cells():
    b={'x':dict(present=True,regions=[5],source='old',error='',location_abstained=False)}
    b['normal']=dict(present=False,regions=[],source='cached_lvlm',error='',location_abstained=False)
    b['invalid']=dict(present=None,regions=None,source='cached_lvlm',error='',location_abstained=False)
    proposal={m:None for m in ('blur_top1','mean_top1','consensus_top1','consensus_multi')}
    p=run.assemble(b,{'x':dict(error='',proposals=proposal)},{'x':{}})
    assert all(p[k]['x']['present'] is True for k in run.POLICIES)
    assert all(p[k]['x']['regions'] is None for k in run.POLICIES[1:])
    assert all(p[k]['normal']==b['normal'] and p[k]['invalid']==b['invalid'] for k in run.POLICIES)


def test_completeness_warning_is_recorded_not_hidden_or_filtered(tmp_path):
    root,out,args,head=setup(tmp_path)
    class Warn(FixtureAttributor):
        def attribute(self,im,expected):
            result=super().attribute(im,expected)
            for d in result['diagnostics'].values():
                d.update(baseline_logit=result['target_logit'],target_minus_baseline=0.,
                         completeness_residual=d['attribution_sum'],relative_residual=d['attribution_sum']/1e-12,completeness_warning=True)
            return result
    assert run.main(args,attributor=Warn(head))==0
    assert run.read(out/'run_status.json')['completeness_warnings']==8
    assert len(run.read(out/'paired.json'))==16


def test_changed_settings_and_unknown_output_are_rejected(tmp_path):
    root,out,args,head=setup(tmp_path);assert run.main(args,attributor=FixtureAttributor(head))==0
    with pytest.raises(ValueError,match='changed source/input/software'):run.main(args[:-1]+['31'],attributor=FixtureAttributor(head))
    p=tmp_path/(run.PREFIX+'_unknown');p.mkdir();(p/'keep.txt').write_text('keep')
    args[args.index('--out-dir')+1]=str(p)
    with pytest.raises(ValueError,match='unknown nonempty'):run.main(args,attributor=FixtureAttributor(head))
    assert (p/'keep.txt').read_text()=='keep'

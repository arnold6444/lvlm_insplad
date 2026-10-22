import ast
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_missing_expert_direct_dev as run
import build_missing_expert_direct_notebook as build


def feature(im):
    return run.oc.vector([np.asarray(im,dtype=float)[:,:,0].std()/10,1],2)


def seal(root):
    manifest={name:run.probe.sha_file(run.ROOT/name) for name in run.prior.shared.code_hashes()}
    manifest.update({p.relative_to(root).as_posix():run.probe.sha_file(p) for p in (root/'inputs').rglob('*') if p.is_file()})
    for name in run.prior.shared.code_hashes():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((run.ROOT/name).read_bytes())
    run.write(root/'source_manifest.json',manifest)


def setup(tmp):
    root=tmp/'capsule'; inp=root/'inputs'; inp.mkdir(parents=True)
    pixels=np.full((90,90,3),128,dtype=np.uint8)
    for y in range(60,90):
        for x in range(30,60):pixels[y,x]=[20 if (x+y)%2 else 240]*3
    image=Image.fromarray(pixels); content=run.oc.png(image); sha=run.sha(content)
    head=dict(model_id=run.probe.MODEL,model_revision=run.probe.REVISION,feature=run.probe.FEATURE,
              gold_scored=False,selection_uses=['fit','calibration'],mean=[0,0],scale=[1,1],coef=[10,0],intercept=-2,threshold=.5)
    run.write(inp/'frozen_detector.json',head)
    rows=[]; manifest=[];refs={};first={};labels=[];seeds={}
    ref='glass-insulator/train/good/ref.png'
    manifest.append(dict(role='reference',member=ref,sha256=sha))
    with ZipFile(tmp/'sup.zip','w') as z:
        for i in range(4):
            core='c'+str(i);member='defect_supervised/glass-insulator/train/'+core+'.png';z.writestr(member,content)
            rows.append(dict(core_id=core,id=core+'-q3',image=member,asset_category='glass-insulator',
                protocol=run.prior.PROTOCOL,question_type=run.prior.agent.Q3,label_basis='dev_image_label_only',
                anomaly_gt='normal' if i==1 else 'anomalous',defect_gt='normal' if i==1 else 'missing-cap',
                regions_gt=[] if i==1 else None))
            refs[member]=ref;manifest.append(dict(role='target',member=member,sha256=sha))
            first[core]=[] if i<2 else [5] if i==2 else None
            if i!=1: labels.append(dict(core_id=core,image=member,image_sha256=sha,
                regions_gt=None if i==3 else [8],normalized_bboxes_xywh=[] if i==3 else [[.4,.8,.1,.1]]))
            requests=[('original',0)] if i<3 else []
            if i==0: requests += [(k,c) for k in run.oc.CONFIG['replacements'] for c in range(1,10)]
            for kind,cell in requests:
                im=image if kind=='original' else run.oc.masked(image,cell,kind);x=feature(im)
                key=run.request(core,kind,cell);path=inp/'features'/(run.digest(key)+'.npy');path.parent.mkdir(exist_ok=True)
                np.save(path,x,allow_pickle=False)
                seeds[key]=dict(id=key,core_id=core,kind=kind,cell=cell,size=list(im.size),pixel_sha256=run.pixel_sha(im),
                    image_sha256=sha,feature_path=path.relative_to(root).as_posix(),feature_sha256=run.probe.sha_file(path),
                    **run.score_feature(head,x))
    data=dict(protocol=run.oc.PROTOCOL,source_run_sha256='fixture',rows=rows,cached_direct=first,cached_cue_review=first,
        experts={},references=refs,image_manifest=manifest,detector_sha256=run.probe.sha_file(inp/'frozen_detector.json'))
    run.write(inp/'dataset.json',data);run.write(inp/'evaluation_gt.json',labels);run.write(inp/'feature_seeds.json',seeds)
    seal(root)
    args=['--capsule-dir',str(root),'--sup-zip',str(tmp/'sup.zip'),'--out-dir',str(tmp/(run.PREFIX+'_fixture')),'--n-boot','30']
    return root,tmp/(run.PREFIX+'_fixture'),args


def test_direct_routes_are_gt_blind_and_prior_invalid_is_replaced_explicitly():
    rows=[dict(core_id='x',asset_category=run.probe.ASSET)]
    tools={'x':dict(positive=True,error='')};maps={'x':dict(error='',proposal={'regions':[8,2]})}
    for baseline in ([],None,[3]):
        a=run.choose_predictions(rows,{'x':baseline},tools,maps)
        assert a['all_positive_top1']['x']['regions']==[8]
        assert a['all_positive_top1']['x']['present'] is True
        assert a['same_route_top1']['x']['regions']==([8] if baseline==[] else baseline)
        assert a['cached_lvlm']['x']['regions']==baseline
    # The selector requires only core and externally supplied asset, no labels/boxes/paths.
    code=ast.parse(__import__('inspect').getsource(run.choose_predictions))
    strings={n.value for n in ast.walk(code) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
    assert not {'anomaly_gt','defect_gt','regions_gt','image'} & strings


def test_location_abstention_preserves_presence_and_unknown_is_not_normal():
    rows=[dict(core_id='x',asset_category=run.probe.ASSET,anomaly_gt='anomalous',regions_gt=[4])]
    p=run.choose_predictions(rows,{'x':[]},{'x':dict(positive=True,error='')},{'x':dict(error='',proposal={'regions':[]})})
    pred=p['all_positive_top1']['x'];assert pred['present'] is True and pred['regions'] is None and pred['location_abstained']
    s=run.metric(rows,p['all_positive_top1'])
    assert s['presence']['tp']==1 and s['location']['exact_positive']==0 and s['location']['region_fn']==1
    rows[0]['regions_gt']=None;s=run.metric(rows,p['all_positive_top1'])
    assert s['presence']['tp']==1 and s['location']['spatial_known_anomalies']==0


def test_complete_resume_zero_new_and_matched_controls(tmp_path):
    root,out,args=setup(tmp_path);calls=[]
    assert run.main(args,extractor=lambda im:(calls.append(im.size) or feature(im)))==0
    status=run.read(out/'run_status.json');meta=run.read(out/'run_metadata.json')
    assert len(calls)==57 and status['new_dino_features']==55 and status['new_canary_features']==2
    assert meta['reused_source_features']==21 and status['lvlm_calls']==0
    p=run.read(out/'predictions.json');s=run.read(out/'metrics.json')
    assert p['cached_lvlm']['c3']['present'] is None
    assert p['all_positive_top1']['c3']['present'] is True
    assert p['same_route_top1']['c3']['present'] is None
    assert s['all_positive_top1']['presence']['tp']==3 and s['all_positive_top1']['location']['exact_positive']==2
    assert p['all_positive_center']['c0']['regions']==[5]
    calls.clear()
    assert run.main(args,extractor=lambda im:(_ for _ in ()).throw(AssertionError('resume loaded model')))==0
    assert not calls and run.read(out/'run_metadata.json')['resumed_features']==76
    assert run.read(out/'run_metadata.json')['resumed_canaries']==2


def test_label_change_does_not_change_predictions(tmp_path):
    root,out,args=setup(tmp_path);assert run.main(args,extractor=feature)==0
    before=run.read(out/'predictions.json')
    gt=run.read(root/'inputs/evaluation_gt.json')
    for r in gt:
        if r['regions_gt'] is not None:r.update(regions_gt=[1],normalized_bboxes_xywh=[[.1,.1,.1,.1]])
    run.write(root/'inputs/evaluation_gt.json',gt);seal(root)
    args[args.index('--out-dir')+1]=str(tmp_path/(run.PREFIX+'_newlabels'))
    assert run.main(args,extractor=feature)==0
    assert run.read(Path(args[args.index('--out-dir')+1])/'predictions.json')==before


@pytest.mark.parametrize('which',['seed','source','mask','output','canary','score'])
def test_tampering_is_rejected_before_new_features(tmp_path,which):
    root,out,args=setup(tmp_path)
    if which in ('output','canary','score'):assert run.main(args,extractor=feature)==0
    if which=='seed':
        p=next((root/'inputs/features').glob('*.npy'));p.write_bytes(b'corrupt')
    elif which=='source': (root/'scripts/run_missing_expert_direct_dev.py').write_text('changed')
    elif which=='mask':
        s=run.read(root/'inputs/feature_seeds.json');s[next(k for k in s if '/blur/' in k)]['pixel_sha256']='bad'
        run.write(root/'inputs/feature_seeds.json',s);seal(root)
    elif which in ('output','canary'):
        p=next((out/('feature_cache' if which=='output' else 'canary_features')).glob('*.npy'));p.write_bytes(b'bad')
    else:
        p=out/'features.json';s=run.read(p);s[next(iter(s))]['score']=.123;run.write(p,s)
    with pytest.raises((ValueError,FileNotFoundError)):
        run.main(args,extractor=lambda im:(_ for _ in ()).throw(AssertionError('must fail before inference')))


def test_canary_mismatch_blocks_uncached_features_and_is_retryable(tmp_path):
    root,out,args=setup(tmp_path);calls=[]
    assert run.main(args,extractor=lambda im:(calls.append(1) or run.oc.vector([1,5],2)))==1
    assert len(calls)==1 and not run.read(out/'canary.json')['passed']
    status=run.read(out/'run_status.json');assert status['new_dino_features']==0 and not status['complete']
    assert run.read(out/'paired.json')==[]
    assert run.main(args,extractor=feature)==0
    assert run.read(out/'canary.json')['passed']


def test_interrupt_resume_preserves_successful_canary_and_features(tmp_path):
    root,out,args=setup(tmp_path);calls=[]
    def extract(im):
        calls.append(1)
        if len(calls)==5:raise KeyboardInterrupt()
        return feature(im)
    with pytest.raises(KeyboardInterrupt):run.main(args,extractor=extract)
    before={k:v['feature_sha256'] for k,v in run.read(out/'features.json').items() if not v['error']}
    assert not run.read(out/'run_status.json')['complete']
    assert run.main(args,extractor=feature)==0
    after=run.read(out/'features.json');assert all(after[k]['feature_sha256']==sha for k,sha in before.items())
    assert run.read(out/'run_metadata.json')['new_canary_features']==0


def test_changed_input_refuses_same_output_and_unknown_folder(tmp_path):
    root,out,args=setup(tmp_path);assert run.main(args,extractor=feature)==0
    with pytest.raises(ValueError,match='changed source/input/software'):run.main(args[:-1]+['31'],extractor=feature)
    unknown=tmp_path/(run.PREFIX+'_unknown');unknown.mkdir();(unknown/'keep.txt').write_text('keep')
    args[args.index('--out-dir')+1]=str(unknown)
    with pytest.raises(ValueError,match='unknown nonempty'):run.main(args,extractor=feature)
    assert (unknown/'keep.txt').read_text()=='keep'


def test_new_feature_failure_is_ineligible_then_retries_only_failed_requests(tmp_path):
    root,out,args=setup(tmp_path);calls=[]
    def extract(im):
        calls.append(1)
        if len(calls)>2:raise RuntimeError('fixture new-feature failure')
        return feature(im)
    assert run.main(args,extractor=extract)==1
    assert len(calls)==3 and run.read(out/'run_status.json')['canary_passed']
    assert run.read(out/'paired.json')==[]
    originals={k:v['feature_sha256'] for k,v in run.read(out/'features.json').items() if not v['error']}
    assert run.main(args,extractor=feature)==0
    after=run.read(out/'features.json')
    assert all(after[k]['feature_sha256']==v for k,v in originals.items())
    assert list((out/'attempt_history').glob('*.json'))


def test_bootstrap_and_invalid_presence_policy():
    rows=[dict(core_id='a',anomaly_gt='anomalous',regions_gt=[2]),dict(core_id='b',anomaly_gt='normal',regions_gt=[])]
    pred={'a':dict(present=None,regions=None,location_abstained=False),'b':dict(present=None,regions=None,location_abstained=False)}
    m=run.metric(rows,pred);assert m['presence']['fn']==1 and m['presence']['fp']==1
    finals={p:pred for p in run.POLICIES}
    ci=run.paired(rows,finals,20);assert len(ci)==20 and all(r['delta']==r['ci_low']==r['ci_high']==0 for r in ci)

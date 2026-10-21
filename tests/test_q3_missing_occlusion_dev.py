import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_q3_missing_occlusion_dev as run


def feature(im):
    return run.vector([np.asarray(im,dtype=float)[:,:,0].std()/10,1],2)


def setup(tmp):
    inp=tmp/'inputs';inp.mkdir();(inp/'original_features').mkdir()
    image=Image.new('RGB',(90,90),(128,128,128));pixels=np.asarray(image).copy()
    for y in range(60,90):
        for x in range(30,60):pixels[y,x]=[20 if (x+y)%2 else 240]*3
    image=Image.fromarray(pixels);payload=run.png(image);sha=run.sha(payload)
    head={'model_id':run.prior.probe.MODEL,'model_revision':run.prior.probe.REVISION,'feature':run.prior.probe.FEATURE,
        'gold_scored':False,'selection_uses':['fit','calibration'],'mean':[0,0],'scale':[1,1],
        'coef':[10,0],'intercept':-2,'threshold':.5}
    detector=inp/'frozen_detector.json';detector.write_text(json.dumps(head),encoding='utf-8')
    rows=[];experts={};manifest=[];refs={};first={};review={}
    ref='glass-insulator/train/good/ref.png'
    with ZipFile(tmp/'sup.zip','w') as sup,ZipFile(tmp/'unsup.zip','w') as unsup:
        unsup.writestr(ref,payload);manifest.append({'role':'reference','member':ref,'sha256':sha})
        for i in range(4):
            core=f'c{i}';member=f'defect_supervised/glass-insulator/train/c{i}.png';sup.writestr(member,payload)
            manifest.append({'role':'target','member':member,'sha256':sha});refs[member]=ref
            rows.append(dict(core_id=core,id=core+'-q3-multiregion-dev',image=member,asset_category='glass-insulator',
                anomaly_gt='normal' if i==1 else 'anomalous',defect_gt='normal' if i==1 else 'missing-cap',
                regions_gt=[] if i==1 else None,protocol=run.prior.PROTOCOL,question_type=run.prior.agent.Q3,label_basis='dev_image_label_only'))
            first[core]=[] if i<2 else [5] if i==2 else None;review[core]=first[core]
            f=inp/'original_features'/(core+'.npy');np.save(f,feature(image),allow_pickle=False)
            experts[core]={'error':'','positive':True,'image':member,'image_sha256':sha,'feature_sha256':run.prior.probe.sha_file(f),
                'score':float(run.prior.probe.predict(head,feature(image)[None,:])[0])}
    data=dict(protocol=run.PROTOCOL,source_run_sha256='fixture',rows=rows,cached_direct=first,cached_cue_review=review,
        experts=experts,references=refs,image_manifest=manifest,detector_sha256=run.prior.probe.sha_file(detector))
    dataset=inp/'dataset.json';dataset.write_text(json.dumps(data),encoding='utf-8')
    out=tmp/(run.PREFIX+'_fixture');args=['--dataset',str(dataset),'--detector',str(detector),'--sup-zip',str(tmp/'sup.zip'),
        '--unsup-zip',str(tmp/'unsup.zip'),'--out-dir',str(out),'--n-boot','30']
    return data,args,out


def execute(args,features,calls,*,invalid=False,feature_error=False,infer_error=False,interrupt_feature=None,interrupt_call=None):
    def extractor(im):
        features.append(im.size)
        if feature_error:raise RuntimeError('fixture feature error')
        if interrupt_feature==len(features):raise KeyboardInterrupt()
        return feature(im)
    def infer(adapter,images,prompt):
        calls.append(images[-1].stem)
        assert len(images)==3 and adapter['image_tile_budgets']==[3,3,2]
        assert 'defect_supervised/' not in prompt and 'anomaly_gt' not in prompt and 'Q2' not in prompt
        if interrupt_call==len(calls):raise KeyboardInterrupt()
        if infer_error and 'occlusion_crop' in images[-1].name:raise RuntimeError('fixture inference error')
        if invalid and 'occlusion_crop' in images[-1].name:return 'explaining without JSON'
        return json.dumps({'regions':[8] if 'occlusion_crop' in images[-1].name else []})
    return run.main(args,extractor=extractor,loader=lambda _: {'kind':'internvl_chat'},infer=infer)


def test_grid_odd_dimensions_covers_exact_original_and_masks_only_one_cell():
    size=(101,67);cover=np.zeros(size[::-1],int)
    im=Image.fromarray(np.arange(101*67*3,dtype=np.uint8).reshape(67,101,3))
    for cell in range(1,10):
        l,t,r,b=run.box(size,cell);cover[t:b,l:r]+=1
        for kind in run.CONFIG['replacements']:
            changed=np.any(np.asarray(run.masked(im,cell,kind))!=np.asarray(im),axis=2)
            changed[t:b,l:r]=False;assert not changed.any()
    assert (cover==1).all()
    for bad in [0,10,True,1.0]:
        with pytest.raises(ValueError):run.box(size,bad)


def test_signed_logit_consensus_and_abstention_not_saturated_probability():
    occluded={'blur':[21,22,20,18,19,20,20,15,20],'local_mean':[19,21,20,19,18,20,20,17,20]}
    assert run.candidates(20,occluded)['regions']==[8,4]
    assert run.candidates(20,{'blur':[21]*9,'local_mean':[19]*9})['abstained']
    assert run.candidates(20,{'blur':[20]*9,'local_mean':[20]*9})['regions']==[]
    with pytest.raises(ValueError):run.candidates(20,{'blur':[float('nan')]*9,'local_mean':[20]*9})


def test_matched_review_original_coordinates_gt_independence_and_resume(tmp_path):
    data,args,out=setup(tmp_path);features=[];calls=[]
    source_before={p:Path(p).read_bytes() for p in [args[1],args[3]]}
    assert execute(args,features,calls)==0 and len(features)==36 and len(calls)==4
    maps=run.read(out/'occlusion_maps.json');assert all(m['proposal']['regions']==[8] for m in maps.values())
    pred=run.read(out/'predictions.json')
    assert pred['occlusion_crop']['c0']==[8] and pred['fixed_crop']['c0']==[]
    assert all(pred[p]['c2']==[5] and pred[p]['c3'] is None for p in run.POLICIES)
    assert run.read(out/'metrics_occlusion_crop.json')['overall']['fp']==1
    assert run.read(out/'metrics_cached_direct.json')['overall']['invalid_anomalous']==1
    assert all(run.read(out/'run_status.json')['eligible'].values())
    assert all(Path(p).read_bytes()==b for p,b in source_before.items())
    for row in data['rows']:
        mutated=dict(row,anomaly_gt='malicious-label',defect_gt='different',regions_gt=[9])
        assert run.routed(mutated,data)==run.routed(row,data)
    features.clear();calls.clear();assert execute(args,features,calls)==0 and not features and not calls


def test_no_agreed_proposal_skips_both_new_arms(tmp_path):
    data,args,out=setup(tmp_path);calls=[]
    # A vector with a greater missing logit than the original gives negative drops.
    assert run.main(args,extractor=lambda _:np.array([1,0],np.float32),loader=lambda _: {},infer=lambda *x:calls.append(x))==0
    assert not calls and run.read(out/'run_status.json')['proposal_abstentions']==2
    pred=run.read(out/'predictions.json');assert pred['fixed_crop']==pred['occlusion_crop']==data['cached_direct']


def test_invalid_kept_penalized_and_not_correctness_retried(tmp_path):
    _,args,out=setup(tmp_path);features=[];calls=[]
    assert execute(args,features,calls,invalid=True)==0
    assert run.read(out/'run_status.json')['invalid_new_outputs']==2
    report=run.read(out/'metrics_occlusion_crop.json')['overall']
    assert report['invalid_anomalous']==2 and report['invalid_normal']==1
    calls.clear();features.clear();assert execute(args,features,calls)==0 and not calls and not features


def test_feature_error_excludes_new_arms_then_recovers(tmp_path):
    _,args,out=setup(tmp_path);features=[];calls=[]
    assert execute(args,features,calls,feature_error=True)==1 and not calls
    s=run.read(out/'run_status.json');assert s['eligible']['cached_direct'] and not s['eligible']['fixed_crop']
    features.clear();assert execute(args,features,calls)==0 and len(features)==36 and len(calls)==4
    assert list((out/'attempt_history').glob('*.json'))


def test_inference_error_retries_only_failed_records(tmp_path):
    _,args,out=setup(tmp_path);features=[];calls=[]
    assert execute(args,features,calls,infer_error=True)==1
    assert run.read(out/'run_status.json')['eligible']['fixed_crop']
    calls.clear();features.clear();assert execute(args,features,calls)==0
    assert len(calls)==2 and not features and all('occlusion_crop' in c for c in calls)


@pytest.mark.parametrize('interrupt',['feature','call'])
def test_interruption_resumes_successful_work(tmp_path,interrupt):
    _,args,out=setup(tmp_path);features=[];calls=[]
    with pytest.raises(KeyboardInterrupt):
        execute(args,features,calls,interrupt_feature=10 if interrupt=='feature' else None,interrupt_call=2 if interrupt=='call' else None)
    features.clear();calls.clear();assert execute(args,features,calls)==0
    if interrupt=='feature':assert len(features)==27 and len(calls)==4
    else:assert not features and len(calls)==3


def test_cache_tamper_rejected_before_model_call(tmp_path):
    _,args,out=setup(tmp_path);features=[];calls=[];assert execute(args,features,calls)==0
    f=next((out/'feature_cache').glob('*.npy'));f.write_bytes(b'corrupt')
    features.clear();calls.clear()
    with pytest.raises(ValueError,match='cache corruption'):execute(args,features,calls)
    assert not features and not calls


def test_changed_code_config_and_source_feature_rejected(tmp_path,monkeypatch):
    _,args,out=setup(tmp_path);features=[];calls=[];assert execute(args,features,calls)==0
    monkeypatch.setitem(run.CONFIG,'max_candidates',1)
    with pytest.raises(ValueError,match='changed input/code/model'):execute(args,[],[])


def test_partial_fixture_injection_and_native_head_guard(tmp_path):
    data,args,out=setup(tmp_path)
    with pytest.raises(ValueError,match='all three'):run.main(args,extractor=lambda _:None)
    with pytest.raises(ValueError,match='frozen head'):run.validate(data,run.read(Path(args[3])),data['detector_sha256'],fixture=False)
    data['rows'][0]['protocol']='gold'
    with pytest.raises(ValueError,match='gold or non-dev'):run.validate(data,run.read(Path(args[3])),data['detector_sha256'],fixture=True)


def test_raw_request_tamper_rejected_and_original_feature_checksum(tmp_path):
    _,args,out=setup(tmp_path);assert execute(args,[],[])==0
    p=out/'calls.json';data=run.read(p);key=next(iter(data));data[key]['prompt']='different prompt';p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='changed LVLM request'):execute(args,[],[])
    original=Path(args[1]).parent/'original_features/c0.npy';original.write_bytes(b'changed')
    with pytest.raises(ValueError,match='original feature changed'):execute(args,[],[])


def test_coordinate_orientation_guard(tmp_path):
    im=Image.new('RGB',(60,90));exif=Image.Exif();exif[274]=6
    path=tmp_path/'rotated.jpg';im.save(path,exif=exif)
    with pytest.raises(ValueError,match='EXIF orientation'):run.original_image(path)

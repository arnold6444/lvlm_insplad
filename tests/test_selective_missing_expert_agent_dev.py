import hashlib
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_selective_missing_expert_agent_dev as run
import test_gold_v2_q23_zoom_prompt_dev as fixtures


def setup(tmp):
    rows,first,old_args,_,_=fixtures.setup(tmp,6)
    get=lambda k:Path(old_args[old_args.index('--'+k)+1])
    refs={}
    for row in rows:
        i=int(row['core_id'][1:]);row['asset_category']='glass-insulator' if i<4 else 'vari-grip'
        if i in (1,3):
            row['defect_gt']='missing-cap'
            if row['question_type']==run.agent.Q2:row['answer']='C'
        refs[row['image']]=row['asset_category']+'/train/good/ref.png'
    with ZipFile(get('sup-zip'),'w') as z:
        for r in rows[::2]:z.writestr(r['image'],fixtures.fixtures._png(10+20*int(r['core_id'][1:])))
    with ZipFile(get('unsup-zip'),'w') as z:
        for name in sorted(set(refs.values())):z.writestr(name,fixtures.fixtures._png(200))
    manifest=[]
    for role,path in [('target',get('sup-zip')),('reference',get('unsup-zip'))]:
        with ZipFile(path) as z:
            manifest.extend({'role':role,'member':n,'sha256':hashlib.sha256(z.read(n)).hexdigest()} for n in z.namelist())
    artifact={'model_id':run.probe.MODEL,'model_revision':run.probe.REVISION,'feature':run.probe.FEATURE,
        'gold_scored':False,'selection_uses':['fit','calibration'],'mean':[0,0],'scale':[1,1],
        'coef':[10,-10],'intercept':0,'threshold':.5}
    inp=tmp/'inputs';inp.mkdir()
    detector=inp/'frozen_detector.json';detector.write_text(json.dumps(artifact),encoding='utf-8')
    data={'protocol':run.PROTOCOL,'rows':rows,'q1_fixed':{r['id']:first[r['id']] for r in rows if r['question_type']==run.agent.Q1},
        'references':refs,'image_manifest':manifest,'detector_sha256':run.probe.sha_file(detector)}
    dataset=inp/'dataset.json';dataset.write_text(json.dumps(data),encoding='utf-8')
    out=tmp/'selective_missing_expert_agent_dev_fixture'
    args=['--dataset',str(dataset),'--detector',str(detector),'--sup-zip',str(get('sup-zip')),'--unsup-zip',str(get('unsup-zip')),
        '--out-dir',str(out),'--n-images','6','--n-boot','20']
    return data,args,out


def mocks(calls,features,invalid=False,error=False,interrupt_at=None,feature_error=False):
    def infer(adapter,images,prompt):
        core=Path(images[-1]).stem;qt='q2' if 'Visual definitions' in prompt else 'q3'
        step='agent' if 'separate frozen supervised image classifier' in prompt else 'matched' if 'Second examination' in prompt or 'Re-examine this SAME' in prompt else 'initial'
        calls.append((core,qt,step))
        assert len(images)==2 and adapter['image_tile_budgets']==[3,3]
        if interrupt_at and len(calls)==interrupt_at:raise KeyboardInterrupt()
        if error and len(calls)==1:raise RuntimeError('fixture inference failure')
        if invalid and (core,qt,step)==('c1','q3','agent'):return 'not JSON'
        if qt=='q2':return ('C.' if core in ('c2','c3') or step=='agent' else 'A.')
        return json.dumps({'regions':[1,9] if core=='c2' else [5] if step=='agent' else []})
    def extractor(image):
        i=int(round((image.getpixel((0,0))[0]-10)/20));features.append(i)
        if feature_error:raise RuntimeError('fixture feature failure')
        return np.array([1,0] if i%2 else [0,1],dtype=np.float32)
    return infer,extractor


def execute(args,calls,features,**kwargs):
    infer,extractor=mocks(calls,features,**kwargs)
    return run.main(args,loader=lambda _: {'kind':'internvl_chat'},infer=infer,extractor=extractor)


def test_conditional_integrated_flow_matched_calls_independent_questions_and_resume(tmp_path):
    data,args,out=setup(tmp_path);calls=[];features=[]
    before={p:Path(p).read_bytes() for p in (args[1],args[3])}
    assert execute(args,calls,features)==0
    assert len(calls)==18 and sorted(features)==[0,1,3]
    assert all(c[0] not in ('c0','c2','c4','c5') for c in calls if c[2]!='initial')
    pred=json.loads((out/'predictions.json').read_text())
    assert pred['q12']['selective_agent']['c1-q2']=='C'
    assert pred['q12']['matched_reask']['c1-q2']=='A'
    assert all(pred['q12'][p]['c2-q2']=='C' and pred['q3'][p]['c2']==[1,9] for p in run.POLICIES)
    assert pred['q3']['selective_agent']['c3']==[5]
    assert all(pred['q12'][p][rid]==answer for p in run.POLICIES for rid,answer in data['q1_fixed'].items())
    statuses=json.loads((out/'run_status.json').read_text())['policies']
    assert {s['logical_lvlm_calls'] for s in statuses if s['policy'] in ('selective_agent','matched_reask')}=={15}
    assert all(s['q1_fixed'] and s['eligible_for_comparison'] for s in statuses)
    assert all(Path(p).read_bytes()==b for p,b in before.items())
    for p in run.POLICIES:
        metrics=json.loads((out/('metrics_'+p+'.json')).read_text())
        assert metrics['q3_positive_location_accuracy'] is None and metrics['q3']['overall']['spatial_known_anomalies']==0
    def forbidden(*args):pytest.fail('resume performed new model operation')
    assert run.main(args,loader=forbidden,infer=forbidden,extractor=forbidden)==0
    meta=json.loads((out/'run_metadata.json').read_text())
    assert meta['new_lvlm_calls']==0 and meta['reused_lvlm_calls']==18 and meta['new_expert_features']==0


def test_prompts_and_routes_do_not_read_gt_or_other_question_answers(tmp_path):
    data,_,_=setup(tmp_path)
    row=next(r for r in data['rows'] if r['id']=='c1-q2');r3=run.q3_row(row)
    for r,first in [(row,'A'),(r3,[])]:
        dirty=dict(r,answer='SECRET',defect_gt='SECRET',anomaly_gt='normal',regions_gt=[9],image='SECRET',q1_answer='B',other_answer='SECRET')
        assert run.initial_prompt(r)==run.initial_prompt(dirty)
        assert run.review_prompt(r,first,True)==run.review_prompt(dirty,first,True)
        assert 'SECRET' not in run.review_prompt(dirty,first,True)
    assert run.parse(row,'A.')=='A' and run.parse(row,'A or B') is None
    assert run.parse(r3,'{"regions":[1,9]}')==[1,9]
    assert run.parse(r3,'{"regions":[1,1]}') is None
    assert not run.needs_expert('vari-grip',run.agent.Q2,'A','C')
    assert not run.needs_expert('glass-insulator',run.agent.Q2,'C','C')
    assert not run.needs_expert('glass-insulator',run.agent.Q3,[1])
    assert not run.needs_expert('glass-insulator',run.agent.Q3,None)


def test_invalid_review_is_counted_wrong_retained_and_never_retried(tmp_path):
    _,args,out=setup(tmp_path)
    assert execute(args,[],[],invalid=True)==0
    metrics=json.loads((out/'metrics_selective_agent.json').read_text())
    assert metrics['q3']['overall']['invalid_anomalous']==1
    calls=[];features=[]
    assert execute(args,calls,features)==0 and not calls and not features
    assert json.loads((out/'run_metadata.json').read_text())['invalid_lvlm_outputs']==1


def test_retry_only_execution_errors_and_preserve_attempts(tmp_path):
    _,args,out=setup(tmp_path)
    assert execute(args,[],[],error=True)==1
    calls=[];features=[]
    assert execute(args,calls,features)==0 and len(calls)==1
    assert list((out/'attempt_history').glob('*.json'))


def test_interrupted_run_resumes_without_losing_saved_results(tmp_path):
    _,args,out=setup(tmp_path);calls=[]
    with pytest.raises(KeyboardInterrupt):execute(args,calls,[],interrupt_at=8)
    old=json.loads((out/'calls.json').read_text());assert len(old)==7
    successful=calls[:7];resumed=[]
    assert execute(args,resumed,[])==0
    assert not set(successful)&set(resumed)


@pytest.mark.parametrize('change',['dataset','detector','image','feature','request'])
def test_input_image_feature_and_request_drift_blocked_without_prediction_overwrite(tmp_path,change):
    _,args,out=setup(tmp_path)
    assert execute(args,[],[])==0
    calls_path=out/'calls.json';before=calls_path.read_bytes()
    if change in ('dataset','detector'):
        p=Path(args[args.index('--'+change)+1]);p.write_bytes(p.read_bytes()+b'\n')
    elif change=='image':
        p=Path(args[args.index('--sup-zip')+1])
        with ZipFile(p) as z:members={n:z.read(n) for n in z.namelist()}
        members[next(iter(members))]=fixtures.fixtures._png(99)
        with ZipFile(p,'w') as z:
            for n,b in members.items():z.writestr(n,b)
    elif change=='feature':next((out/'tool_features').glob('*.npy')).write_bytes(b'broken')
    else:
        records=json.loads(before);next(iter(records.values()))['prompt']='tampered'
        calls_path.write_text(json.dumps(records),encoding='utf-8');before=calls_path.read_bytes()
    with pytest.raises(ValueError):execute(args,[],[])
    assert calls_path.read_bytes()==before


def test_expert_failure_excludes_dependent_policies_only(tmp_path):
    _,args,out=setup(tmp_path)
    assert execute(args,[],[],feature_error=True)==1
    states=json.loads((out/'run_status.json').read_text())['policies']
    assert {s['policy'] for s in states if s['eligible_for_comparison']}=={'lvlm_only'}


def test_gold_and_wrong_native_detector_rejected_before_any_model(tmp_path):
    data,args,_=setup(tmp_path)
    art=json.loads(Path(args[3]).read_text())
    with pytest.raises(ValueError,match='detector SHA'):run.validate_inputs(data,art,data['detector_sha256'],False)
    data['rows'][0]['protocol']='gold_v2'
    with pytest.raises(ValueError,match='gold forbidden'):run.validate_inputs(data,art,data['detector_sha256'],True)


def test_model_load_failure_attempted_once_without_expert_fallback(tmp_path):
    _,args,out=setup(tmp_path);attempts=[]
    def fail(_):attempts.append(1);raise RuntimeError('fixture model load failure')
    def forbidden(*args):pytest.fail('unexpected neural operation after failed load')
    assert run.main(args,loader=fail,infer=forbidden,extractor=forbidden)==1
    assert len(attempts)==1
    meta=json.loads((out/'run_metadata.json').read_text())
    assert meta['new_lvlm_calls']==1 and meta['blocked_lvlm_calls']==11 and meta['logical_expert_requests']==0


def test_existing_missing_and_nonempty_locations_never_load_expert(tmp_path):
    _,args,out=setup(tmp_path)
    def infer(adapter,images,prompt):return 'C.' if 'Visual definitions' in prompt else '{"regions":[1,9]}'
    def forbidden(*args):pytest.fail('expert requested although both own-question answers already positive')
    assert run.main(args,loader=lambda _: {'kind':'internvl_chat'},infer=infer,extractor=forbidden)==0
    meta=json.loads((out/'run_metadata.json').read_text())
    assert meta['raw_lvlm_calls']==12 and meta['logical_expert_requests']==meta['raw_expert_images']==0

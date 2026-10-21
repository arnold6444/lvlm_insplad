import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_missing_presence_independent_dev as run
import test_q3_gt_crop_diagnostic_dev as previous


def setup(tmp):
    data,args,old=previous.setup(tmp)
    data.update(experiment_protocol=run.PROTOCOL,source_presence_gt='all20_missing_cap',
                source_result_sha256='fixture',previous_joint_q3={p:{r['core_id']:[] for r in data['rows']} for p in run.ARMS})
    run.write(Path(args[1]),data);out=tmp/(run.PREFIX+'_fixture');args[7]=str(out)
    return data,args,out


def execute(data,args,seen,*,invalid=False,error=False,interrupt=None):
    def infer(adapter,images,prompt):
        seen.append(images[-1].stem)
        if interrupt==len(seen):raise KeyboardInterrupt()
        assert len(images)==3 and adapter['image_tile_budgets']==[3,3,2]
        assert run.crop.Image.open(images[-1]).size==(1024,548)
        assert all(s not in prompt.lower() for s in ['earlier','previous','regions','grid','oracle','anomaly_gt','missing-cap/'])
        if invalid and 'oracle_box' in images[-1].name:return '{"present":0}'
        if error and 'oracle_box' in images[-1].name:raise RuntimeError('fixture failure')
        return '{"present":false}' if 'candidate_grid' in images[-1].name else '{"present":true}'
    return run.main(args,loader=lambda _:{'kind':'internvl_chat'},infer=infer)


@pytest.mark.parametrize('raw,expected',[
    ('{"present":true}',True),('{"present":false}',False),
    (chr(96)*3+'json\n{"present":true}\n'+chr(96)*3,True),
    ('{"present":0}',None),('{"present":"false"}',None),('{"regions":[]}',None),
    ('{"present":true,"why":"yes"}',None),('{"present":true,"present":false}',None),('yes',None),
])
def test_strict_binary_parser(raw,expected):
    assert run.parse(raw) is expected


def test_no_prior_answer_gt_coordinates_or_location_question():
    p=run.prompt()
    assert '{"present":true}' in p and '{"present":false}' in p
    assert all(s not in p.lower() for s in ['earlier','previous','regions','grid','oracle','human'])
    assert 'exposed hardware' in p


def test_binary_metrics_requests_and_zero_call_resume(tmp_path):
    data,args,out=setup(tmp_path);seen=[]
    assert execute(data,args,seen)==0 and len(seen)==6
    assert run.read(out/'metrics_candidate_grid.json')['fn']==2
    score=run.read(out/'metrics_oracle_box.json')
    assert score['tp']==2 and score['missing_recall']==1 and score['normal_fpr'] is None and score['location_accuracy'] is None
    calls=run.read(out/'calls.json')
    assert len({x['prompt'] for x in calls.values()})==1
    assert all(type(x['prediction']) is bool for x in calls.values())
    assert run.read(out/'run_status.json')['tokenizer_gate']=='fixture_unrun'
    seen.clear();assert execute(data,args,seen)==0 and not seen
    assert run.read(out/'run_metadata.json')['reused_lvlm_calls']==6


def test_invalid_stays_fn_and_not_retried(tmp_path):
    data,args,out=setup(tmp_path);seen=[]
    assert execute(data,args,seen,invalid=True)==0
    assert run.read(out/'metrics_oracle_box.json')['invalid_anomalous']==2
    assert run.read(out/'run_status.json')['invalid_new_outputs']==2
    seen.clear();assert execute(data,args,seen)==0 and not seen


def test_failed_records_only_retry_with_history(tmp_path):
    data,args,out=setup(tmp_path);seen=[]
    assert execute(data,args,seen,error=True)==1
    assert not run.read(out/'run_status.json')['eligible']['oracle_box']
    seen.clear();assert execute(data,args,seen)==0 and len(seen)==2
    assert all('oracle_box' in s for s in seen)
    assert len(list((out/'attempt_history').glob('*.json')))==2


def test_interruption_reuses_success(tmp_path):
    data,args,out=setup(tmp_path);seen=[]
    with pytest.raises(KeyboardInterrupt):execute(data,args,seen,interrupt=3)
    assert len(run.read(out/'calls.json'))==2
    seen.clear();assert execute(data,args,seen)==0 and len(seen)==4


@pytest.mark.parametrize('what',['request','parsing','crop','input','config'])
def test_tampering_prevents_new_inference(tmp_path,what,monkeypatch):
    data,args,out=setup(tmp_path);assert execute(data,args,[])==0
    if what in ['request','parsing']:
        c=run.read(out/'calls.json');r=c[next(iter(c))]
        r['prompt' if what=='request' else 'prediction']='changed';run.write(out/'calls.json',c)
    elif what=='crop':next((out/'crops').glob('*.png')).write_bytes(b'bad')
    elif what=='input':
        data['source_result_sha256']='changed';run.write(Path(args[1]),data)
    else:monkeypatch.setitem(run.CONFIG,'max_new_tokens',60)
    seen=[]
    with pytest.raises(ValueError):execute(data,args,seen)
    assert not seen


def test_tokenizer_refusal_stops_before_model(tmp_path,monkeypatch):
    data,args,out=setup(tmp_path)
    base=data['rows'];data['rows']=[]
    for i in range(20):
        row=copy.deepcopy(base[i%2]);row['core_id']=f'c{i}';data['rows'].append(row)
    run.write(Path(args[1]),data)
    snapshot=tmp_path/run.gate.REVISION;snapshot.mkdir();args.extend(['--model-path',str(snapshot)])
    def refuse(*a,**k):raise ValueError('tokenizer mismatch fixture')
    loads=[]
    monkeypatch.setattr(run.gate,'verify',refuse)
    monkeypatch.setattr(run.crop.old.prior.hf,'load_hf_adapter',lambda _:loads.append(1))
    with pytest.raises(ValueError,match='tokenizer mismatch'):run.main(args)
    assert not loads and run.read(out/'run_status.json')['phase']=='tokenizer_gate'


def test_metrics_rejects_int_and_no_normal_claims():
    rows=[{'core_id':'a'},{'core_id':'b'}]
    m=run.metrics(rows,{'a':True,'b':None})
    assert m['tp']==1 and m['fn']==1 and m['invalid_anomalous']==1
    assert m['balanced_accuracy'] is None and m['normal_support']==0
    with pytest.raises(ValueError):run.metrics(rows,{'a':1,'b':False})


def test_wrong_protocol_and_partial_callbacks(tmp_path):
    d,args,_=setup(tmp_path);d['experiment_protocol']='gold'
    with pytest.raises(ValueError):run.validate(d,True)
    with pytest.raises(ValueError,match='both fixture'):run.main(args,loader=lambda _:None)

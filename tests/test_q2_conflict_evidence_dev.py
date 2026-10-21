import hashlib
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_q2_conflict_evidence_dev as run
import test_gold_v2_q23_zoom_prompt_dev as fixtures


def setup(tmp):
    rows,first,old_args,_,_ = fixtures.setup(tmp,6)
    first={r['id']:first[r['id']] for r in rows}
    get=lambda k:Path(old_args[old_args.index('--'+k)+1])
    refs={r['image']:r['retrieved_diffdate'] for r in run.read_csv(get('refs-csv'))}
    manifest=[]
    for role,path in [('target',get('sup-zip')),('reference',get('unsup-zip'))]:
        with ZipFile(path) as z:
            manifest.extend({'role':role,'member':n,'sha256':hashlib.sha256(z.read(n)).hexdigest()} for n in z.namelist())
    review,cues={},{}
    for r in rows:
        if r['question_type']!=run.agent.Q2:continue
        choice={'c1-q2':'B','c3-q2':'A'}.get(r['id'],first[r['id']])
        images=['/old/_images/normal/'+refs[r['image']],'/old/_images/target/'+r['image']]
        common={'question_id':r['id'],'question_type':run.agent.Q2,'model_key':run.MODEL,'images':images,'error':''}
        review[r['id']]={**common,'step':'q2_define','output':choice,'predicted_choice':choice,
            'prompt':run.agent.build_prompt(r,first[r['id']],'define')}
        cues[r['id']]={**common,'step':'q2_cues','output':r['answer'],'predicted_choice':r['answer'],
            'prompt':run.refined.definition_prompt(r,True)}
    data={'rows':rows,'initial':first,'reviewed':review,'cue_records':cues,'references':refs,'image_manifest':manifest}
    source=tmp/'inputs/dataset.json';source.parent.mkdir();source.write_text(json.dumps(data),encoding='utf-8')
    out=tmp/'q2_conflict_evidence_dev_test'
    args=['--dataset',str(source),'--sup-zip',str(get('sup-zip')),'--unsup-zip',str(get('unsup-zip')),
        '--out-dir',str(out),'--n-images','6','--n-boot','10']
    return data,args,out


def fake(calls,invalid=False,error=False,interrupt=False):
    def infer(adapter,images,prompt):
        core=Path(images[-1]).stem
        calls.append((core,prompt))
        assert adapter['image_tile_budgets']==[3,3] and len(images)==2
        if len(calls)==1:
            if interrupt:raise KeyboardInterrupt()
            if error:raise RuntimeError('fixture error')
            if invalid:return 'A or B'
        if 'Candidate condition:' in prompt:
            candidate=prompt.split('Candidate condition: ')[1].split('\n')[0]
            supported=('Rust' in candidate)==(int(core[1:])%2==1)
            text='Supported by visible evidence.' if supported else 'Not supported by visible evidence.'
            return next(k for k,v in re.findall(r'^([AB])\. (.+)$',prompt,re.M) if v==text)
        return 'B' if int(core[1:])%2 else 'A'
    return infer


def execute(args,calls,**kwargs):
    return run.main(args,loader=lambda _: {'kind':'internvl_chat'},infer=fake(calls,**kwargs))


def test_matched_route_q1_fixed_conservative_decision_and_resume(tmp_path):
    data,args,out=setup(tmp_path)
    before=Path(args[1]).read_bytes()
    calls=[]
    assert execute(args,calls)==0 and len(calls)==8
    assert {c[0] for c in calls}=={'c1','c3'}
    decisions=json.loads((out/'decisions.json').read_text())
    assert {r['id']:r['final'] for r in decisions}=={'c1-q2':'B','c3-q2':'B'}
    statuses=json.loads((out/'run_status.json').read_text())['policies']
    assert {s['additional_calls'] for s in statuses if s['policy'] in ('candidate_support','matched_reask')}=={4}
    answers=run.read_csv(out/'answers.csv')
    assert all(a['predicted_choice']==data['initial'][a['id']] for a in answers if a['question_type']==run.agent.Q1)
    assert all(r['question_type']==run.agent.Q2 for r in run.agent.read_jsonl(out/'predictions'/run.MODEL/'steps.jsonl'))
    assert Path(args[1]).read_bytes()==before
    assert not (out/'frozen_config.json').exists()
    assert run.main(args,loader=lambda _:pytest.fail('resume loaded model'),infer=lambda *a:pytest.fail('new call'))==0
    assert json.loads((out/'run_metadata.json').read_text())['reused_calls']==8


@pytest.mark.parametrize('a,b,result',[(True,False,'A'),(False,True,'B'),(True,True,'A'),(False,False,'A'),(None,True,None),(False,None,None)])
def test_resolution(a,b,result):
    assert run.resolve_support('A','B',a,b)==result


def test_prompt_label_independence_and_strict_outputs(tmp_path):
    data,_,_=setup(tmp_path)
    row=next(r for r in data['rows'] if r['id']=='c1-q2')
    task,prompt=run.support_task(row,'A')
    assert run.support_task(dict(row,answer='D',defect_gt='SECRET',image='SECRET',q1_answer='D',q3_answer='H'),'A')==(task | {'answer':'D','defect_gt':'SECRET','image':'SECRET','q1_answer':'D','q3_answer':'H'},prompt)
    assert 'SECRET' not in prompt
    for text in ('A or B','A\nB','The answer is A','{"answer":"A"}'):
        assert run.strict_choice(text,task['options']) is None
    assert run.strict_choice('A',task['options'])=='A'
    with pytest.raises(ValueError):run.support_task(dict(row,question_type=run.agent.Q1),'A')


def test_invalid_is_retained_excluded_and_not_retried(tmp_path):
    _,args,out=setup(tmp_path)
    assert execute(args,[],invalid=True)==1
    initial=(out/'predictions'/run.MODEL/'steps.jsonl').read_bytes()
    calls=[]
    assert execute(args,calls)==1 and not calls
    assert (out/'predictions'/run.MODEL/'steps.jsonl').read_bytes()==initial
    assert len(run.read_csv(out/'leaderboard.csv'))==5
    meta=json.loads((out/'run_metadata.json').read_text())
    assert meta['invalid_tool_answers']==1 and meta['reused_calls']==8


def test_error_retry_and_interruption_preserve_cache(tmp_path):
    _,args,out=setup(tmp_path)
    assert execute(args,[],error=True)==1
    path=out/'predictions'/run.MODEL/'steps.jsonl'
    assert len(run.agent.read_jsonl(path))==8
    with pytest.raises(KeyboardInterrupt):execute(args,[],interrupt=True)
    assert len(run.agent.read_jsonl(path))==8
    calls=[]
    assert execute(args,calls)==0 and len(calls)==1


def test_changed_source_or_image_blocked(tmp_path):
    _,args,out=setup(tmp_path)
    assert execute(args,[])==0
    pred=out/'predictions'/run.MODEL/'steps.jsonl'
    before=pred.read_bytes()
    source=Path(args[1]);original=source.read_bytes();source.write_bytes(original+b'\n')
    with pytest.raises(ValueError,match='inputs/code/model changed'):execute(args,[])
    source.write_bytes(original)
    zpath=Path(args[args.index('--sup-zip')+1])
    with ZipFile(zpath) as z:members={n:z.read(n) for n in z.namelist()}
    members[next(iter(members))]=fixtures.fixtures._png(99)
    with ZipFile(zpath,'w') as z:
        for n,b in members.items():z.writestr(n,b)
    with pytest.raises(ValueError,match='image bytes'):execute(args,[])
    assert pred.read_bytes()==before


def test_agreement_no_load_and_gold_rejected(tmp_path):
    data,args,out=setup(tmp_path)
    for r in data['rows']:
        if r['question_type']==run.agent.Q2:
            rec=data['reviewed'][r['id']];rec['output']=rec['predicted_choice']=data['initial'][r['id']]
    Path(args[1]).write_text(json.dumps(data),encoding='utf-8')
    assert run.main(args,loader=lambda _:pytest.fail('no conflict loaded model'))==0
    assert json.loads((out/'run_metadata.json').read_text())['raw_calls']==0
    data['rows'][0]['protocol']='gold_v2'
    with pytest.raises(ValueError,match='gold_v2_dev'):run.validate_dataset(data)


def test_model_load_failure_single_attempt(tmp_path):
    _,args,out=setup(tmp_path)
    attempts=[]
    def loader(_):
        attempts.append(1)
        raise RuntimeError('failed load')
    assert run.main(args,loader=loader)==1
    assert len(attempts)==1
    meta=json.loads((out/'run_metadata.json').read_text())
    assert meta['blocked_calls']==7 and meta['row_errors']==8


def test_failed_first_reask_blocks_dependent_call_then_recovers_without_repeating_success(tmp_path):
    _,args,out=setup(tmp_path)
    seen=[]
    normal=fake(seen)
    failed=[]
    def infer(adapter,images,prompt):
        if 'This is a second look' in prompt and not failed:
            failed.append(1)
            raise RuntimeError('first reask transient error')
        return normal(adapter,images,prompt)
    assert run.main(args,loader=lambda _: {'kind':'internvl_chat'},infer=infer)==1
    records=run.agent.read_jsonl(out/'predictions'/run.MODEL/'steps.jsonl')
    assert sum(r['error'].startswith('BlockedDependency:') for r in records)==1
    calls=[]
    assert execute(args,calls)==0 and len(calls)==2
    assert len(list((out/'attempt_history').glob('*.json')))==2

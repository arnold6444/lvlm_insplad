import copy
import importlib.util
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import q3_multiregion as q3
import run_q3_multiregion as runner


def inputs():
    master=[]
    specs=[('a','rust'),('b','normal'),('c','bird-nest'),('d','missing-cap')]
    for core,defect in specs:
        row=dict(core_id=core,image=f'asset/val/{defect}/{core}.jpg',asset_category='asset',
                 anomaly_gt='normal' if defect=='normal' else 'anomalous',defect_gt=defect,
                 normal_reference_image='asset/train/good/ref.jpg')
        for kind in ('anomaly_detection','defect_classification'):
            master.append(dict(row,id=f'{core}-{kind}',question_type=kind))
    anns={}
    loc=[]
    for core,boxes in [('a',[[.05,.05,.1,.1],[.85,.85,.1,.1]]),('d',[[.4,.4,.1,.1]])]:
        row=next(r for r in master if r['core_id']==core)
        anns[core]={k:row[k] for k in ('core_id','image','asset_category','defect_gt')}
        anns[core].update(status='boxed',boxes_xywh=boxes)
        loc.append(dict(row,protocol='gold_v2_loc_mmad',localization_gt=dict(normalized_bboxes_xywh=boxes,basis='manual_box_annotation')))
    row=next(r for r in master if r['core_id']=='c')
    anns['c']={k:row[k] for k in ('core_id','image','asset_category','defect_gt')}
    anns['c'].update(status='unlocatable',boxes_xywh=[])
    return master,loc,anns


def dataset():
    return q3.build_dataset(*inputs())


def archives(tmp_path,rows):
    sup,uns=tmp_path/'sup.zip',tmp_path/'uns.zip'
    with ZipFile(sup,'w') as z:
        for r in rows:
            z.writestr(r['image'],r['core_id'].encode())
    with ZipFile(uns,'w') as z:
        z.writestr(rows[0]['normal_reference_image'],b'reference')
    return sup,uns


def test_separate_centers_preserve_opposite_defects_and_unknown():
    rows={r['core_id']:r for r in dataset()}
    assert rows['a']['regions_gt']==[1,9]
    assert rows['b']['regions_gt']==[]
    assert rows['c']['regions_gt'] is None and rows['c']['anomaly_gt']=='anomalous'
    assert q3.boxes_to_regions([[.3,.3,1/15,1/15]])==[5]  # exact first grid boundary
    assert q3.boxes_to_regions([[.1,.1,.1,.1],[.15,.15,.1,.1]])==[1]


@pytest.mark.parametrize('box',[[0,0,0,.1],[-.1,0,.2,.2],[0,0,float('nan'),.2],[.9,0,.5,.2]])
def test_invalid_box_rejected(box):
    with pytest.raises(ValueError):
        q3.boxes_to_regions([box])


@pytest.mark.parametrize('bad',['{}','{"regions":[true]}','{"regions":["1"]}',
    '{"regions":[1,1]}','{"regions":[0]}','{"regions":[10]}','{"regions":null}',
    'Answer: {"regions":[1]}','{"regions":[],"why":"none"}', '{"regions":[1],"regions":[9]}'])
def test_invalid_output_never_becomes_normal(bad):
    assert q3.parse_json(bad,'regions') is None


def test_strict_json_and_presence_types():
    assert q3.parse_json('{"regions":[]}','regions')==[]
    assert q3.parse_json('```json\n{"regions":[9,1]}\n```','regions')==[1,9]
    assert q3.parse_json('{"present":false}','present') is False
    assert q3.parse_json('{"present":0}','present') is None


def test_missing_manual_labels_and_conflicts_block_build():
    master,loc,ann=inputs()
    del ann['c']
    with pytest.raises(ValueError,match='missing annotation'):
        q3.build_dataset(master,loc,ann)
    master,loc,ann=inputs()
    loc[0]['localization_gt']['normalized_bboxes_xywh']=[[.4,.4,.1,.1]]
    with pytest.raises(ValueError,match='boxes differ'):
        q3.build_dataset(master,loc,ann)
    master,loc,ann=inputs()
    with pytest.raises(ValueError,match='duplicate'):
        q3.build_dataset(master,loc+[loc[0]],ann)


def test_missed_sites_false_sites_and_unknown_denominators():
    rows=dataset()
    result=q3.score(rows,{'a':[1,5],'b':[5],'c':[],'d':None})
    assert (result['tp'],result['fp'],result['tn'],result['fn'])==(1,1,0,2)
    assert result['spatial_known_images']==3 and result['spatial_unknown_images']==1
    assert result['spatial_known_anomalies']==2 and result['anomaly_exact_set_accuracy']==0
    assert (result['region_tp'],result['region_fp'],result['region_fn'])==(1,2,2)
    assert result['invalid_anomalous']==1 and result['false_none']==1
    invalid=q3.score(rows,{r['core_id']:None for r in rows})
    assert (invalid['fp'],invalid['fn'])==(1,3)
    normal=q3.score([r for r in rows if r['anomaly_gt']=='normal'],{'b':[]})
    assert normal['presence_balanced_accuracy'] is None
    assert normal['region_micro_f1'] is None
    report=q3.reports(rows,{'a':[1,5],'b':[5],'c':[],'d':None})
    assert report['per_cell'][4]['fp']==2 and report['per_cell'][4]['fn']==1
    unknown=next(r for r in report['per_image'] if r['core_id']=='c')
    assert unknown['missed_regions'] is None and unknown['regions_gt'] is None


def test_prompt_cannot_receive_labels_and_same_for_every_image():
    text=q3.prompt('regions')
    assert 'center' in text and 'average' in text
    assert not any(r['core_id']+'-q3-multiregion' in text or r['image'] in text for r in dataset())
    with pytest.raises(TypeError):
        q3.prompt(row=dataset()[0])


def test_direct_staged_routing_resume_and_preservation(tmp_path):
    rows=dataset()
    before=copy.deepcopy(rows)
    sup,uns=archives(tmp_path,rows)
    seen=[]
    def infer(adapter,paths,prompt):
        core=paths[-1].stem
        assert len(paths)==2 and paths[0].name=='ref.jpg'
        seen.append((core,prompt))
        if '"present"' in prompt:
            return '{"present":false}' if core=='b' else '{"present":true}'
        return json.dumps({'regions':{'a':[1,9],'b':[],'c':[],'d':[5]}[core]})
    kwargs=dict(model_snapshot=tmp_path/'fixture',infer=infer,loader=lambda spec:object())
    result=runner.run(rows,sup,uns,tmp_path/'out',**kwargs)
    assert result['status']=='complete' and result['called_this_run']==11
    assert result['calls_by_policy']=={'direct':4,'staged':7}
    assert not any(c=='b' and 'every visible defect site' in p for c,p in seen[4:])
    result=runner.run(rows,sup,uns,tmp_path/'out',**kwargs)
    assert result['called_this_run']==0 and result['reused_calls']==11
    assert rows==before
    changed=copy.deepcopy(rows); changed[0]['defect_gt']='missing-cap'
    with pytest.raises(ValueError,match='changed'):
        runner.run(changed,sup,uns,tmp_path/'out',**kwargs)
    with ZipFile(sup,'w') as z:
        for r in rows:z.writestr(r['image'],b'changed')
    with pytest.raises(ValueError,match='cached image changed'):
        runner.run(rows,sup,uns,tmp_path/'out',**kwargs)


def test_error_retries_preserve_attempts_but_invalid_is_not_reasked(tmp_path):
    rows=dataset()[:2]
    sup,uns=archives(tmp_path,rows)
    count={}
    def infer(adapter,paths,prompt):
        key=paths[-1].stem
        count[key]=count.get(key,0)+1
        if key=='a' and count[key]==1:
            raise RuntimeError('transient fixture error')
        return '{"regions":[1,9]}' if key=='a' else 'not json'
    kwargs=dict(model_snapshot=tmp_path/'fixture',infer=infer,loader=lambda spec:object(),policies=('direct',))
    a=runner.run(rows,sup,uns,tmp_path/'out',**kwargs)
    assert a['status']=='has_errors'
    b=runner.run(rows,sup,uns,tmp_path/'out',**kwargs)
    assert b['status']=='complete' and b['called_this_run']==1 and b['invalid_final_outputs']==1
    assert count=={'a':2,'b':1}
    calls=json.loads((tmp_path/'out/calls.json').read_text())
    assert len(calls['a/direct/regions']['previous_attempts'])==1


def test_no_model_download_or_gpu_without_explicit_snapshot(tmp_path):
    rows=dataset()
    sup,uns=archives(tmp_path,rows)
    with pytest.raises(ValueError,match='pinned HF'):
        runner.run(rows,sup,uns,tmp_path/'out',model_snapshot=tmp_path/'missing')
    assert not (tmp_path/'out').exists()

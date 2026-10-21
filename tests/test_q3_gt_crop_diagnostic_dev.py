import copy
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_q3_gt_crop_diagnostic_dev as run


def setup(tmp):
    inp=tmp/'inputs';inp.mkdir()
    image=Image.fromarray(np.arange(101*67*3,dtype=np.uint8).reshape(67,101,3))
    payload=run.old.png(image); h=run.sha(payload); ref='glass-insulator/train/good/ref.png'
    rows=[];manifest=[dict(role='reference',member=ref,sha256=h)]
    with ZipFile(tmp/'sup.zip','w') as sup,ZipFile(tmp/'unsup.zip','w') as unsup:
        unsup.writestr(ref,payload)
        for i,boxes in enumerate([[[.4,.8,.1,.1]],[[0,0,.1,.1],[.8,.8,.2,.2]]]):
            member=f'defect_supervised/glass-insulator/train/missing-cap/{i}.png';sup.writestr(member,payload)
            manifest.append(dict(role='target',member=member,sha256=h))
            rows.append(dict(core_id=f'c{i}',image=member,asset_category='glass-insulator',defect_gt='missing-cap',
                anomaly_gt='anomalous',protocol=run.PROTOCOL,source_protocol='gold_v2_dev',annotation_status='boxed',
                image_sha256=h,size=[101,67],boxes=boxes,regions_gt=run.annotation.box_regions(boxes),
                label_basis='human_dev_box_centers',candidate_cells=[5,8],cached_first=[],
                previous_candidate_prediction=[],normal_reference_image=ref))
    data=dict(protocol=run.PROTOCOL,purpose='oracle_diagnosis_only',new_holdout=False,rows=rows,
              image_manifest=manifest,annotation_sha256='fixture',source_run_sha256='fixture')
    p=inp/'dataset.json';run.write(p,data);out=tmp/(run.PREFIX+'_fixture')
    args=['--dataset',str(p),'--sup-zip',str(tmp/'sup.zip'),'--unsup-zip',str(tmp/'unsup.zip'),
          '--out-dir',str(out),'--n-boot','30']
    return data,args,out


def execute(data,args,seen,*,invalid=False,error=False,interrupt=None):
    def infer(adapter,images,prompt):
        seen.append(images[-1].stem)
        if interrupt==len(seen):raise KeyboardInterrupt()
        assert len(images)==3 and adapter['image_tile_budgets']==[3,3,2]
        assert Image.open(images[-1]).size==(1024,548)
        assert all(v not in prompt for v in ['oracle','human','regions_gt','defect_supervised/','is defective','confirmed defect'])
        stem=images[-1].stem; core=stem.split('_')[0]
        if invalid and 'oracle_box' in stem:return 'No JSON, explanation only'
        if error and 'oracle_box' in stem:raise RuntimeError('fixture inference error')
        return json.dumps({'regions':[] if 'candidate_grid' in stem else next(r['regions_gt'] for r in data['rows'] if r['core_id']==core)})
    return run.main(args,loader=lambda _:{'kind':'internvl_chat'},infer=infer)


def test_matched_requests_original_coordinates_and_zero_call_resume(tmp_path):
    data,args,out=setup(tmp_path);before=Path(args[1]).read_bytes();seen=[]
    assert execute(data,args,seen)==0 and len(seen)==6
    status=run.read(out/'run_status.json');assert status['complete'] and status['diagnostic_only']
    candidate=run.read(out/'metrics_candidate_grid.json')['overall']
    oracle=run.read(out/'metrics_oracle_box.json')['overall']
    assert candidate['any_location_hit']==0 and oracle['any_location_hit']==2
    assert oracle['normal_support']==0 and oracle['normal_fpr'] is None and oracle['presence_balanced_accuracy'] is None
    calls=run.read(out/'calls.json');assert len(calls)==6
    assert len({r['prompt'] for r in calls.values()})==1
    for rec in calls.values():
        assert len(rec['image_sha256'])==3
        assert rec['generation']=={'max_new_tokens':80,'do_sample':False}
    assert Path(args[1]).read_bytes()==before
    seen.clear();assert execute(data,args,seen)==0 and not seen
    assert run.read(out/'run_metadata.json')['reused_lvlm_calls']==6


def test_padding_edges_and_actual_original_crop_pixels():
    row=dict(size=[101,67],candidate_cells=[5,8],regions_gt=[1,9],boxes=[[0,0,.1,.1],[.8,.8,.2,.2]])
    assert run.padded_box(row['size'],row['boxes'][0])==[0,0,13,9]
    assert run.padded_box(row['size'],row['boxes'][1])==[75,50,101,67]
    im=Image.new('RGB',(101,67),'red')
    for arm in run.ARMS:
        cells,bounds=run.geometry(row,arm);sheet=run.make_sheet(im,cells,bounds)
        assert sheet.size==(1024,548) and np.asarray(sheet)[300,256,0]==255
        assert all(0<=l<r<=101 and 0<=t<b<=67 for l,t,r,b in bounds)
    sheet=run.make_sheet(im,[1],[[0,0,13,9]])
    assert np.all(np.asarray(sheet)[:,512:]==255) # unused panel is blank, never a duplicate site
    # Same crop pixels with a different GT cell produce exactly the same visible sheet.
    assert np.array_equal(np.asarray(run.make_sheet(im,[1],[[0,0,13,9]])),np.asarray(run.make_sheet(im,[9],[[0,0,13,9]])))
    # Oracle grid deduplicates cells, while box arm retains distinct sites in same cell.
    row['boxes']=[[.05,.05,.05,.05],[.15,.15,.05,.05]];row['regions_gt']=[1]
    assert len(run.geometry(row,'oracle_grid')[0])==1 and len(run.geometry(row,'oracle_box')[0])==2


@pytest.mark.parametrize('mutation',[
    lambda d:d.update(protocol='gold'),
    lambda d:d.update(new_holdout=True),
    lambda d:d['rows'][0].update(source_protocol='gold_v2'),
    lambda d:d['rows'][0].update(annotation_status='unlocatable'),
    lambda d:d['rows'][0].update(anomaly_gt='normal'),
    lambda d:d['rows'][0].update(regions_gt=[1]),
    lambda d:d['rows'][0].update(core_id='../unsafe'),
    lambda d:d['rows'][0].update(size=[True,67]),
    lambda d:d['rows'][0].update(previous_candidate_prediction=[True]),
    lambda d:d['rows'][0].update(boxes=[[0,0,.1,.1]]*3),
    lambda d:d['rows'][0].update(candidate_cells=[True]),
    lambda d:d['rows'][0].update(normal_reference_image='glass-insulator/train/missing-cap/a.png'),
])
def test_refuse_contract_drift_unknown_and_nondev(tmp_path,mutation):
    data,_,_=setup(tmp_path);mutation(data)
    with pytest.raises(ValueError):run.validate(data,True)


def test_invalid_is_penalized_and_never_correctness_retried(tmp_path):
    data,args,out=setup(tmp_path);seen=[]
    assert execute(data,args,seen,invalid=True)==0
    assert run.read(out/'run_status.json')['invalid_new_outputs']==2
    metrics=run.read(out/'metrics_oracle_box.json')['overall']
    assert metrics['invalid_anomalous']==2 and metrics['fn']==2 and metrics['region_fn']==3
    seen.clear();assert execute(data,args,seen)==0 and not seen


def test_only_execution_errors_retried_with_history(tmp_path):
    data,args,out=setup(tmp_path);seen=[]
    assert execute(data,args,seen,error=True)==1
    assert not run.read(out/'run_status.json')['eligible']['oracle_box']
    seen.clear();assert execute(data,args,seen)==0
    assert len(seen)==2 and all('oracle_box' in x for x in seen)
    assert len(list((out/'attempt_history').glob('*.json')))==2


def test_interrupted_run_reuses_successful_records(tmp_path):
    data,args,out=setup(tmp_path);seen=[]
    with pytest.raises(KeyboardInterrupt):execute(data,args,seen,interrupt=3)
    assert len(run.read(out/'calls.json'))==2
    seen.clear();assert execute(data,args,seen)==0 and len(seen)==4


@pytest.mark.parametrize('what',['request','parsing','crop','dataset','config'])
def test_tampering_stops_without_new_neural_call(tmp_path,what,monkeypatch):
    data,args,out=setup(tmp_path);assert execute(data,args,[])==0
    if what in ['request','parsing']:
        calls=run.read(out/'calls.json');key=next(iter(calls))
        calls[key]['prompt' if what=='request' else 'prediction']='changed'
        run.write(out/'calls.json',calls)
    elif what=='crop':next((out/'crops').glob('*.png')).write_bytes(b'bad')
    elif what=='dataset':
        data['annotation_sha256']='changed';run.write(Path(args[1]),data)
    else:monkeypatch.setitem(run.CONFIG,'box_padding_per_side',.5)
    seen=[]
    with pytest.raises(ValueError):execute(data,args,seen)
    assert not seen


def test_wrong_original_sha_refused_before_model_load(tmp_path):
    data,args,out=setup(tmp_path)
    for r in data['image_manifest']:r['sha256']='0'*64
    for r in data['rows']:r['image_sha256']='0'*64
    run.write(Path(args[1]),data);seen=[]
    with pytest.raises(ValueError,match='image SHA'):execute(data,args,seen)
    assert not seen


def test_native_requires_pinned_model_and_complete_fixture_pair(tmp_path):
    data,args,out=setup(tmp_path)
    with pytest.raises(ValueError,match='both fixture'):run.main(args,loader=lambda _:None)
    with pytest.raises(ValueError,match='frozen paired20'):run.main(args)


def test_paired_diagnostic_deltas_no_normal_fpr():
    rows=[dict(core_id='a',regions_gt=[1]),dict(core_id='b',regions_gt=[2])]
    finals=dict(candidate_grid={'a':[],'b':None},oracle_grid={'a':[1],'b':[]},oracle_box={'a':[1],'b':[2]})
    result=run.paired(rows,finals,{p:True for p in run.ARMS},30)
    assert len(result)==9 and all(r['diagnostic_only'] for r in result)
    assert next(r for r in result if r['baseline']=='candidate_grid' and r['condition']=='oracle_box' and r['metric']=='any_location')['delta']==1
    assert not run.paired(rows,finals,{p:False for p in run.ARMS},30)

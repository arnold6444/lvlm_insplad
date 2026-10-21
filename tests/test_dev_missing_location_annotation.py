import base64
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import dev_missing_location_annotation as a


def setup(tmp):
    buf=io.BytesIO();Image.new('RGB',(300,180),(120,140,150)).save(buf,format='PNG');payload=buf.getvalue()
    targets=[];archive=tmp/'sup.zip'
    with ZipFile(archive,'w') as z:
        for i in range(3):
            name=f'defect_supervised/glass-insulator/train/missing-cap/test{i}.png';z.writestr(name,payload)
            targets.append(dict(core_id=f'dev-{i}',image=name,sha256=a.sha(payload),size=[300,180],asset_category='glass-insulator',
                defect_gt='missing-cap',anomaly_gt='anomalous',source_protocol='gold_v2_dev'))
    manifest=dict(protocol=a.PROTOCOL,source_snapshot_sha256='fixture',cohort='fixture',new_holdout=False,targets=targets)
    return manifest,archive,a.prepare(manifest,archive,tmp/(a.PREFIX+'_fixture'))


def test_draft_resume_confirm_multisite_and_model_blind_payload(tmp_path):
    m,z,s=setup(tmp_path);boxes=[[.05,.05,.1,.1],[.8,.8,.1,.1]]
    v=s.save(0,boxes,'draft','memo',0);assert v['done']==0 and v['revision']==1
    resumed=a.prepare(m,z,s.out);item=resumed.item(0)
    assert item['boxes']==boxes and item['status']=='draft' and item['revision']==1
    assert not {'core_id','image','image_path','defect_gt','regions_gt','proposal','prediction'} & item.keys()
    assert a.sha(base64.b64decode(item['image_url'].split(',')[1]))==m['targets'][0]['sha256']
    v=resumed.save(0,boxes,'boxed','memo',1);assert v['done']==1 and v['revision']==2
    path,summary=resumed.export();assert path.is_file() and summary['positive_spatial_support']==1 and not summary['complete']
    gt=a.read(s.out/'dev_missing_location_gt.json');assert gt[0]['regions_gt']==[1,9] and gt[1]['regions_gt'] is None
    assert len(list((s.out/'history/dev-0').glob('*.json')))==1
    with ZipFile(path) as archive:
        assert all('_images' not in n for n in archive.namelist())
        assert json.loads(archive.read('annotation/dev_missing_location_gt.json'))==gt


def test_unknown_and_disputed_are_complete_but_never_normal_or_empty_gt(tmp_path):
    _,_,s=setup(tmp_path)
    s.save(0,[[.1,.1,.1,.1]],'boxed','',0)
    s.save(1,[],'unlocatable','가려져 위치 판단 불가',0)
    s.save(2,[],'disputed','누락인지 확신 불가',0)
    _,summary=s.export();assert summary['complete'] and summary['positive_spatial_support']==1 and summary['unknown_spatial_support']==2
    rows=a.read(s.out/'dev_missing_location_gt.json')
    assert all(r['anomaly_gt']=='anomalous' for r in rows)
    assert rows[1]['regions_gt'] is None and rows[2]['regions_gt'] is None
    assert a.read(s.out/'run_status.json')['complete']


@pytest.mark.parametrize('boxes',[[],[[float('nan'),0,.1,.1]],[[0,0,float('inf'),.1]],[[True,0,.1,.1]],
    [[-.1,0,.1,.1]],[[.99,0,.2,.1]],[[0,0,.001,.1]],[[0,0,1]],'not-list'])
def test_invalid_boxes_rejected_without_creating_label(tmp_path,boxes):
    _,_,s=setup(tmp_path)
    with pytest.raises(ValueError):s.save(0,boxes,'boxed','',0)
    assert s.progress()['done']==0 and not s.path.exists()


def test_notes_indices_and_revisions_guard(tmp_path):
    _,_,s=setup(tmp_path)
    for status in ['unlocatable','disputed']:
        with pytest.raises(ValueError):s.save(0,[],status,'',0)
        with pytest.raises(ValueError):s.save(0,[[0,0,.2,.2]],status,'reason',0)
    for i in [-1,3,True,1.0]:
        with pytest.raises(ValueError):s.save(i,[],'draft','',0)
    s.save(0,[[.1,.1,.1,.1]],'draft','',0)
    with pytest.raises(ValueError,match='다른 화면'):s.save(0,[],'draft','',0)
    with pytest.raises(ValueError):s.save(0,[],'draft','',True)
    assert s.item(0)['boxes']==[[.1,.1,.1,.1]]


def test_edit_confirmed_record_returns_to_draft_without_losing_history(tmp_path):
    m,z,s=setup(tmp_path);s.save(0,[[.1,.1,.1,.1]],'boxed','',0)
    second=a.prepare(m,z,s.out);second.save(0,[[.6,.6,.1,.1]],'draft','edited',1)
    assert s.progress()['done']==0 and s.item(0)['status']=='draft'
    gt=s.export()[1];assert gt['positive_spatial_support']==0
    assert a.read(next((s.out/'history/dev-0').glob('*.json')))['status']=='boxed'


def test_manifest_or_image_changes_preserve_existing_annotations(tmp_path):
    m,z,s=setup(tmp_path);s.save(0,[],'draft','keep',0);before=s.path.read_bytes()
    changed=json.loads(json.dumps(m));changed['cohort']='changed'
    with pytest.raises(ValueError,match='対象|대상'):a.prepare(changed,z,s.out)
    p=s.paths['dev-0'];p.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='사진이 바뀌었습니다'):s.item(0)
    assert s.path.read_bytes()==before


def test_unsafe_or_model_polluted_manifest_rejected(tmp_path):
    m,_,_=setup(tmp_path)
    for key,value in [('core_id','../../escape'),('image','../escape.png'),('source_protocol','gold'),('prediction',[5])]:
        x=json.loads(json.dumps(m));x['targets'][0][key]=value
        with pytest.raises(ValueError):a.validate_manifest(x)


def test_atomic_write_failure_keeps_acknowledged_state(tmp_path,monkeypatch):
    _,_,s=setup(tmp_path);s.save(0,[],'draft','keep',0);before=s.path.read_bytes();original=a.os.replace
    def fail(src,dst):
        if Path(dst)==s.path:raise OSError('fixture interrupted write')
        return original(src,dst)
    monkeypatch.setattr(a.os,'replace',fail)
    with pytest.raises(OSError):s.save(0,[[.1,.1,.1,.1]],'boxed','',1)
    assert s.path.read_bytes()==before and s.item(0)['revision']==1


def test_grid_centers_do_not_average_sites_and_boundaries():
    assert a.box_regions([[0,0,.1,.1],[.8,.8,.1,.1]])==[1,9]
    assert a.box_regions([[0,0,.1,.1],[.1,.1,.1,.1]])==[1]
    assert a.box_regions([[.2,.2,.4,.4]])==[5]


def test_colab_callbacks_are_wrapped_and_ui_does_not_show_model_answers(tmp_path,monkeypatch):
    import types
    _,_,s=setup(tmp_path);callbacks={};shown=[]
    google=types.ModuleType('google');colab=types.ModuleType('google.colab')
    colab.output=types.SimpleNamespace(register_callback=lambda n,f:callbacks.update({n:f}))
    ip=types.ModuleType('IPython');display=types.ModuleType('IPython.display')
    display.HTML=lambda x:x;display.JSON=lambda x:x;display.display=shown.append
    for name,module in [('google',google),('google.colab',colab),('IPython',ip),('IPython.display',display)]:monkeypatch.setitem(sys.modules,name,module)
    a.register_colab(s);assert set(callbacks)=={'devmissing.item','devmissing.progress','devmissing.save'}
    assert callbacks['devmissing.item'](0)['index']==0
    assert callbacks['devmissing.save'](0,[],'draft','note',0)['revision']==1
    assert 'invokeFunction' in shown[0] and '초안 저장' in shown[0]

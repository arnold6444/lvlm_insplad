"""Model-blind human location labels for all missing-cap photos in fixed dev240."""
from __future__ import annotations
import base64
from collections import Counter
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path, PurePosixPath
import time
import uuid
from zipfile import ZipFile, ZIP_DEFLATED

PROTOCOL='dev_missing_location_annotation_v1'
GT_PROTOCOL='dev_missing_location_gt_v1'
PREFIX='dev_missing_location_annotation'
FINAL=('boxed','unlocatable','disputed')
sha=lambda b:hashlib.sha256(b).hexdigest()
digest=lambda x:sha(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode())


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def validate_manifest(manifest):
    if set(manifest)!={'protocol','source_snapshot_sha256','cohort','new_holdout','targets'} or manifest['protocol']!=PROTOCOL or manifest['new_holdout'] is not False:
        raise ValueError('expected separate fixed-dev annotation manifest')
    targets=manifest['targets']
    if not targets or len({r['core_id'] for r in targets})!=len(targets):raise ValueError('empty/duplicate targets')
    for r in targets:
        if set(r)!={'core_id','image','sha256','size','asset_category','defect_gt','anomaly_gt','source_protocol'}:raise ValueError('identity-only manifest required; no model outputs')
        if r['source_protocol']!='gold_v2_dev' or r['asset_category']!='glass-insulator' or r['defect_gt']!='missing-cap' or r['anomaly_gt']!='anomalous':raise ValueError('only dev missing-cap, never gold')
        if not isinstance(r['core_id'],str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',r['core_id']):raise ValueError('unsafe image ID')
        name=r['image']
        if not isinstance(name,str) or '\\' in name or ':' in name or name.startswith('/') or '..' in PurePosixPath(name).parts:raise ValueError('unsafe image member')
        if len(r['size'])!=2 or any(type(x) is not int or x<3 for x in r['size']):raise ValueError('invalid image dimensions')
        if len(r['sha256'])!=64 or any(c not in '0123456789abcdef' for c in r['sha256']):raise ValueError('invalid image SHA')
    return digest(manifest)


def clean_boxes(boxes,size,allow_empty=False):
    if not isinstance(boxes,list) or (not boxes and not allow_empty):raise ValueError('박스를 하나 이상 그리세요.')
    result=[]
    for b in boxes:
        if not isinstance(b,(tuple,list)) or len(b)!=4 or any(type(x) not in (int,float) or not math.isfinite(x) for x in b):raise ValueError('잘못된 박스 좌표입니다.')
        x,y,w,h=map(float,b)
        if min(x,y)<0 or min(w,h)<=0 or x+w>1.00000001 or y+h>1.00000001 or w*size[0]<2 or h*size[1]<2:raise ValueError('박스는 사진 안에 두고 너무 작은 클릭은 지워주세요.')
        result.append([x,y,w,h])
    return result


def box_regions(boxes):
    return sorted({1+min(2,int((b[1]+b[3]/2)*3))*3+min(2,int((b[0]+b[2]/2)*3)) for b in boxes})


def prepare(manifest,archive,out):
    from PIL import Image
    out=Path(out);archive=Path(archive);identity=validate_manifest(manifest)
    if not out.name.startswith(PREFIX) or archive.resolve().is_relative_to(out.resolve()):raise ValueError('separate dev annotation folder required')
    lock=out/'manifest.json'
    if lock.exists() and read(lock)!=manifest:raise ValueError('대상 사진이 바뀌었습니다. 기존 파일을 덮어쓰지 말고 새 폴더를 사용하세요.')
    if not lock.exists() and out.exists() and any(out.iterdir()):raise ValueError('알 수 없는 기존 폴더입니다. 새 폴더를 사용하세요.')
    # Check all original bytes before creating annotation files.
    payloads={}
    with ZipFile(archive) as z:
        for r in manifest['targets']:
            b=z.read(r['image'])
            if sha(b)!=r['sha256']:raise ValueError('원본 사진 SHA가 다릅니다: '+r['core_id'])
            with Image.open(io.BytesIO(b)) as im:
                if list(im.size)!=r['size'] or im.getexif().get(274,1)!=1:raise ValueError('사진 크기/EXIF 방향이 바뀌었습니다.')
            payloads[r['core_id']]=b
    out.mkdir(parents=True,exist_ok=True);atomic(lock,manifest)
    paths={}
    for r in manifest['targets']:
        p=out/'_images'/(sha(r['core_id'].encode())+PurePosixPath(r['image']).suffix)
        p.parent.mkdir(exist_ok=True)
        if p.exists() and sha(p.read_bytes())!=r['sha256']:raise ValueError('저장된 사진 파일이 손상됐습니다.')
        if not p.exists():p.write_bytes(payloads[r['core_id']])
        paths[r['core_id']]=p
    return Session(manifest,paths,out)


class Session:
    def __init__(self,manifest,paths,out):
        self.manifest=manifest;self.identity=validate_manifest(manifest);self.targets=manifest['targets'];self.paths=paths;self.out=Path(out)
        self.path=self.out/'annotations.json';self._state();self._status()

    def _state(self):
        state=read(self.path) if self.path.exists() else {'protocol':PROTOCOL,'manifest_identity':self.identity,'records':{}}
        if set(state)!={'protocol','manifest_identity','records'} or state['protocol']!=PROTOCOL or state['manifest_identity']!=self.identity:raise ValueError('표시 파일의 대상 목록이 다릅니다.')
        byid={r['core_id']:r for r in self.targets}
        if not isinstance(state['records'],dict) or set(state['records'])-set(byid):raise ValueError('알 수 없는 표시 ID입니다.')
        for core,r in state['records'].items():
            if set(r)!={'status','boxes_xywh','note','revision','image_sha256','image_size','saved_at'}:raise ValueError('잘못된 표시 기록 형식입니다.')
            if r['image_sha256']!=byid[core]['sha256'] or r['image_size']!=byid[core]['size'] or r['status'] not in ('draft',*FINAL):raise ValueError('표시의 사진/상태가 다릅니다.')
            if type(r['revision']) is not int or r['revision']<1:raise ValueError('잘못된 표시 버전입니다.')
            if r['status'] in ('draft','boxed'):clean_boxes(r['boxes_xywh'],r['image_size'],r['status']=='draft')
            elif r['boxes_xywh'] or not r['note'].strip():raise ValueError('위치 미확정에는 사유가 필요합니다.')
        return state

    def progress(self):
        records=self._state()['records'];counts=Counter(r['status'] for r in records.values())
        return {'total':len(self.targets),'done':sum(counts[x] for x in FINAL),'boxed':counts['boxed'],
            'unlocatable':counts['unlocatable'],'disputed':counts['disputed'],'drafts':counts['draft'],
            'first_todo':next((i for i,t in enumerate(self.targets) if records.get(t['core_id'],{}).get('status') not in FINAL),len(self.targets))}

    def _status(self):
        progress=self.progress();atomic(self.out/'run_status.json',{'protocol':PROTOCOL,'phase':'human_annotation',
            'complete':progress['done']==progress['total'],**progress,'neural_calls':0,'gold_changed':False,'new_holdout':False})

    def item(self,index):
        from PIL import Image
        if type(index) is not int:raise ValueError('invalid image index')
        index=max(0,min(index,len(self.targets)-1));target=self.targets[index];p=self.paths[target['core_id']]
        b=p.read_bytes()
        if sha(b)!=target['sha256']:raise ValueError('사진이 바뀌었습니다.')
        with Image.open(io.BytesIO(b)) as im:mime=Image.MIME[im.format]
        rec=self._state()['records'].get(target['core_id'],{})
        # No core ID, file name/path, class label, model answers, proposals or grid GT.
        return {'index':index,**self.progress(),'image_url':'data:'+mime+';base64,'+base64.b64encode(b).decode(),
            'width':target['size'][0],'height':target['size'][1],'boxes':rec.get('boxes_xywh',[]),
            'status':rec.get('status',''),'note':rec.get('note',''),'revision':rec.get('revision',0),'saved_at':rec.get('saved_at','')}

    def save(self,index,boxes,status,note='',revision=0):
        if type(index) is not int or not 0<=index<len(self.targets):raise ValueError('invalid image index')
        if status not in ('draft',*FINAL) or not isinstance(note,str):raise ValueError('invalid annotation state')
        if len(note)>4000:raise ValueError('메모가 너무 깁니다.')
        target=self.targets[index];state=self._state();old=state['records'].get(target['core_id'])
        current=old['revision'] if old else 0
        if type(revision) is not int or revision!=current:raise ValueError('다른 화면에서 표시가 변경됐습니다. 다시 불러오세요.')
        if status in ('draft','boxed'):clean=clean_boxes(boxes,target['size'],status=='draft')
        else:
            if boxes or not note.strip():raise ValueError('위치 판단 불가/라벨 이견에는 박스 없이 사유를 입력하세요.')
            clean=[]
        record={'status':status,'boxes_xywh':clean,'note':note.strip(),'revision':current+1,
            'image_sha256':target['sha256'],'image_size':target['size'],'saved_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
        if old:atomic(self.out/'history'/target['core_id']/(str(current)+'_'+uuid.uuid4().hex+'.json'),old)
        updated={**state,'records':{**state['records'],target['core_id']:record}}
        atomic(self.path,updated);self._status()
        return {'revision':record['revision'],'status':status,'saved_at':record['saved_at'],**self.progress()}

    def export(self):
        state=self._state();records=state['records'];rows=[]
        for target in self.targets:
            rec=records.get(target['core_id'],{});known=rec.get('status')=='boxed';boxes=rec['boxes_xywh'] if known else []
            rows.append({'id':target['core_id']+'-q3-multiregion-dev','core_id':target['core_id'],
                'image':target['image'],'image_sha256':target['sha256'],'asset_category':target['asset_category'],
                'anomaly_gt':target['anomaly_gt'],'defect_gt':target['defect_gt'],'question_type':'defect_localization',
                'protocol':GT_PROTOCOL,'regions_gt':box_regions(boxes) if known else None,'normalized_bboxes_xywh':boxes,
                'location_gt_known':known,'annotation_status':rec.get('status','pending'),
                'label_basis':'human_dev_box_centers' if known else 'human_dev_spatial_unknown',
                'note':rec.get('note',''),'annotation_revision':rec.get('revision',0)})
        atomic(self.out/'dev_missing_location_gt.json',rows)
        progress=self.progress();summary={**progress,'complete':progress['done']==progress['total'],
            'positive_spatial_support':sum(r['location_gt_known'] for r in rows),'unknown_spatial_support':sum(not r['location_gt_known'] for r in rows),
            'manifest_identity':self.identity,'annotations_sha256':sha(self.path.read_bytes()) if self.path.exists() else None,
            'new_holdout':False,'model_predictions_shown':False,'gold_changed':False,'neural_calls':0}
        atomic(self.out/'annotation_summary.json',summary);self._status()
        result=self.out.parent/'dev_missing_location_annotation25_results.zip'
        with ZipFile(result,'w',ZIP_DEFLATED) as z:
            for p in sorted(self.out.rglob('*')):
                if p.is_file() and '_images' not in p.relative_to(self.out).parts:z.write(p,'annotation/'+p.relative_to(self.out).as_posix())
        return result,summary


def register_colab(session):
    from google.colab import output
    from IPython.display import HTML,JSON,display
    output.register_callback('devmissing.item',lambda i:JSON(session.item(i)))
    output.register_callback('devmissing.progress',lambda:JSON(session.progress()))
    output.register_callback('devmissing.save',lambda i,b,s,n,v:JSON(session.save(i,b,s,n,v)))
    display(HTML(Path(__file__).with_name('dev_missing_location_ui.html').read_text(encoding='utf-8')))

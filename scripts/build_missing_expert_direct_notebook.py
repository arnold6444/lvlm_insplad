"""Seal audited features and evaluation-only location labels for the direct-expert Colab."""
import argparse
import ast
import base64
import io
import json
from pathlib import Path
import textwrap
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED
import run_missing_expert_direct_dev as exp

ORIGINAL_SHA='38b050835bbb786ff849f944e0ab2d95d7a1d652f1a617982a22025c93ee82de'
OCCLUSION_SHA='dc60885b0e8a98cd2bc30804385d7bb3dc0b33445059e9ed1b847e41171268d4'
ANNOTATION_SHA='9a9e71b995174288e3507e0d60f97fd155abb2149de2bf4ee0feb10fe02d52d3'


def package(args):
    for path, expected in ((args.original_run,ORIGINAL_SHA),(args.occlusion_run,OCCLUSION_SHA),(args.annotations,ANNOTATION_SHA)):
        if exp.probe.sha_file(path)!=expected: raise ValueError('use audited source artifact: '+str(path))
    with ZipFile(args.occlusion_run) as old, ZipFile(args.original_run) as orig:
        read=lambda z,n:json.loads(z.read(n))
        meta=read(old,'run/run_metadata.json'); original_meta=read(orig,'run/run_metadata.json')
        for z,m in ((old,meta),(orig,original_meta)):
            if not read(z,'run/run_status.json')['complete'] or m['fixture'] or m['backend']!='hf' or m['gold_used']:
                raise ValueError('actual successful dev source required')
        data=read(old,'run/dataset_snapshot.json'); head_bytes=old.read('run/frozen_detector.json'); head=json.loads(head_bytes)
        expected=exp.oc.validate(data,head,exp.sha(head_bytes))
        if len(data['rows'])!=240: raise ValueError('changed dev cohort')
        # Reuse the exact feature and mask implementation audited in PR64.
        with ZipFile(io.BytesIO(old.read('source_bundle.zip'))) as source:
            for name in ('scripts/run_q3_missing_occlusion_dev.py','scripts/run_missing_component_probe.py'):
                saved=source.read(name)
                if exp.sha(saved)!=meta['code_sha256'][name]: raise ValueError('source code checksum mismatch')
                if (exp.ROOT/name).read_bytes().replace(b'\r\n',b'\n')!=saved.replace(b'\r\n',b'\n'):
                    raise ValueError('changed source feature operation: '+name)
        targets=[r for r in data['rows'] if r['asset_category']==exp.probe.ASSET]
        if len(targets)!=50: raise ValueError('changed expert cohort')
        paths=exp.prior.old.verified_images(args.sup_zip,[r['image'] for r in targets],args.evidence_dir/'_images')
        images={}
        for r in targets:
            if exp.probe.sha_file(paths[r['image']])!=expected['target',r['image']]: raise ValueError('changed original')
            images[r['core_id']]=exp.oc.original_image(paths[r['image']])
        tools=read(orig,'run/tool_calls.json'); mask_records=read(old,'run/features.json')
        payloads={'inputs/dataset.json':old.read('run/dataset_snapshot.json'),'inputs/frozen_detector.json':head_bytes}
        seeds={}
        def add(core,kind,cell,content,rec,source):
            x=exp.check_vector(content,len(head['coef'])); vals=exp.score_feature(head,x)
            if exp.sha(content)!=rec['feature_sha256']: raise ValueError('cached feature SHA changed')
            if 'score' in rec and not exp.np.isclose(rec['score'],vals['score'],rtol=1e-12,atol=1e-12): raise ValueError('source score changed')
            if 'logit' in rec and not exp.np.isclose(rec['logit'],vals['logit'],rtol=1e-12,atol=1e-12): raise ValueError('source logit changed')
            key=exp.request(core,kind,cell); row=next(r for r in targets if r['core_id']==core)
            if rec['image_sha256']!=expected['target',row['image']] or rec['error']: raise ValueError('wrong cache source image/error')
            im=images[core] if kind=='original' else exp.oc.masked(images[core],cell,kind)
            if kind!='original' and rec['box']!=list(exp.oc.box(im.size,cell)): raise ValueError('source mask geometry changed')
            name='inputs/features/'+exp.digest(key)+'.npy'; payloads[name]=content
            seeds[key]=dict(id=key,core_id=core,kind=kind,cell=cell,feature_path=name,feature_sha256=exp.sha(content),
                            image_sha256=rec['image_sha256'],pixel_sha256=exp.pixel_sha(im),size=list(im.size),**vals,
                            source_archive_sha256=source,source_identity=rec['identity'],
                            source_variant_png_sha256=rec.get('variant_sha256'),
                            cache_note='same original RGB/mask operation and decoded pixels; PNG reencoding bytes may differ')
        for r in targets:
            core=r['core_id']; rec=tools.get(core)
            if rec:
                if rec['image']!=r['image']: raise ValueError('wrong source image')
                add(core,'original',0,orig.read('run/tool_features/'+rec['image_sha256']+'.npy'),rec,ORIGINAL_SHA)
        for key,rec in mask_records.items():
            core,kind,cell=key.split('/')
            if rec['id']!=key or rec['identity']!=meta['identity']: raise ValueError('wrong mask identity')
            add(core,kind,int(cell),old.read('run/feature_cache/'+exp.digest(key)+'.npy'),rec,OCCLUSION_SHA)
        if len(seeds)!=463 or sum(r['kind']=='original' for r in seeds.values())!=49: raise ValueError('changed source cache support')
        # Verify manually derived cells against original confirmed annotations, not model proposals.
        gt=exp.read(args.gt_json); annotations=exp.read(args.annotations)
        gt_map={r['core_id']:r for r in gt}
        if len(gt_map)!=25: raise ValueError('expected all25 human labels')
        records=annotations.get('records',annotations)
        for core,r in gt_map.items():
            if r['derived_locally_from_annotations_sha256']!=ANNOTATION_SHA: raise ValueError('wrong label provenance')
            row=next(x for x in targets if x['core_id']==core)
            if r['image']!=row['image'] or r['image_sha256']!=expected['target',row['image']]: raise ValueError('label identity mismatch')
            boxes=r['normalized_bboxes_xywh']; cells=r['regions_gt']
            if cells is not None and exp.spatial.boxes_to_regions(boxes)!=cells: raise ValueError('wrong box centers')
            if cells is None and (boxes or r['annotation_status']!='unlocatable'): raise ValueError('unknown is not normal')
            raw=records[core]
            if raw['boxes_xywh']!=boxes or raw['image_sha256']!=r['image_sha256'] or raw['status']!=r['annotation_status']:
                raise ValueError('derived annotation changed')
        if sum(r['regions_gt'] is not None for r in gt)!=24: raise ValueError('changed known support')
        payloads['inputs/evaluation_gt.json']=args.gt_json.read_bytes()
        payloads['inputs/annotations.json']=args.annotations.read_bytes()
        payloads['inputs/feature_seeds.json']=(json.dumps(seeds,indent=2)+'\n').encode()
        payloads['inputs/provenance.json']=(json.dumps(dict(original_sha256=ORIGINAL_SHA,occlusion_sha256=OCCLUSION_SHA,
            annotation_sha256=ANNOTATION_SHA,source_software=meta['software'],head_sha256=exp.sha(head_bytes),
            new_inference_this_build=0,feature_contract=exp.probe.FEATURE),indent=2)+'\n').encode()
    payloads.update({name:(exp.ROOT/name).read_bytes() for name in exp.prior.shared.code_hashes()})
    for name in ('scripts/build_missing_expert_direct_notebook.py','tests/test_missing_expert_direct_dev.py'):
        payloads[name]=(exp.ROOT/name).read_bytes()
    return payloads


def build(args):
    args.evidence_dir.mkdir(parents=True,exist_ok=True)
    payloads=package(args); manifest={n:exp.sha(b) for n,b in payloads.items()}; buffer=io.BytesIO()
    with ZipFile(buffer,'w',ZIP_DEFLATED) as z:
        for n,b in sorted(payloads.items()):
            info=ZipInfo(n,date_time=(2026,10,22,0,0,0));info.compress_type=ZIP_DEFLATED;z.writestr(info,b)
    payload=buffer.getvalue(); source_sha=exp.sha(payload); cells=[]
    def cell(kind,text):
        c=dict(id='expert-direct-'+str(len(cells)),cell_type=kind,metadata={},source=textwrap.dedent(text).strip().splitlines(keepends=True))
        if kind=='code': ast.parse(''.join(c['source'])); c.update(execution_count=None,outputs=[])
        cells.append(c)
    cell('markdown','''
    # 누락 판별기 + 1순위 위치후보 직접 채택 — dev 검증

    **파일: `lvlm_missing_expert_direct_dev_colab.ipynb`**

    **L4/A100 → 기본값 유지 → 모두 실행 → Drive 연결 허용. 기능 제작 완료이며 실제 새 구조의 효과는 아직 미검증입니다.**
    이전 코랩을 다시 돌리거나 추가 표시할 필요가 없습니다. 기존 supervised 원본 ZIP 한 개를 자동 검색합니다.

    - 기존 dev240의 LVLM 답은 보존합니다. 유리 절연체50장에 동결 누락 판별기를 적용합니다.
    - 기존 빈답 경로와 이전 답에 무관한 전체 판별기 양성 경로를 비교합니다. 양성이면 LVLM의 재거부 없이 유무를 채택합니다.
    - 위치는 두 가림 방식에서 점수가 함께 떨어진 1순위 칸만 선택합니다. 같은 경로에 중앙/고정 무작위 칸 대조군도 둡니다.
    - 후보가 없으면 **누락 있음·위치 보류**입니다. 위치 보류를 정상으로 바꾸지 않습니다.
    - 검산한 원본49/가림414 특징을 내장했습니다. 새 환경에서 기존 특징2개를 먼저 재계산해 동등성을 확인하고, 미계산 원본1개와 새 양성 경로의 가림만 계산합니다. DINO는 필요할 때만 로드합니다.
    - 새 LVLM 호출0/재학습0/Q1Q2/gold0. 사용자 위치 정답24장/판단불가1장은 채점에만 씁니다.
    - 이미 반복 검토한 dev이므로 새 holdout/최종 온라인 Agent 점수로 해석하지 않습니다. 정상 오탐과 위치 오답을 함께 확인합니다.
    - 중간 기록은 별도 Drive 폴더에 저장합니다. 같은 소스/입력/환경으로 재실행하면 성공 특징을 재사용하고 오류만 재시도합니다.
    ''')
    cell('code','''
    RUN_FOLDER = "lvlm_insplad_results/missing_expert_direct_dev240_v1"
    SUP_ZIP_PATH = ""
    ''')
    cell('code','''
    import subprocess,sys
    subprocess.run([sys.executable,'-m','pip','install','-q','numpy==2.1.3','pillow==11.3.0',
        'transformers==5.14.0','accelerate==1.14.0','timm==1.0.30'],check=True)
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<20*1024**3:
        raise RuntimeError('L4/A100 VRAM20GB 이상 런타임을 선택하세요.')
    ''')
    cell('code',f'''
    import base64,hashlib,io,json,zipfile
    from pathlib import Path
    from google.colab import drive
    drive.mount('/content/drive',force_remount=False)
    DRIVE=Path('/content/drive/MyDrive')
    SOURCE_SHA256={source_sha!r}
    SOURCE_MANIFEST={manifest!r}
    SOURCE_BYTES=base64.b64decode({base64.b64encode(payload).decode()!r})
    assert hashlib.sha256(SOURCE_BYTES).hexdigest()==SOURCE_SHA256
    CODE=Path('/content/missing_expert_direct_'+SOURCE_SHA256[:12])
    with zipfile.ZipFile(io.BytesIO(SOURCE_BYTES)) as archive:
        assert set(archive.namelist())==set(SOURCE_MANIFEST)
        for name in archive.namelist():
            target=CODE/name;content=archive.read(name)
            assert target.resolve().is_relative_to(CODE.resolve())
            assert hashlib.sha256(content).hexdigest()==SOURCE_MANIFEST[name]
            if target.exists() and target.read_bytes()!=content:raise RuntimeError('내장 소스 변경; 새 런타임을 사용하세요.')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    (CODE/'source_manifest.json').write_text(json.dumps(SOURCE_MANIFEST,indent=2))
    OUT=DRIVE/RUN_FOLDER
    if not OUT.name.startswith({exp.PREFIX!r}):raise ValueError('별도 실행 폴더 이름을 유지하세요.')
    RETURN_CODE=98;LOG=DRIVE/'missing_expert_direct_dev240_console.log'
    print('SOURCE',SOURCE_SHA256,'기존463특징 재사용 + 부족한 특징과 동등성검사2개만 새 계산; LVLM0')
    ''')
    cell('code','''
    def find_zip(override):
        name='supervised_fault_classification.zip';size=874753167
        if override:
            path=Path(override)
            if not path.is_file() or path.stat().st_size!=size:raise FileNotFoundError(str(path))
            return path
        for path in [DRIVE/name,DRIVE/'lvlm_insplad_data'/name]:
            if path.is_file() and path.stat().st_size==size:return path
        matches=sorted(p for p in DRIVE.rglob(name) if p.is_file() and p.stat().st_size==size)
        if not matches:raise FileNotFoundError(name)
        return matches[0]
    SUP=find_zip(SUP_ZIP_PATH)
    command=[sys.executable,str(CODE/'scripts/run_missing_expert_direct_dev.py'),
        '--capsule-dir',str(CODE),'--sup-zip',str(SUP),'--out-dir',str(OUT)]
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    with LOG.open('a',encoding='utf-8') as log:
        for line in process.stdout:print(line,end='');log.write(line);log.flush()
    RETURN_CODE=process.wait();print('RUNNER_RETURN_CODE',RETURN_CODE)
    ''')
    cell('code','''
    from google.colab import files
    RESULT=DRIVE/'missing_expert_direct_dev240_results.zip'
    with zipfile.ZipFile(RESULT,'w',zipfile.ZIP_DEFLATED) as archive:
        if OUT.exists():
            for path in sorted(OUT.rglob('*')):
                if path.is_file() and '_images' not in path.relative_to(OUT).parts:
                    archive.write(path,'run/'+path.relative_to(OUT).as_posix())
        archive.writestr('source_bundle.zip',SOURCE_BYTES)
        archive.writestr('source_manifest.json',json.dumps(SOURCE_MANIFEST,indent=2))
        archive.writestr('runner_return_code.json',json.dumps({'return_code':RETURN_CODE}))
        if LOG.exists():archive.write(LOG,'console.log')
    status=json.loads((OUT/'run_status.json').read_text()) if (OUT/'run_status.json').exists() else {'complete':False}
    print('RUN STATUS',status)
    if RETURN_CODE or not status.get('complete'):print('실패/미완료입니다. 같은 입력으로 재개하거나 ZIP을 전달하세요.')
    elif (OUT/'leaderboard.csv').exists():print((OUT/'leaderboard.csv').read_text())
    print('ZIP',RESULT,'SHA256',hashlib.sha256(RESULT.read_bytes()).hexdigest())
    files.download(str(RESULT))
    ''')
    notebook=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'colab':{'provenance':[]}},nbformat=4,nbformat_minor=5)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    (args.evidence_dir/'source_bundle.zip').write_bytes(payload)
    receipt=dict(source_bundle_sha256=source_sha,notebook_sha256=exp.probe.sha_file(args.output),source_files=manifest,
        compiled_code_cells=sum(c['cell_type']=='code' for c in cells),dev_images=240,expert_images=50,
        original_seed_features=49,masked_seed_features=414,new_neural_calls_this_build=0,new_lvlm_calls=0)
    exp.write(args.evidence_dir/'notebook_build.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_files'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('original-run','occlusion-run','annotations','gt-json','sup-zip','output','evidence-dir'):
        p.add_argument('--'+key,type=Path,required=True)
    build(p.parse_args())

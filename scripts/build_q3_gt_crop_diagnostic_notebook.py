"""Seal the audited human labels and prior frozen dev experiment into a real-run Colab."""
import argparse
import ast
import base64
import io
import json
from pathlib import Path
import textwrap
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

import run_q3_gt_crop_diagnostic_dev as experiment

ROOT=Path(__file__).resolve().parents[1]
SOURCE_RUN_SHA='dc60885b0e8a98cd2bc30804385d7bb3dc0b33445059e9ed1b847e41171268d4'
ANNOTATION_SHA='9a9e71b995174288e3507e0d60f97fd155abb2149de2bf4ee0feb10fe02d52d3'


def dataset(args):
    if experiment.old.prior.probe.sha_file(args.source_run)!=SOURCE_RUN_SHA:
        raise ValueError('audited actual PR64 archive required')
    if experiment.old.prior.probe.sha_file(args.annotations)!=ANNOTATION_SHA:
        raise ValueError('frozen actual25 confirmed annotations required')
    manifest=experiment.read(args.annotation_manifest)
    session=experiment.annotation.Session.__new__(experiment.annotation.Session)
    session.manifest=manifest;session.targets=manifest['targets'];session.path=args.annotations
    session.identity=experiment.annotation.validate_manifest(manifest)
    records=session._state()['records']; progress=session.progress()
    if progress!={'total':25,'done':25,'boxed':24,'unlocatable':1,'disputed':0,'drafts':0,'first_todo':25}:
        raise ValueError('expected24 boxed +1 unknown, all25 confirmed')
    with ZipFile(args.source_run) as z:
        source=json.loads(z.read('run/dataset_snapshot.json'))
        if experiment.sha(z.read('run/dataset_snapshot.json'))!=manifest['source_snapshot_sha256']:
            raise ValueError('annotation/prior snapshot mismatch')
        status=json.loads(z.read('run/run_status.json'));meta=json.loads(z.read('run/run_metadata.json'))
        if not status['complete'] or meta['fixture'] or meta['backend']!='hf' or meta['gold_used'] or any(status['execution_errors_by_policy'].values()):
            raise ValueError('complete actual prior dev required')
        if meta['model_revision']!=experiment.old.prior.MODEL_REVISION:raise ValueError('changed prior model')
        routes={r['core_id']:r for r in json.loads(z.read('run/routes.json'))}
        maps=json.loads(z.read('run/occlusion_maps.json'))
        previous=json.loads(z.read('run/predictions.json'))['occlusion_crop']
        crops={(r['core_id'],r['policy']):r for r in json.loads(z.read('run/crop_manifest.json'))}
    prior_rows={r['core_id']:r for r in source['rows']}
    expected={(r['role'],r['member']):r['sha256'] for r in source['image_manifest']}
    targets=manifest['targets']
    if {r['core_id'] for r in targets}!={k for k,r in prior_rows.items() if r['defect_gt']=='missing-cap'}:
        raise ValueError('all25 dev missing identities required')
    rows=[];excluded=[]
    for t in targets:
        k=t['core_id'];a=records[k];r=prior_rows[k];route=routes.get(k)
        if t['image']!=r['image'] or t['sha256']!=expected['target',r['image']]:raise ValueError('source photo mismatch')
        reason='unknown_location' if a['status']!='boxed' else 'not_previously_routed' if not route else 'prior_candidate_abstention' if not route['review_requested'] else None
        if reason:
            excluded.append(dict(core_id=k,reason=reason,annotation_status=a['status']));continue
        selected=maps[k]['proposal']['regions']
        if maps[k]['error'] or route['first']!=[] or route['proposal_cells']!=selected or crops[k,'occlusion_crop']['cells']!=selected:
            raise ValueError('frozen candidate/routing mismatch')
        rows.append({**{key:r[key] for key in ['core_id','image','asset_category','defect_gt','anomaly_gt']},
            'protocol':experiment.PROTOCOL,'source_protocol':'gold_v2_dev','annotation_status':'boxed',
            'image_sha256':t['sha256'],'size':t['size'],'boxes':a['boxes_xywh'],
            'regions_gt':experiment.annotation.box_regions(a['boxes_xywh']),
            'label_basis':'human_dev_box_centers','candidate_cells':selected,'cached_first':route['first'],
            'previous_candidate_prediction':previous[k],'normal_reference_image':source['references'][r['image']]})
    wanted={('target',r['image']) for r in rows}|{('reference',r['normal_reference_image']) for r in rows}
    data=dict(protocol=experiment.PROTOCOL,purpose='oracle_diagnosis_only',new_holdout=False,
        source_run_sha256=SOURCE_RUN_SHA,annotation_sha256=ANNOTATION_SHA,
        annotation_manifest_identity=session.identity,source_snapshot_sha256=manifest['source_snapshot_sha256'],
        source_total=25,source_boxed=24,source_unknown=1,rows=rows,excluded=excluded,
        image_manifest=[r for r in source['image_manifest'] if (r['role'],r['member']) in wanted])
    experiment.validate(data)
    if len(excluded)!=5:raise ValueError('frozen20/5 reconciliation changed')
    return data


def package(args):
    data=dataset(args)
    # Import traversal reuses the already-tested native adapter dependencies, no model files/images.
    payloads={n:(ROOT/n).read_bytes() for n in experiment.old.prior.shared.code_hashes()}
    payloads.update({'inputs/dataset.json':(json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode(),
        'inputs/annotations.json':args.annotations.read_bytes(),'inputs/annotation_manifest.json':args.annotation_manifest.read_bytes()})
    return payloads


def build(args):
    payloads=package(args); hashes={n:experiment.sha(b) for n,b in payloads.items()}; buf=io.BytesIO()
    with ZipFile(buf,'w',ZIP_DEFLATED) as z:
        for n,b in sorted(payloads.items()):
            info=ZipInfo(n,date_time=(2026,10,21,0,0,0));info.compress_type=ZIP_DEFLATED;z.writestr(info,b)
    source=buf.getvalue();source_sha=experiment.sha(source);cells=[]
    def cell(kind,text):
        c=dict(id='q3-gt-crop-'+str(len(cells)),cell_type=kind,metadata={},source=textwrap.dedent(text).strip().splitlines(keepends=True))
        if kind=='code':ast.parse(''.join(c['source']));c.update(execution_count=None,outputs=[])
        cells.append(c)
    cell('markdown','''
    # Q3 누락 진단: 기존 후보 / 사람 정답 칸 / 사람 정답 박스 확대

    파일: `lvlm_q3_gt_crop_diagnostic_dev_colab.ipynb`

    **L4/A100 → 기본 설정 유지 → 모두 실행 → Drive 연결 허용.**
    실제 추론용 코랩입니다. 준비/CPU 검증과 실제 효과는 별개이며, 실제 효과는 실행 후 확인합니다.

    - 완료한25장 실제 표시를 내장했습니다(박스24/판단불가1). 다시 표시하거나 결과 ZIP을 입력할 필요 없습니다.
    - 이전 후보 확대를 실행했고 위치 정답이 있는 동일20장만 세 조건으로 비교합니다. 판단불가1/후보보류1/이전경로밖3은 제외하고 이유를 기록합니다.
    - 기존 후보 칸 확대 / 사람 정답 칸 확대 / 정답 박스에 각 방향25% 여백을 붙인 확대. 정상 참조+원본+확대판3이미지, 동일3/3/2타일·80토큰·조건당1회입니다.
    - **최대60회 LVLM, DINO/재학습/Q1/Q2/gold0.** 기존 판별기·가림 실험을 다시 돌리지 않습니다.
    - **정답을 입력하는 진단 실험입니다. 최종 Agent나 독립 holdout 점수가 아닙니다.** 정상 사진이 없어 정상 오탐률은 평가하지 않습니다.
    - 정답 칸 조건으로 위치 선정 문제를, 정답 박스 조건으로 크기/문맥 문제를 살핍니다. 원본9칸 좌표로 답하며 무효는 오답으로 유지합니다.
    - 확대판과 질문에는 정답 칸 번호·박스 좌표를 적지 않습니다. 세 조건은 완전히 같은 질문을 사용하며, 정답은 확대 부위 선택에만 씁니다.
    - 같은 코드/입력/모델/폴더면 성공 기록에서 재개합니다. 실행 오류만 재시도합니다. 결과가 틀렸다고 반복하지 않습니다.

    출력: `q3_gt_crop_diagnostic_dev20_results.zip`. 기존 원본 데이터 ZIP 두 개를 자동 검색합니다.
    ''')
    cell('code','''
    RUN_FOLDER = "lvlm_insplad_results/q3_gt_crop_diagnostic_dev20_v1"
    SUP_ZIP_PATH = ""
    UNSUP_ZIP_PATH = ""
    ''')
    cell('code',f'''
    import base64,hashlib,io,json,sys,zipfile
    from pathlib import Path
    from google.colab import drive
    drive.mount('/content/drive',force_remount=False)
    DRIVE=Path('/content/drive/MyDrive')
    SOURCE_SHA256={source_sha!r}
    SOURCE_MANIFEST={hashes!r}
    SOURCE_BYTES=base64.b64decode({base64.b64encode(source).decode()!r})
    assert hashlib.sha256(SOURCE_BYTES).hexdigest()==SOURCE_SHA256
    CODE=Path('/content/q3_gt_crop_'+SOURCE_SHA256[:12])
    with zipfile.ZipFile(io.BytesIO(SOURCE_BYTES)) as archive:
        assert set(archive.namelist())==set(SOURCE_MANIFEST)
        for name in archive.namelist():
            dest=CODE/name;content=archive.read(name)
            assert dest.resolve().is_relative_to(CODE.resolve())
            assert hashlib.sha256(content).hexdigest()==SOURCE_MANIFEST[name]
            if dest.exists() and dest.read_bytes()!=content:raise RuntimeError('내장 코드와 다른 파일; 새 런타임을 사용하세요.')
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
    sys.path[:0]=[str(CODE/'scripts'),str(CODE/'src')]
    import run_q3_gt_crop_diagnostic_dev as experiment
    DATASET=CODE/'inputs/dataset.json';OUT=DRIVE/RUN_FOLDER
    assert OUT.name.startswith(experiment.PREFIX)
    print('진단 전용: 동일20장 ×3조건=최대60회. SOURCE',SOURCE_SHA256)
    ''')
    cell('code','''
    import subprocess
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<20*1024**3:
        raise RuntimeError('L4/A100 VRAM20GB 이상 런타임을 선택하세요.')
    subprocess.run([sys.executable,'-m','pip','install','-q','transformers==5.14.0','accelerate==1.14.0',
        'sentencepiece==0.2.2','numpy>=1.26','pillow>=10','timm'],check=True)
    def find_zip(name,size,override):
        if override:
            path=Path(override)
            if not path.is_file() or path.stat().st_size!=size:raise FileNotFoundError(str(path))
            return path
        for path in [DRIVE/name,DRIVE/'lvlm_insplad_data'/name]:
            if path.is_file() and path.stat().st_size==size:return path
        matches=[p for p in DRIVE.rglob(name) if p.is_file() and p.stat().st_size==size]
        if not matches:raise FileNotFoundError(name)
        return sorted(matches)[0]
    SUP=find_zip('supervised_fault_classification.zip',874753167,SUP_ZIP_PATH)
    UNSUP=find_zip('unsupervised_anomaly_detection.zip',1170967719,UNSUP_ZIP_PATH)
    from huggingface_hub import snapshot_download
    snapshot=snapshot_download(experiment.old.prior.MODEL_ID,revision=experiment.old.prior.MODEL_REVISION)
    command=[sys.executable,str(CODE/'scripts/run_q3_gt_crop_diagnostic_dev.py'),
        '--dataset',str(DATASET),'--sup-zip',str(SUP),'--unsup-zip',str(UNSUP),
        '--model-path',snapshot,'--out-dir',str(OUT)]
    LOG=DRIVE/'q3_gt_crop_diagnostic_dev20_console.log'
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    with LOG.open('a',encoding='utf-8') as log:
        for line in process.stdout:print(line,end='');log.write(line);log.flush()
    RETURN_CODE=process.wait();print('RUNNER_RETURN_CODE',RETURN_CODE)
    ''')
    cell('code','''
    from google.colab import files
    RESULT=DRIVE/'q3_gt_crop_diagnostic_dev20_results.zip'
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
    if RETURN_CODE or not status.get('complete'):print('실행 오류가 있습니다. ZIP의 원시 기록을 확인해야 합니다.')
    elif status.get('invalid_new_outputs'):print('평가 완료이나 무효 출력은 오답으로 포함했습니다.')
    print('진단 전용 / Agent 점수 아님. ZIP',RESULT,'SHA256',hashlib.sha256(RESULT.read_bytes()).hexdigest())
    files.download(str(RESULT))
    ''')
    notebook=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'colab':{'provenance':[]}},nbformat=4,nbformat_minor=5)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    args.evidence_dir.mkdir(parents=True,exist_ok=True);(args.evidence_dir/'source_bundle.zip').write_bytes(source)
    receipt=dict(source_bundle_sha256=source_sha,notebook_sha256=experiment.old.prior.probe.sha_file(args.output),
        source_files=hashes,compiled_code_cells=4,paired_images=20,max_lvlm_calls=60,new_neural_calls=0,
        diagnostic_only=True,annotation_sha256=ANNOTATION_SHA,source_run_sha256=SOURCE_RUN_SHA)
    (args.evidence_dir/'notebook_build.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_files'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source-run','annotations','annotation-manifest','output','evidence-dir']:p.add_argument('--'+name,type=Path,required=True)
    build(p.parse_args())

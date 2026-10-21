"""Seal audited PR63 Q3 context and frozen head into a self-contained real-run Colab."""
import argparse
import ast
import base64
import io
import json
from pathlib import Path
import textwrap
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED

import run_q3_missing_occlusion_dev as experiment

SOURCE_RUN_SHA = '38b050835bbb786ff849f944e0ab2d95d7a1d652f1a617982a22025c93ee82de'


def package(archive):
    if experiment.prior.probe.sha_file(archive)!=SOURCE_RUN_SHA:raise ValueError('use the audited original PR63 ZIP')
    with ZipFile(archive) as z:
        read=lambda n:json.loads(z.read(n))
        meta=read('run/run_metadata.json');status=read('run/run_status.json')
        if not status['complete'] or meta['backend']!='hf' or meta['fixture'] or meta['gold_used'] or meta['row_errors']:
            raise ValueError('source must be a complete actual dev run')
        if meta['model_revision']!=experiment.prior.MODEL_REVISION:raise ValueError('source model revision changed')
        source=read('run/dataset_snapshot.json');pred=read('run/predictions.json');calls=read('run/calls.json')
        rows=[json.loads(line) for line in z.read('run/q3_dev_partial.jsonl').decode().splitlines()]
        tools=read('run/tool_calls.json');head_bytes=z.read('run/frozen_detector.json')
        if experiment.sha(head_bytes)!=experiment.prior.DETECTOR_SHA256:raise ValueError('wrong frozen head')
        for row in rows:
            rid=row['id'];core=row['core_id'];first=calls[rid+'/initial']
            if first['error'] or experiment.prior.spatial.parse_json(first['output'],'regions')!=first['prediction']:
                raise ValueError('invalid source execution/parsing')
            if first['prediction']!=pred['q3']['lvlm_only'][core]:raise ValueError('source first answer mismatch')
            review=calls.get(rid+'/selective_agent',first)
            if review['error'] or experiment.prior.spatial.parse_json(review['output'],'regions')!=pred['q3']['selective_agent'][core]:
                raise ValueError('source review mismatch')
        data={'protocol':experiment.PROTOCOL,'source_run_sha256':SOURCE_RUN_SHA,'rows':rows,
            'cached_direct':pred['q3']['lvlm_only'],'cached_cue_review':pred['q3']['selective_agent'],
            'experts':tools,'references':source['references'],'image_manifest':source['image_manifest'],
            'detector_sha256':experiment.sha(head_bytes),'source_model_revision':meta['model_revision']}
        experiment.validate(data,json.loads(head_bytes),experiment.sha(head_bytes))
        payloads={'inputs/dataset.json':(json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode(),
                  'inputs/frozen_detector.json':head_bytes}
        active=[r for r in rows if experiment.routed(r,data)]
        if len(active)!=23:raise ValueError('audited own-Q3 routing changed')
        for row in active:
            core=row['core_id'];tool=tools[core]
            content=z.read('run/tool_features/'+tool['image_sha256']+'.npy')
            if experiment.sha(content)!=tool['feature_sha256']:raise ValueError('source feature checksum mismatch')
            payloads['inputs/original_features/'+core+'.npy']=content
    payloads.update({name:(experiment.ROOT/name).read_bytes() for name in experiment.prior.shared.code_hashes()})
    return payloads


def build(args):
    payloads=package(args.source_run);manifest={n:experiment.sha(b) for n,b in payloads.items()}
    buf=io.BytesIO()
    with ZipFile(buf,'w',ZIP_DEFLATED) as z:
        for n,b in sorted(payloads.items()):
            info=ZipInfo(n,date_time=(2026,10,21,0,0,0));info.compress_type=ZIP_DEFLATED;z.writestr(info,b)
    payload=buf.getvalue();source_sha=experiment.sha(payload);cells=[]
    def cell(kind,text):
        c={'id':'q3-occlusion-'+str(len(cells)),'cell_type':kind,'metadata':{},'source':textwrap.dedent(text).strip().splitlines(keepends=True)}
        if kind=='code':ast.parse(''.join(c['source']));c.update(execution_count=None,outputs=[])
        cells.append(c)
    cell('markdown','''
    # 다음 Q3 실험: 누락 판별기의 가림 반응 → 원본 후보 부위 확대 → 위치 검토

    파일: `lvlm_q3_missing_occlusion_dev_colab.ipynb`

    **L4/A100 → 모두 실행 → Drive 연결 허용. 실제 추론 실행용입니다. 새 방법의 실제 효과는 아직 미검증입니다.**
    기존 원본 ZIP 두 개를 자동 검색합니다. 앞선 실험을 다시 돌리거나 토큰·추가 라벨을 입력할 필요가 없습니다.

    - 이전 실제 dev240의 Q3와 동결 판별기 결과를 내장했습니다. Q1/Q2/gold 추론·재학습·임계값 변경0.
    - 기존 Q3가 빈 위치이고 유리 절연체 누락 판별기가 양성인 동일23장에만 새 위치 도구를 호출합니다.
    - 원본 전체 사진의9칸을 각각 흐리기/평균색 가리기로 검사: 동결 DINO 특징 최대414회(23×9×2).
    - 두 가림 방식에서 점수가 함께 떨어진 최대2칸을 후보로 골라, 원본의 해당 부위를 확대해 LVLM이 위치를 다시 판단합니다.
    - 같은 개수·크기의 고정5,2번 칸을 확대하는 대조군도 실행합니다. 새 LVLM은 최대46회(방법23+대조23), 각3이미지/동일예산입니다.
    - 두 가림 방식이 동의하는 칸이 없으면 양쪽 모두 추가 판단을 보류합니다. 기존 양성/무효 응답을 사후 재파싱하지 않습니다.
    - **가림 반응은 약한 위치 후보이며 정답 위치가 아닙니다.** 배경·가림 인공물 때문에 틀릴 수 있습니다. LVLM은 후보 밖 위치나 빈 집합도 답할 수 있습니다.
    - 이 dev에는 결함 위치 정답이 없어 위치 정확도는null입니다. 미탐·정상 오탐/무효율과 원본 후보 crops/maps를 출력합니다.
    - 코드/입력/모델/폴더가 같으면 중간 저장부터 재개합니다. 오류만 재시도하며 무효는 오답으로 유지합니다.
    ''')
    cell('code','''
    RUN_FOLDER = "lvlm_insplad_results/q3_missing_occlusion_dev240_v1"
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
    SOURCE_MANIFEST={manifest!r}
    SOURCE_BYTES=base64.b64decode({base64.b64encode(payload).decode()!r})
    assert hashlib.sha256(SOURCE_BYTES).hexdigest()==SOURCE_SHA256
    CODE=Path('/content/q3_missing_occlusion_'+SOURCE_SHA256[:12])
    with zipfile.ZipFile(io.BytesIO(SOURCE_BYTES)) as archive:
        assert set(archive.namelist())==set(SOURCE_MANIFEST)
        for name in archive.namelist():
            target=CODE/name;content=archive.read(name)
            assert target.resolve().is_relative_to(CODE.resolve())
            assert hashlib.sha256(content).hexdigest()==SOURCE_MANIFEST[name]
            if target.exists() and target.read_bytes()!=content:raise RuntimeError('내장 코드와 다른 런타임 파일; 새 런타임을 사용하세요.')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    sys.path[:0]=[str(CODE/'scripts'),str(CODE/'src')]
    import run_q3_missing_occlusion_dev as experiment
    DATASET=CODE/'inputs/dataset.json';DETECTOR=CODE/'inputs/frozen_detector.json'
    OUT=DRIVE/RUN_FOLDER
    assert OUT.name.startswith(experiment.PREFIX)
    print('Q3 후보 부위 실제 실험: 같은23장, 최대414 DINO 특징 +46 LVLM 호출. SOURCE',SOURCE_SHA256)
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
    snapshot=snapshot_download(experiment.prior.MODEL_ID,revision=experiment.prior.MODEL_REVISION)
    command=[sys.executable,str(CODE/'scripts/run_q3_missing_occlusion_dev.py'),
        '--dataset',str(DATASET),'--detector',str(DETECTOR),'--sup-zip',str(SUP),'--unsup-zip',str(UNSUP),
        '--model-path',snapshot,'--out-dir',str(OUT)]
    LOG=DRIVE/'q3_missing_occlusion_dev240_console.log'
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    with LOG.open('a',encoding='utf-8') as log:
        for line in process.stdout:print(line,end='');log.write(line);log.flush()
    RETURN_CODE=process.wait();print('RUNNER_RETURN_CODE',RETURN_CODE)
    ''')
    cell('code','''
    from google.colab import files
    RESULT=DRIVE/'q3_missing_occlusion_dev240_results.zip'
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
    elif status.get('invalid_new_outputs'):print('평가 완료이나 무효 출력이 있어 오답으로 포함했습니다.')
    print('ZIP',RESULT,'SHA256',hashlib.sha256(RESULT.read_bytes()).hexdigest())
    files.download(str(RESULT))
    ''')
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'colab':{'provenance':[]}},'nbformat':4,'nbformat_minor':5}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    args.evidence_dir.mkdir(parents=True,exist_ok=True);(args.evidence_dir/'source_bundle.zip').write_bytes(payload)
    receipt={'source_bundle_sha256':source_sha,'notebook_sha256':experiment.prior.probe.sha_file(args.output),
        'source_files':manifest,'compiled_code_cells':4,'dev_images':240,'routed_images':23,'max_feature_calls':414,'max_lvlm_calls':46,'new_neural_calls':0}
    (args.evidence_dir/'notebook_build.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_files'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source-run','output','evidence-dir']:p.add_argument('--'+key,type=Path,required=True)
    build(p.parse_args())

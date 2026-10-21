#!/usr/bin/env python3
"""Package a frozen trained head and verified dev inputs with exact runtime source."""
import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile

import run_selective_missing_expert_agent_dev as experiment


def build(args):
    source=json.loads(args.source_data.read_text('utf-8'))
    experiment.shared.validate_dataset(source)
    artifact=json.loads(args.detector.read_text('utf-8'))
    metadata=json.loads(args.probe_metadata.read_text('utf-8'));status=json.loads(args.probe_status.read_text('utf-8'))
    head_sha=hashlib.sha256(args.detector.read_bytes()).hexdigest()
    if not status.get('complete') or not metadata.get('complete') or metadata.get('gold_scored') or (
        metadata.get('frozen_detector_sha256')!=head_sha or status.get('frozen_detector_sha256')!=head_sha):
        raise ValueError('only the complete, original frozen probe may be packaged')
    data={'protocol':experiment.PROTOCOL,'rows':source['rows'],
        'q1_fixed':{r['id']:source['initial'][r['id']] for r in source['rows'] if r['question_type']==experiment.agent.Q1},
        'references':source['references'],'image_manifest':source['image_manifest'],'detector_sha256':head_sha,
        'probe_provenance':metadata,'source_data_sha256':hashlib.sha256(args.source_data.read_bytes()).hexdigest()}
    experiment.validate_inputs(data,artifact,head_sha)
    payloads={n:(experiment.ROOT/n).read_bytes() for n in experiment.shared.code_hashes()}
    payloads['inputs/dataset.json']=(json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode()
    payloads['inputs/frozen_detector.json']=args.detector.read_bytes()
    manifest={n:hashlib.sha256(b).hexdigest() for n,b in payloads.items()}
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,content in sorted(payloads.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,10,21,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,content)
    payload=buf.getvalue();source_sha=hashlib.sha256(payload).hexdigest()
    cells=[]
    def cell(kind,text):
        c={'id':f'selective-expert-{len(cells):02d}','cell_type':kind,'metadata':{},'source':textwrap.dedent(text).strip().splitlines(keepends=True)}
        if kind=='code':ast.parse(''.join(c['source']));c.update(execution_count=None,outputs=[])
        cells.append(c)
    cell('markdown','''
    # 3번 — 정상 Reference + 누락 Vision Expert + LVLM 통합 Agent (dev240)

    **기능 제작·CPU 검증 단계. 실제 새 통합 효과는 아직 미검증입니다.**

    파일: `lvlm_selective_missing_expert_agent_dev_colab.ipynb`

    L4/A100 선택 → 모두 실행 → Drive 연결 허용. **실제 추론 실행용**입니다. GitHub 토큰·추가 라벨링이 필요 없습니다.
    기존 원본 ZIP 두 개를 자동 검색합니다. 1·2번을 다시 돌리거나 판별기를 다시 학습하지 않습니다.

    - 고정된 dev240에서 새 Q2 유형 + Q3 9칸 복수 위치 첫 판단480호출. Q1은 기존 답 그대로/호출0.
    - 유리 절연체에서 해당 질문의 Q2가 누락이 아니거나 Q3가 빈 위치일 때만 동결 DINOv2 누락 판별기 호출.
    - 판별기 양성일 때만 같은 질문의 LVLM 확인1회 + 동일 사례 재질문 대조1회. 신규 LVLM480~680호출, DINO 최대50장(공유 캐시).
    - 네 조건: LVLM 단독 / 판별기 양성의 Q2 단순 결합 / 도구 신호를 받고 재검토 / 같은 라우팅·횟수의 재질문.
    - Q2·Q3는 서로 답을 전달하지 않습니다. 도구 음성으로 기존 누락/위치를 취소하지 않습니다. 도구는 위치를 출력하지 않습니다.
    - 이 도구는 glass-insulator 정상/누락 라벨을 학습한 **지도학습 보조 조건**이며, 다른 자산/결함에 일반화했다고 주장하지 않습니다.
    - 이미 분석한 dev입니다. 양성 위치 GT가 없어 위치 정확도는 null; Q3 유무/미탐·정상 오탐과 유형을 평가합니다. gold·재학습·임계값 튜닝0.
    - 무효 출력은 원시 그대로 유지하고 오답으로 집계하며, 추론/도구 오류가 난 조건은 비교에서 제외합니다. ZIP 다운로드만으로 성공을 판단하지 마세요.
    - 출력 폴더/코드/입력/모델이 같으면 중간 저장을 재사용합니다. 오류만 재시도하고 무효를 정답이 나올 때까지 다시 묻지 않습니다.
    ''')
    cell('code','''
    N_IMAGES = 240
    RUN_FOLDER = f"lvlm_insplad_results/selective_missing_expert_agent_dev{N_IMAGES}_v1"
    SUP_ZIP_PATH = ""     # 자동 검색 실패 시에만 실제 Drive 경로
    UNSUP_ZIP_PATH = ""
    ''')
    cell('code',f'''
    import base64, hashlib, io, json, sys, zipfile
    from pathlib import Path
    from google.colab import drive
    drive.mount('/content/drive',force_remount=False)
    DRIVE=Path('/content/drive/MyDrive')
    SOURCE_SHA256={source_sha!r}
    SOURCE_MANIFEST={manifest!r}
    SOURCE_BYTES=base64.b64decode({base64.b64encode(payload).decode()!r})
    assert hashlib.sha256(SOURCE_BYTES).hexdigest()==SOURCE_SHA256
    CODE=Path('/content/selective_missing_expert_'+SOURCE_SHA256[:12])
    with zipfile.ZipFile(io.BytesIO(SOURCE_BYTES)) as archive:
        assert set(archive.namelist())==set(SOURCE_MANIFEST)
        for name in archive.namelist():
            target=CODE/name
            assert target.resolve().is_relative_to(CODE.resolve())
            content=archive.read(name)
            assert hashlib.sha256(content).hexdigest()==SOURCE_MANIFEST[name]
            if target.exists() and target.read_bytes()!=content:
                raise RuntimeError('런타임 파일이 내장 코드와 다릅니다. 새 런타임을 사용하세요.')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    sys.path[:0]=[str(CODE/'scripts'),str(CODE/'src')]
    import run_selective_missing_expert_agent_dev as experiment
    assert N_IMAGES in (40,240), '기본240장. 작은 실행은40장/별도 폴더에서만 가능합니다.'
    DATASET=CODE/'inputs/dataset.json'
    DETECTOR=CODE/'inputs/frozen_detector.json'
    assert hashlib.sha256(DETECTOR.read_bytes()).hexdigest()==experiment.DETECTOR_SHA256
    OUT=DRIVE/RUN_FOLDER
    assert OUT.name.startswith('selective_missing_expert_agent_dev')
    print('실제 통합 실험 준비:',N_IMAGES,'장 / Q2·Q3 직접 질문',2*N_IMAGES,'회 + 필요한 검토/대조')
    print('동결 판별기만 사용, 재학습0. SOURCE',SOURCE_SHA256)
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
        for path in (DRIVE/name,DRIVE/'lvlm_insplad_data'/name):
            if path.is_file() and path.stat().st_size==size:return path
        matches=[p for p in DRIVE.rglob(name) if p.is_file() and p.stat().st_size==size]
        if not matches:raise FileNotFoundError(name)
        return sorted(matches)[0]
    SUP=find_zip('supervised_fault_classification.zip',874753167,SUP_ZIP_PATH)
    UNSUP=find_zip('unsupervised_anomaly_detection.zip',1170967719,UNSUP_ZIP_PATH)
    from huggingface_hub import snapshot_download
    snapshot=snapshot_download(experiment.MODEL_ID,revision=experiment.MODEL_REVISION)
    # DINOv2 is loaded lazily only if the own-question route requests it.
    command=[sys.executable,str(CODE/'scripts/run_selective_missing_expert_agent_dev.py'),
        '--dataset',str(DATASET),'--detector',str(DETECTOR),'--sup-zip',str(SUP),'--unsup-zip',str(UNSUP),
        '--model-path',snapshot,'--out-dir',str(OUT),'--n-images',str(N_IMAGES)]
    LOG=DRIVE/f'selective_missing_expert_agent_dev{N_IMAGES}_console.log'
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    with LOG.open('a',encoding='utf-8') as log:
        for line in process.stdout:
            print(line,end='');log.write(line);log.flush()
    RETURN_CODE=process.wait()
    print('RUNNER_RETURN_CODE',RETURN_CODE)
    ''')
    cell('code','''
    from google.colab import files
    RESULT=DRIVE/f'selective_missing_expert_agent_dev{N_IMAGES}_results.zip'
    with zipfile.ZipFile(RESULT,'w',zipfile.ZIP_DEFLATED) as archive:
        if OUT.exists():
            for path in sorted(OUT.rglob('*')):
                if path.is_file() and '_images' not in path.relative_to(OUT).parts:
                    archive.write(path,'run/'+path.relative_to(OUT).as_posix())
        archive.writestr('source_bundle.zip',SOURCE_BYTES)
        archive.writestr('source_manifest.json',json.dumps(SOURCE_MANIFEST,indent=2))
        if LOG.exists():archive.write(LOG,'console.log')
        archive.writestr('runner_return_code.json',json.dumps({'return_code':RETURN_CODE}))
    status_path=OUT/'run_status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {'complete':False,'status':'runner failed before status'}
    print('RUN STATUS',status)
    if RETURN_CODE!=0 or not status.get('complete'):
        print('오류가 난 조건은 비교 불가합니다. 원시 기록은 ZIP에 보존했습니다.')
    elif status.get('invalid_lvlm_outputs'):
        print('평가 완료이나 무효 출력이 있습니다. 무효는 오답으로 집계했습니다:',status['invalid_lvlm_outputs'])
    print('ZIP',RESULT,'SHA256',hashlib.sha256(RESULT.read_bytes()).hexdigest())
    files.download(str(RESULT))
    ''')
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'colab':{'provenance':[]}},'nbformat':4,'nbformat_minor':5}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    args.evidence_dir.mkdir(parents=True,exist_ok=True)
    (args.evidence_dir/'source_bundle.zip').write_bytes(payload)
    receipt={'source_bundle_sha256':source_sha,'source_files':manifest,'notebook_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'compiled_code_cells':4,'images':240,'first_lvlm_calls':480,'maximum_lvlm_calls':680,'new_neural_calls':0}
    (args.evidence_dir/'notebook_build.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_files'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source-data','detector','probe-metadata','probe-status','output','evidence-dir'):
        p.add_argument('--'+key,type=Path,required=True)
    build(p.parse_args())

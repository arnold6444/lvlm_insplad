#!/usr/bin/env python3
"""Build a self-contained, source/input-pinned real-inference Colab notebook."""
import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile

import run_q2_conflict_evidence_dev as runner


def build(dataset, output, evidence_dir):
    data=runner.validate_dataset(json.loads(dataset.read_text('utf-8')))
    # Capture only the imported runtime dependency closure, not every repository script.
    files={name:(runner.ROOT/name).read_bytes() for name in runner.code_hashes()}
    files['inputs/dataset.json']=dataset.read_bytes()
    source_manifest={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name,payload in sorted(files.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,10,21,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,payload)
    payload=buf.getvalue();sha=hashlib.sha256(payload).hexdigest()
    selected,_=runner.refined.select_rows(data['rows'],40,42)
    conflicts=sum(r['question_type']==runner.agent.Q2 and data['initial'][r['id']]!=data['reviewed'][r['id']]['predicted_choice'] for r in selected)
    cells=[]
    def cell(kind,text):
        entry={'cell_type':kind,'id':f'q2-conflict-{len(cells):02d}','metadata':{},'source':textwrap.dedent(text).strip().splitlines(keepends=True)}
        if kind=='code':
            ast.parse(''.join(entry['source']));entry.update(execution_count=None,outputs=[])
        cells.append(entry)
    cell('markdown',f'''
    # Q2 충돌 확인 + 같은 호출 수 재질문 — dev40

    **준비/CPU 검증 완료. 새 구조의 실제 모델 효과는 아직 미검증입니다.**

    실행 파일: `lvlm_q2_conflict_evidence_dev_colab.ipynb`

    L4/A100 GPU 선택 → 모두 실행 → Drive 연결 허용. **이 노트북은 실제 모델 추론을 실행합니다.**
    기존 dev40 답을 재사용하고, 첫 판단과 정의 재검토가 다른 {conflicts}건에서만 새 {4*conflicts}회 호출합니다.
    결과 파일: `q2_conflict_evidence_dev40_results.zip`. 새 GitHub 토큰/수동 라벨 입력은 필요 없습니다.

    - 후보별 사진 지지 확인2회 vs 순차 재질문2회. 두 정책은 동일 사례·모델·이미지·최대출력80토큰·호출 수입니다. 질문 내용/길이·실제 시간은 다릅니다.
    - 한 후보만 지지될 때 그 답을 사용. 둘 다 지지/둘 다 미지지는 초기 답 유지. 무효/오류는 숨기지 않고 정책 제외 및 원시 ZIP 보존.
    - 초기/정의 재검토/기존 cue 답도 비교합니다. 초기 조건보다만 좋아져도 성공이라고 하지 않습니다.
    - 과거 240장의 77충돌: 정답→오답18, 오답→정답42, 다른 오답17. 이미 분석한 dev이며 최종 holdout이 아닙니다.
    - Q1 신규 호출0·원래 답 고정. Q3/gold/학습0. 과거 두 판단의 모델 revision은 미기록이며 새 비교 호출만 같은 snapshot으로 고정합니다.
    - 끊기면 같은 설정으로 다시 실행. 저장된 성공/무효 호출은 재사용하고 오류만 재시도합니다. 추가200장은 이40장 결과 검토 후 따로 준비합니다.
    - 원본 ZIP 두 개는 Drive에 있어야 합니다. 결과는 별도 폴더에 저장하며 원본 실험을 덮어쓰지 않습니다.
    ''')
    cell('code','''
    N_IMAGES = 40       # 이번 파일은 고정된 첫40장만 실행
    RUN_FOLDER = "lvlm_insplad_results/q2_conflict_evidence_dev40_v1"
    SUP_ZIP_PATH = ""   # 자동 검색 실패 시에만 실제 Drive 경로 입력
    UNSUP_ZIP_PATH = ""
    ''')
    cell('code',f'''
    import base64, hashlib, io, json, sys, zipfile
    from pathlib import Path
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    DRIVE = Path('/content/drive/MyDrive')
    SOURCE_SHA256 = {sha!r}
    SOURCE_MANIFEST = {source_manifest!r}
    PAYLOAD = base64.b64decode({base64.b64encode(payload).decode()!r})
    assert hashlib.sha256(PAYLOAD).hexdigest() == SOURCE_SHA256
    CODE = Path('/content/q2_conflict_' + SOURCE_SHA256[:12])
    with zipfile.ZipFile(io.BytesIO(PAYLOAD)) as archive:
        assert set(archive.namelist()) == set(SOURCE_MANIFEST)
        for name in archive.namelist():
            target = CODE / name
            assert target.resolve().is_relative_to(CODE.resolve())
            content = archive.read(name)
            assert hashlib.sha256(content).hexdigest() == SOURCE_MANIFEST[name]
            if target.exists() and target.read_bytes() != content:
                raise RuntimeError('런타임 소스가 다릅니다. 새 런타임으로 실행하세요.')
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(content)
    sys.path[:0] = [str(CODE/'scripts'), str(CODE/'src')]
    import run_q2_conflict_evidence_dev as experiment
    DATASET = CODE/'inputs/dataset.json'
    assert N_IMAGES == 40, '확대 실행은 40장 결과 검토 후 별도 파일로 진행합니다.'
    OUT = DRIVE / RUN_FOLDER
    assert OUT.name.startswith('q2_conflict_evidence_dev')
    print('실제 추론 준비:', N_IMAGES, '장 / 충돌{conflicts}건 / 신규{4*conflicts}호출. 소스:', SOURCE_SHA256)
    ''')
    cell('code','''
    import subprocess
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 20*1024**3:
        raise RuntimeError('L4/A100 VRAM20GB 이상 런타임을 선택하세요.')
    subprocess.run([sys.executable,'-m','pip','install','-q','transformers==5.14.0',
        'accelerate==1.14.0','sentencepiece==0.2.2','numpy>=1.26','pillow>=10','timm'],check=True)
    def find_zip(name,size,override):
        if override:
            path=Path(override)
            if not path.is_file() or path.stat().st_size!=size:
                raise FileNotFoundError('입력한 ZIP 경로/크기 확인: '+str(path))
            return path
        for path in (DRIVE/name,DRIVE/'lvlm_insplad_data'/name):
            if path.is_file() and path.stat().st_size==size:
                return path
        matches=[p for p in DRIVE.rglob(name) if p.is_file() and p.stat().st_size==size]
        if not matches:
            raise FileNotFoundError(name+'를 Drive에서 찾을 수 없습니다.')
        return sorted(matches)[0]
    SUP=find_zip('supervised_fault_classification.zip',874753167,SUP_ZIP_PATH)
    UNSUP=find_zip('unsupervised_anomaly_detection.zip',1170967719,UNSUP_ZIP_PATH)
    from huggingface_hub import snapshot_download
    snapshot=snapshot_download(experiment.MODEL_ID,revision=experiment.MODEL_REVISION)
    command=[sys.executable,str(CODE/'scripts/run_q2_conflict_evidence_dev.py'),
        '--dataset',str(DATASET),'--sup-zip',str(SUP),'--unsup-zip',str(UNSUP),
        '--model-path',snapshot,'--out-dir',str(OUT),'--n-images',str(N_IMAGES)]
    # Runner failures still reach the ZIP export cell, preserving diagnostics.
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    with (DRIVE/'q2_conflict_evidence_dev40_console.log').open('a',encoding='utf-8') as log:
        for line in process.stdout:
            print(line,end='');log.write(line);log.flush()
    RETURN_CODE=process.wait()
    print('RUNNER_RETURN_CODE',RETURN_CODE)
    ''')
    cell('code','''
    from google.colab import files
    RESULT=DRIVE/'q2_conflict_evidence_dev40_results.zip'
    with zipfile.ZipFile(RESULT,'w',zipfile.ZIP_DEFLATED) as archive:
        if OUT.exists():
            for path in sorted(OUT.rglob('*')):
                if path.is_file() and '_images' not in path.relative_to(OUT).parts:
                    archive.write(path,'run/'+path.relative_to(OUT).as_posix())
        archive.writestr('source_bundle.zip',PAYLOAD)
        archive.writestr('source_manifest.json',json.dumps(SOURCE_MANIFEST,indent=2))
        log=DRIVE/'q2_conflict_evidence_dev40_console.log'
        if log.exists():archive.write(log,'console.log')
        archive.writestr('runner_return_code.json',json.dumps({'return_code':RETURN_CODE}))
    print('ZIP',RESULT,'SHA256',hashlib.sha256(RESULT.read_bytes()).hexdigest())
    status_path=OUT/'run_status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {'complete':False,'status':'runner failed before status'}
    print('RUN STATUS',status)
    if RETURN_CODE != 0 or not status.get('complete'):
        print('미완료/무효 정책을 성공으로 해석하지 마세요. ZIP에 원시 오류·응답을 보존했습니다.')
    files.download(str(RESULT))
    ''')
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'colab':{'provenance':[]}},'nbformat':4,'nbformat_minor':5}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    evidence_dir.mkdir(parents=True,exist_ok=True)
    (evidence_dir/'source_bundle.zip').write_bytes(payload)
    receipt={'source_bundle_sha256':sha,'files':source_manifest,'notebook_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'compiled_code_cells':sum(c['cell_type']=='code' for c in cells),'pilot_conflicts':conflicts,'new_calls':4*conflicts}
    (evidence_dir/'notebook_build.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evidence-dir',type=Path,required=True)
    args=p.parse_args()
    build(args.dataset,args.output,args.evidence_dir)

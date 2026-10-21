"""Build a source-pinned, tokenizer-gated presence-only diagnostic notebook."""
import argparse
import ast
import base64
import io
import json
from pathlib import Path
import textwrap
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED
import run_missing_presence_independent_dev as run

ROOT=Path(__file__).resolve().parents[1]
SOURCE_RESULT_SHA='e5baeefd1020ce8122e1b4946fa97a0d73e44b5b9887f93132286ba4f48e3b6e'
SOURCE_CAPSULE_SHA='cf89e774f465a46055a1f33103a48f0aac8e65e3e5c05f1cbe0d819edea830a2'


def package(args):
    if run.crop.old.prior.probe.sha_file(args.source_run)!=SOURCE_RESULT_SHA:raise ValueError('audited actual PR66 ZIP required')
    with ZipFile(args.source_run) as z:
        meta=json.loads(z.read('run/run_metadata.json'));status=json.loads(z.read('run/run_status.json'))
        if not status['complete'] or meta['fixture'] or meta['backend']!='hf' or any(status['execution_errors_by_arm'].values()):
            raise ValueError('complete actual source required')
        raw=z.read('run/dataset_snapshot.json');data=json.loads(raw)
        if run.sha(raw)!=meta['input_sha256']:raise ValueError('source dataset changed')
        run.crop.validate(data)
        predictions=json.loads(z.read('run/predictions.json'));calls=json.loads(z.read('run/calls.json'))
        if len(calls)!=60 or meta['model_revision']!=run.gate.REVISION:raise ValueError('source execution changed')
        for rec in calls.values():
            if rec['error'] or run.crop.old.prior.spatial.parse_json(rec['output'],'regions')!=predictions[rec['arm']][rec['core_id']]:
                raise ValueError('source raw/prediction mismatch')
        capsule=z.read('source_bundle.zip')
        if run.sha(capsule)!=SOURCE_CAPSULE_SHA:raise ValueError('source capsule changed')
    data.update(experiment_protocol=run.PROTOCOL,source_presence_gt='all20_missing_cap',
        source_result_sha256=SOURCE_RESULT_SHA,previous_joint_q3=predictions)
    run.validate(data)
    payloads={n:(ROOT/n).read_bytes() for n in run.crop.old.prior.shared.code_hashes()}
    payloads['inputs/dataset.json']=(json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode()
    with ZipFile(io.BytesIO(capsule)) as z:
        for n in ['inputs/annotations.json','inputs/annotation_manifest.json']:payloads[n]=z.read(n)
    return payloads


def build(args):
    payloads=package(args);manifest={n:run.sha(b) for n,b in payloads.items()};buf=io.BytesIO()
    with ZipFile(buf,'w',ZIP_DEFLATED) as z:
        for n,b in sorted(payloads.items()):
            info=ZipInfo(n,date_time=(2026,10,21,0,0,0));info.compress_type=ZIP_DEFLATED;z.writestr(info,b)
    source=buf.getvalue();source_sha=run.sha(source);cells=[]
    def cell(kind,text):
        c=dict(id='missing-independent-'+str(len(cells)),cell_type=kind,metadata={},source=textwrap.dedent(text).strip().splitlines(keepends=True))
        if kind=='code':ast.parse(''.join(c['source']));c.update(execution_count=None,outputs=[])
        cells.append(c)
    cell('markdown','''
    # 누락 유무 독립 질문 — 이전 답/위치 질문 없이

    파일: **lvlm_missing_presence_independent_dev_colab.ipynb**

    **L4/A100 → 기본값 유지 → 모두 실행 → Drive 연결 허용.**
    같은20장×후보확대/정답칸/정답박스3조건, 최대60회입니다. 실제 효과는 이번 실행 후 확인합니다.

    - 누락/파손 유무만 {"present":true} 또는 {"present":false}로 묻습니다. 이전 답·대화 기록·위치 번호는 질문에 없습니다.
    - 정상 참조+전체 원본+확대판, 동일3/3/2타일·greedy80토큰. 확대판/실제25장 표시 내장, 추가 표시·이전 코랩 재실행 불필요.
    - GPU 모델을 받기 전에 CPU 토크나이저 검사. 공식 파일 SHA·전체 backend·인코딩158건·특수토큰이 다르면 중단합니다.
    - **Mistral regex 경고는 그대로 기록합니다.** 원본 토큰화와 일치하는지 확인하며 경고를 숨기거나 Mistral 수정을 강제로 적용하지 않습니다.
    - 실제 모델이 로드한 토크나이저도 검사합니다. FlashAttention 미설치/torch_dtype 경고는 숨기지 않고 동일 native 설정을 유지합니다.
    - 정답 부위를 쓰는 양성dev20 진단입니다. 위치 정확도/정상 오탐률/최종 Agent/새 holdout 성능은 평가하지 않습니다.
    - task와 이전 답 문구를 함께 바꾸므로 결과가 좋아져도 둘 중 무엇 때문인지는 분리할 수 없습니다.
    - 실행 오류만 이력과 함께 재시도. 무효는 FN으로 유지하고 성공/무효 기록은 재사용합니다.

    출력: **missing_presence_independent_dev20_results.zip**. 원본 데이터 ZIP 두 개는 Drive에서 자동 검색합니다.
    ''')
    cell('code','''
    RUN_FOLDER = "lvlm_insplad_results/missing_presence_independent_dev20_v1"
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
    SOURCE_BYTES=base64.b64decode({base64.b64encode(source).decode()!r})
    assert hashlib.sha256(SOURCE_BYTES).hexdigest()==SOURCE_SHA256
    CODE=Path('/content/missing_presence_independent_'+SOURCE_SHA256[:12])
    with zipfile.ZipFile(io.BytesIO(SOURCE_BYTES)) as archive:
        assert set(archive.namelist())==set(SOURCE_MANIFEST)
        for name in archive.namelist():
            dest=CODE/name;content=archive.read(name)
            assert dest.resolve().is_relative_to(CODE.resolve())
            assert hashlib.sha256(content).hexdigest()==SOURCE_MANIFEST[name]
            if dest.exists() and dest.read_bytes()!=content:raise RuntimeError('내장 코드와 다른 파일입니다. 새 런타임을 사용하세요.')
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
    sys.path[:0]=[str(CODE/'scripts'),str(CODE/'src')]
    import run_missing_presence_independent_dev as experiment
    DATASET=CODE/'inputs/dataset.json';OUT=DRIVE/RUN_FOLDER
    assert OUT.name.startswith(experiment.PREFIX)
    print('이전 답 없는 누락 유무 진단: 동일20장×3조건, 최대60회. SOURCE',SOURCE_SHA256)
    ''')
    cell('code','''
    import subprocess
    import torch
    LOG=DRIVE/'missing_presence_independent_dev20_console.log'
    PREFLIGHT=DRIVE/'missing_presence_independent_tokenizer_preflight_v1.json'
    RETURN_CODE=99
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<20*1024**3:
        raise RuntimeError('L4/A100 VRAM20GB 이상 런타임을 선택하세요.')
    subprocess.run([sys.executable,'-m','pip','install','-q','transformers==5.14.0','tokenizers==0.22.2',
        'accelerate==1.14.0','sentencepiece==0.2.2','numpy>=1.26','pillow>=10','timm'],check=True)
    def find_zip(name,size,override):
        if override:
            p=Path(override)
            if not p.is_file() or p.stat().st_size!=size:raise FileNotFoundError(str(p))
            return p
        for p in [DRIVE/name,DRIVE/'lvlm_insplad_data'/name]:
            if p.is_file() and p.stat().st_size==size:return p
        found=[p for p in DRIVE.rglob(name) if p.is_file() and p.stat().st_size==size]
        if not found:raise FileNotFoundError(name)
        return sorted(found)[0]
    SUP=find_zip('supervised_fault_classification.zip',874753167,SUP_ZIP_PATH)
    UNSUP=find_zip('unsupervised_anomaly_detection.zip',1170967719,UNSUP_ZIP_PATH)
    from huggingface_hub import snapshot_download
    tokenizer_snapshot=snapshot_download(experiment.crop.old.prior.MODEL_ID,revision=experiment.gate.REVISION,
        allow_patterns=list(experiment.gate.FILES))
    try:
        _,receipt=experiment.gate.verify(tokenizer_snapshot,experiment.prompt())
        PREFLIGHT.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
        GATE_PASSED=True
    except Exception as exc:
        GATE_PASSED=False;RETURN_CODE=98
        PREFLIGHT.write_text(json.dumps({'complete':False,'error':str(exc),'neural_calls':0}),encoding='utf-8')
        print('토크나이저 검사 실패: GPU 모델을 로드하지 않습니다.',type(exc).__name__,str(exc))
    if GATE_PASSED:
        print('TOKENIZER GATE PASS: 공식 원본 토큰화 일치. 원래 경고는 기록하며 regex 수정은 적용하지 않습니다.')
        snapshot=snapshot_download(experiment.crop.old.prior.MODEL_ID,revision=experiment.gate.REVISION)
        command=[sys.executable,str(CODE/'scripts/run_missing_presence_independent_dev.py'),
            '--dataset',str(DATASET),'--sup-zip',str(SUP),'--unsup-zip',str(UNSUP),
            '--model-path',snapshot,'--out-dir',str(OUT)]
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        with LOG.open('a',encoding='utf-8') as log:
            for line in process.stdout:print(line,end='');log.write(line);log.flush()
        RETURN_CODE=process.wait()
    print('RUNNER_RETURN_CODE',RETURN_CODE)
    ''')
    cell('code','''
    from google.colab import files
    RESULT=DRIVE/'missing_presence_independent_dev20_results.zip'
    with zipfile.ZipFile(RESULT,'w',zipfile.ZIP_DEFLATED) as archive:
        if OUT.exists():
            for p in sorted(OUT.rglob('*')):
                if p.is_file() and '_images' not in p.relative_to(OUT).parts:archive.write(p,'run/'+p.relative_to(OUT).as_posix())
        archive.writestr('source_bundle.zip',SOURCE_BYTES)
        archive.writestr('source_manifest.json',json.dumps(SOURCE_MANIFEST,indent=2))
        archive.writestr('runner_return_code.json',json.dumps({'return_code':RETURN_CODE}))
        if LOG.exists():archive.write(LOG,'console.log')
        if PREFLIGHT.exists():archive.write(PREFLIGHT,'tokenizer_preflight.json')
    status=json.loads((OUT/'run_status.json').read_text()) if (OUT/'run_status.json').exists() else {'complete':False}
    print('RUN STATUS',status)
    if RETURN_CODE or not status.get('complete'):print('실행/토크나이저 오류가 있습니다. ZIP으로 원인을 확인해야 합니다.')
    elif status.get('invalid_new_outputs'):print('무효 출력은 오답으로 포함했습니다.')
    print('진단 전용 / Agent 점수 아님. ZIP',RESULT,'SHA256',hashlib.sha256(RESULT.read_bytes()).hexdigest())
    files.download(str(RESULT))
    ''')
    nb=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'colab':{'provenance':[]}},nbformat=4,nbformat_minor=5)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    args.evidence_dir.mkdir(parents=True,exist_ok=True);(args.evidence_dir/'source_bundle.zip').write_bytes(source)
    receipt=dict(source_bundle_sha256=source_sha,notebook_sha256=run.crop.old.prior.probe.sha_file(args.output),
        source_files=manifest,compiled_code_cells=4,paired_images=20,max_lvlm_calls=60,neural_calls=0,
        source_result_sha256=SOURCE_RESULT_SHA,diagnostic_only=True)
    (args.evidence_dir/'notebook_build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_files'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['source-run','output','evidence-dir']:p.add_argument('--'+n,type=Path,required=True)
    build(p.parse_args())

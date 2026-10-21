"""Build a CPU-only manual box-label Colab with fixed dev identities and no model outputs."""
import argparse
import ast
import base64
import io
import json
from pathlib import Path
import textwrap
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED

import dev_missing_location_annotation as annotation

ROOT=Path(__file__).resolve().parents[1]
SOURCE_SHA='301011c1b9e7a4c7f36e1cd1385587b6d0dfa09ff4d4cdc514266cc807113a99'


def targets(source,archive):
    from PIL import Image
    if annotation.sha(Path(source).read_bytes())!=SOURCE_SHA:raise ValueError('use the audited PR64 input snapshot')
    data=annotation.read(source);expected={(r['role'],r['member']):r['sha256'] for r in data['image_manifest']}
    rows=[r for r in data['rows'] if r['defect_gt']=='missing-cap']
    if len(rows)!=25 or len({r['core_id'] for r in rows})!=25:raise ValueError('all25 dev missing-cap images required')
    result=[]
    with ZipFile(archive) as z:
        for row in rows:
            if row['protocol']!='selective_missing_expert_agent_dev_v1':raise ValueError('wrong source dev protocol')
            b=z.read(row['image']);h=annotation.sha(b)
            if h!=expected['target',row['image']]:raise ValueError('source image SHA mismatch')
            with Image.open(io.BytesIO(b)) as im:
                if im.getexif().get(274,1)!=1:raise ValueError('unsupported EXIF rotation')
                size=list(im.size)
            result.append({k:row[k] for k in ['core_id','image','asset_category','defect_gt','anomaly_gt']}|
                {'sha256':h,'size':size,'source_protocol':'gold_v2_dev'})
    result.sort(key=lambda r:annotation.sha(('annotation42/'+r['core_id']).encode()))
    manifest={'protocol':annotation.PROTOCOL,'source_snapshot_sha256':SOURCE_SHA,'cohort':'all25_missing_cap_from_fixed_dev240',
        'new_holdout':False,'targets':result}
    annotation.validate_manifest(manifest);return manifest


def build(args):
    manifest=targets(args.source_data,args.sup_zip)
    payloads={'scripts/dev_missing_location_annotation.py':(ROOT/'scripts/dev_missing_location_annotation.py').read_bytes(),
        'scripts/dev_missing_location_ui.html':(ROOT/'scripts/dev_missing_location_ui.html').read_bytes(),
        'inputs/manifest.json':(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()}
    hashes={n:annotation.sha(b) for n,b in payloads.items()};buf=io.BytesIO()
    with ZipFile(buf,'w',ZIP_DEFLATED) as z:
        for n,b in sorted(payloads.items()):
            info=ZipInfo(n,date_time=(2026,10,21,0,0,0));info.compress_type=ZIP_DEFLATED;z.writestr(info,b)
    payload=buf.getvalue();source_sha=annotation.sha(payload);cells=[]
    def cell(kind,text):
        c={'id':'dev-missing-label-'+str(len(cells)),'cell_type':kind,'metadata':{},'source':textwrap.dedent(text).strip().splitlines(keepends=True)}
        if kind=='code':ast.parse(''.join(c['source']));c.update(execution_count=None,outputs=[])
        cells.append(c)
    cell('markdown','''
    # dev 누락 위치 정답 표시 — 25장

    파일: `lvlm_dev_missing_location_annotation_colab.ipynb`

    **GPU 불필요(일반 CPU). 표시 화면까지 실행 → 박스 표시/확정 → 마지막 결과 저장 셀을 실행하세요.**
    이미 사용한 dev240 중 누락25장 전체입니다. 새 holdout이 아니며, 모델 답/가림 후보 위치는 화면에 보이지 않습니다.
    이전 gold119장 표시나 Q1/Q2/모델/기존 실험 결과를 바꾸지 않습니다. 추가 모델·토큰·재학습이 없습니다.

    - 유리 원판/부품이 빠진 연결부 또는 노출된 금속 몸통 등 **누락 위치를 보여주는 부분**을 박스로 감싸세요. 전체 절연체 줄을 크게 감싸지 마세요.
    - 여러 실제 누락이면 각각 박스를 그리세요. 애매한 대안 두 곳을 모두 정답으로 표시하지 말고, `위치 판단 불가` + 사유를 남기세요.
    - 누락 라벨 자체에 동의하기 어려우면 `누락 라벨 이견` + 사유. 두 경우 모두 위치 미확정이며 정상 정답으로 바꾸지 않습니다.
    - 박스 추가/삭제·메모는 Drive에 **초안 자동 저장**됩니다. `저장 후 다음(Enter)`으로 확정하세요. 재접속하면 저장한 초안을 복원합니다.
    - `초안 저장됨` 표시를 확인한 뒤 나가세요. 아직 그리는 중인 박스나 저장 응답을 못 받은 변경은 보장되지 않습니다.
    - 원본 크기/확대/이동 가능. 9칸 안내선은 선택이며 정답은 각 박스 중심으로 자동 변환됩니다. 서로 다른 누락의 중심을 평균내지 않습니다.

    완료 결과: `dev_missing_location_annotation25_results.zip`. 마지막 셀은 미완료 상태에서도 중간 백업을 만들 수 있습니다.
    ''')
    cell('code','''
    RUN_FOLDER = "lvlm_insplad_results/dev_missing_location_annotation25_v1"
    SUP_ZIP_PATH = ""  # 자동 검색이 안 될 때만 실제 Drive ZIP 경로를 입력
    ''')
    cell('code',f'''
    import base64,hashlib,io,json,sys,zipfile
    from pathlib import Path
    from google.colab import drive
    drive.mount('/content/drive',force_remount=False)
    DRIVE=Path('/content/drive/MyDrive')
    SOURCE_SHA256={source_sha!r}
    SOURCE_MANIFEST={hashes!r}
    SOURCE_BYTES=base64.b64decode({base64.b64encode(payload).decode()!r})
    assert hashlib.sha256(SOURCE_BYTES).hexdigest()==SOURCE_SHA256
    CODE=Path('/content/dev_missing_location_'+SOURCE_SHA256[:12])
    with zipfile.ZipFile(io.BytesIO(SOURCE_BYTES)) as archive:
        assert set(archive.namelist())==set(SOURCE_MANIFEST)
        for name in archive.namelist():
            dest=CODE/name;data=archive.read(name)
            assert dest.resolve().is_relative_to(CODE.resolve())
            assert hashlib.sha256(data).hexdigest()==SOURCE_MANIFEST[name]
            if dest.exists() and dest.read_bytes()!=data:raise RuntimeError('내장 코드와 다른 파일입니다. 새 런타임을 사용하세요.')
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    sys.path.insert(0,str(CODE/'scripts'))
    import dev_missing_location_annotation as annotation
    def find_zip():
        if SUP_ZIP_PATH:
            p=Path(SUP_ZIP_PATH)
            if not p.is_file():raise FileNotFoundError(str(p))
            return p
        for p in [DRIVE/'supervised_fault_classification.zip',DRIVE/'lvlm_insplad_data/supervised_fault_classification.zip']:
            if p.is_file() and p.stat().st_size==874753167:return p
        matches=[p for p in DRIVE.rglob('supervised_fault_classification.zip') if p.is_file() and p.stat().st_size==874753167]
        if not matches:raise FileNotFoundError('supervised_fault_classification.zip 원본을 Drive에서 찾지 못했습니다.')
        return sorted(matches)[0]
    OUT=DRIVE/RUN_FOLDER
    assert OUT.name.startswith(annotation.PREFIX)
    manifest=annotation.read(CODE/'inputs/manifest.json')
    session=annotation.prepare(manifest,find_zip(),OUT)
    (OUT/'source_bundle.zip').write_bytes(SOURCE_BYTES)
    annotation.atomic(OUT/'source_manifest.json',SOURCE_MANIFEST)
    print('원본 SHA 확인 완료. 저장/재개 상태:',session.progress())
    print('저장 위치:',OUT/'annotations.json')
    ''')
    cell('code','''
    # 이 화면에서 박스를 표시합니다. 박스 추가/삭제는 초안 저장, Enter는 확정 후 다음.
    annotation.register_colab(session)
    ''')
    cell('markdown','''
    ## 표시를 마친 뒤 아래 셀 실행

    먼저 화면에서 `저장 후 다음`으로 확정하세요. 마지막 상태가 `25/25`인지 확인합니다.
    아래 셀은 Drive의 **서버 저장 상태**로 결과를 만듭니다. 브라우저에서 아직 저장 중인 변경은 기다린 뒤 실행하세요.
    중간 백업도 가능하며, 미완료/위치 불가/라벨 이견은 위치 정답으로 취급하지 않습니다.
    ''')
    cell('code','''
    from google.colab import files
    result,summary=session.export()
    print('완료/미완료 및 위치 정답 지원 수:',summary)
    if not summary['complete']:print('미확정 사진이 있습니다. 중간 백업이며 표시를 끝낸 뒤 이 셀을 다시 실행하세요.')
    print('ZIP:',result)
    files.download(str(result))
    ''')
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'colab':{'provenance':[]}},'nbformat':4,'nbformat_minor':5}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    args.evidence_dir.mkdir(parents=True,exist_ok=True);(args.evidence_dir/'source_bundle.zip').write_bytes(payload)
    receipt={'source_bundle_sha256':source_sha,'source_files':hashes,'notebook_working_sha256':annotation.sha(args.output.read_bytes()),
        'images':25,'code_cells_compiled':4,'neural_calls':0,'annotation_completed':False,'new_holdout':False}
    annotation.atomic(args.evidence_dir/'notebook_build.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source-data','sup-zip','output','evidence-dir']:p.add_argument('--'+key,type=Path,required=True)
    build(p.parse_args())

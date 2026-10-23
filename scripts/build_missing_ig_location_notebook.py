"""Build a self-contained, frozen-input IG localization dev Colab from audited PR68."""
import argparse
import ast
import base64
import io
import json
from pathlib import Path
import textwrap
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED
import run_missing_ig_location_dev as exp

SOURCE_SHA='bad35ced5b0b7322630c4630ea28a1638c520d32d79b635520006fffe80e1a38'
PR68_CAPSULE='2ddb37fc4235fd00c881a3d6714a1378b78717983979a64f7bd1b2fd15cdf852'


def package(args):
    if exp.old.probe.sha_file(args.source_run)!=SOURCE_SHA:raise ValueError('use audited actual PR68 ZIP')
    with ZipFile(args.source_run) as z:
        read=lambda name:json.loads(z.read(name))
        meta=read('run/run_metadata.json');status=read('run/run_status.json')
        if meta['fixture'] or meta['backend']!='hf' or not status['complete'] or status['feature_errors'] or not status['canary_passed']:
            raise ValueError('successful native source required')
        capsule=z.read('source_bundle.zip')
        if exp.sha(capsule)!=PR68_CAPSULE:raise ValueError('source capsule changed')
        data=read('run/dataset.json');head_bytes=z.read('run/frozen_detector.json');head=json.loads(head_bytes)
        expected=exp.old.oc.validate(data,head,exp.sha(head_bytes))
        tools=read('run/tools.json');maps=read('run/occlusion_maps.json');features=read('run/features.json')
        rows=[r for r in data['rows'] if r['asset_category']==exp.old.probe.ASSET]
        if len(rows)!=50 or len(data['rows'])!=240 or len(features)!=518:raise ValueError('source cohort changed')
        paths=exp.old.prior.old.verified_images(args.sup_zip,[r['image'] for r in rows],args.evidence_dir/'_images')
        for row in rows:
            if exp.old.probe.sha_file(paths[row['image']])!=expected['target',row['image']]:raise ValueError('source original changed')
        vectors={}
        for key,rec in features.items():
            b=z.read('run/feature_cache/'+exp.digest(key)+'.npy')
            if rec['error'] or exp.sha(b)!=rec['feature_sha256']:raise ValueError('source feature failed/corrupted')
            x=exp.old.check_vector(b,len(head['coef']));vals=exp.old.score_feature(head,x)
            if any(not exp.np.isclose(vals[k],rec[k],atol=1e-12,rtol=1e-12) for k in ('score','logit')):raise ValueError('source feature score changed')
            vectors[key]=b
        baseline=read('run/predictions.json')['all_positive_top1']
        if sum(t['positive'] for t in tools.values())!=26 or set(maps)!={c for c,t in tools.items() if t['positive']}:raise ValueError('source routing changed')
        for row in rows:
            core=row['core_id'];r=features[core+'/original/0'];t=tools[core]
            if t['score']!=r['score'] or t['positive']!=(r['score']>=head['threshold']):raise ValueError('source expert changed')
            if t['positive'] and baseline[core]['present'] is not True:raise ValueError('positive source presence missing')
        payloads={'inputs/dataset.json':z.read('run/dataset.json'),'inputs/head.json':head_bytes,
            'inputs/baseline.json':(json.dumps(baseline,indent=2)+'\n').encode(),
            'inputs/tools.json':z.read('run/tools.json'),'inputs/old_maps.json':z.read('run/occlusion_maps.json'),
            'inputs/evaluation_gt.json':z.read('run/evaluation_gt.json')}
        with ZipFile(io.BytesIO(capsule)) as inner:
            payloads['inputs/annotations.json']=inner.read('inputs/annotations.json')
            if payloads['inputs/evaluation_gt.json']!=inner.read('inputs/evaluation_gt.json'):raise ValueError('source GT changed')
        if exp.sha(payloads['inputs/annotations.json'])!='9a9e71b995174288e3507e0d60f97fd155abb2149de2bf4ee0feb10fe02d52d3':raise ValueError('wrong manual GT')
        selected=[k for k,r in features.items() if r['kind']=='original']
        selected.append(next(k for k,r in sorted(features.items()) if r['kind']=='blur'))
        seeds={}
        for key in selected:
            rec=features[key];core=rec['core_id'];row=next(r for r in rows if r['core_id']==core)
            im=exp.old.oc.original_image(paths[row['image']])
            if rec['kind']!='original':im=exp.old.oc.masked(im,rec['cell'],rec['kind'])
            if exp.sha(im.tobytes())!=rec['pixel_sha256'] or list(im.size)!=rec['size']:raise ValueError('source feature image/mask changed')
            path='inputs/features/'+exp.digest(key)+'.npy';payloads[path]=vectors[key]
            seeds[key]=dict(id=key,core_id=core,kind=rec['kind'],cell=rec['cell'],feature_path=path,
                feature_sha256=rec['feature_sha256'],image_sha256=expected['target',row['image']],pixel_sha256=rec['pixel_sha256'],
                source_archive_sha256=SOURCE_SHA)
        if len(seeds)!=51:raise ValueError('wrong canary/original seeds')
        payloads['inputs/seeds.json']=(json.dumps(seeds,indent=2)+'\n').encode()
        payloads['inputs/provenance.json']=(json.dumps(dict(source_archive_sha256=SOURCE_SHA,source_capsule_sha256=PR68_CAPSULE,
            source_software=meta['software'],source_identity=meta['identity'],source_head_sha256=exp.sha(head_bytes),
            method='input IG of frozen logit; custom component localization; no efficacy claim',
            references=['https://proceedings.mlr.press/v70/sundararajan17a','https://captum.ai/api/integrated_gradients.html']),indent=2)+'\n').encode()
    payloads.update({n:(exp.ROOT/n).read_bytes() for n in exp.runtime_hashes()})
    for n in ('scripts/build_missing_ig_location_notebook.py','tests/test_missing_ig_location_dev.py'):
        payloads[n]=(exp.ROOT/n).read_bytes()
    return payloads


def build(args):
    args.evidence_dir.mkdir(parents=True,exist_ok=True)
    payloads=package(args);manifest={n:exp.sha(b) for n,b in payloads.items()};buf=io.BytesIO()
    with ZipFile(buf,'w',ZIP_DEFLATED) as z:
        for n,b in sorted(payloads.items()):
            info=ZipInfo(n,date_time=(2026,10,23,0,0,0));info.compress_type=ZIP_DEFLATED;z.writestr(info,b)
    payload=buf.getvalue();source_sha=exp.sha(payload);cells=[]
    def cell(kind,text):
        c=dict(id='missing-ig-'+str(len(cells)),cell_type=kind,metadata={},source=textwrap.dedent(text).strip().splitlines(keepends=True))
        if kind=='code':ast.parse(''.join(c['source']));c.update(execution_count=None,outputs=[])
        cells.append(c)
    cell('markdown','''
    # 다음 위치 도구: IG 기여도 → 여러 부위 후보 → 실제 9칸 평가

    **파일: `lvlm_missing_ig_location_dev_colab.ipynb`**

    **L4/A100 → 기본값 유지 → 모두 실행 → Drive 연결 허용. 기능 제작 완료, 실제 위치 개선 효과는 실행 전입니다.**
    이전 코랩 재실행/추가 표시/토큰 입력은 불필요합니다. supervised 원본 ZIP 한 개를 자동 검색합니다.

    - 이전 실제 PR68의 누락 유무와 모든240답을 고정하고, 판별기 양성26장에만 새 위치 도구를 적용합니다. 누락25/25 검출은 고정 입력이며 이번 도구의 새 개선으로 주장하지 않습니다.
    - 같은 동결 DINO+분류기의 점수에 기여한 픽셀을 Integrated Gradients로 계산합니다. 흐린 사진/전체 평균색 두 기준을 쓰며, 각32단계입니다. **역전파는 해석 계산이며 가중치 학습/변경은 없습니다.**
    - 기존의 큰칸 가림 top1과 IG top1/여러부위를 비교합니다. IG 여러부위와 같은 수의 기존가림순위/중앙/해시무작위 칸 대조도 비교합니다. 단순히 칸만 늘린 효과를 구분합니다.
    - 원본 좌표로 복원하며 정답 위치24장/판단불가1장은 채점에만 씁니다. 위치를 보류해도 누락 유무는 바꾸지 않습니다.
    - 기존 특징2개를 재계산해 동등성 검사 후26장 각각의 원본특징도 비교합니다. 최대 약1744 forward/1664 backward, DINO만 사용합니다. 새 LVLM/재학습/Q1Q2gold0.
    - IG는 판별기 설명이지 결함 위치 정답이나 분할 모델이 아닙니다. 배경에 반응할 수 있고 두 기준에 민감할 수 있습니다. completeness 수치오차와 원시 기여도도 저장합니다. 효과는 실행 후 확인합니다.
    - 기존 dev를 반복 탐색한 실험입니다. 최종 Agent/새 holdout 성능이 아닙니다. 정상 유무 오탐은 고정되어 있으므로 개선됐다고 해석하지 않습니다.
    - 이미지별로 별도 Drive 폴더에 저장합니다. 같은 입력/코드/환경은 성공 이미지부터 재개하고 실패만 재시도합니다. 결과 ZIP은 기여도 맵을 포함해 수십 MB일 수 있습니다.
    ''')
    cell('code','''
    RUN_FOLDER = "lvlm_insplad_results/missing_ig_location_dev240_v1"
    SUP_ZIP_PATH = ""
    ''')
    cell('code','''
    import subprocess,sys
    subprocess.run([sys.executable,'-m','pip','install','-q','numpy==2.1.3','pillow==11.3.0',
        'transformers==5.14.0','accelerate==1.14.0','timm==1.0.30'],check=True)
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<20*1024**3:
        raise RuntimeError('L4/A100 VRAM20GB 이상을 선택하세요.')
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
    CODE=Path('/content/missing_ig_location_'+SOURCE_SHA256[:12])
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
    RETURN_CODE=98;LOG=DRIVE/'missing_ig_location_dev240_console.log'
    print('SOURCE',SOURCE_SHA256,'26장 위치 도구 실험. 유무 고정/새LVLM0/학습0/IG역전파해석만 수행')
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
    command=[sys.executable,str(CODE/'scripts/run_missing_ig_location_dev.py'),
        '--capsule-dir',str(CODE),'--sup-zip',str(SUP),'--out-dir',str(OUT)]
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    with LOG.open('a',encoding='utf-8') as log:
        for line in process.stdout:print(line,end='');log.write(line);log.flush()
    RETURN_CODE=process.wait();print('RUNNER_RETURN_CODE',RETURN_CODE)
    ''')
    cell('code','''
    from google.colab import files
    RESULT=DRIVE/'missing_ig_location_dev240_results.zip'
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
    if RETURN_CODE or not status.get('complete'):print('실패/미완료입니다. ZIP을 보존해 전달하세요.')
    if status.get('completeness_warnings'):print('일부 IG 적분 오차가 큽니다. 원시 맵·진단과 함께 분석해야 합니다.')
    if not RETURN_CODE and status.get('complete') and (OUT/'leaderboard.csv').exists():print((OUT/'leaderboard.csv').read_text())
    print('ZIP',RESULT,'SHA256',hashlib.sha256(RESULT.read_bytes()).hexdigest())
    files.download(str(RESULT))
    ''')
    nb=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'colab':{'provenance':[]}},nbformat=4,nbformat_minor=5)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    (args.evidence_dir/'source_bundle.zip').write_bytes(payload)
    receipt=dict(source_bundle_sha256=source_sha,notebook_working_sha256=exp.old.probe.sha_file(args.output),source_files=manifest,
        code_cells=5,images=240,routed_images=26,original_features=50,canary_seed_features=1,
        max_forward_passes=1744,max_backward_passes=1664,actual_pretrained_neural_calls=0,effect='UNRUN')
    exp.write(args.evidence_dir/'notebook_build.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_files'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','sup-zip','output','evidence-dir'):p.add_argument('--'+name,type=Path,required=True)
    build(p.parse_args())

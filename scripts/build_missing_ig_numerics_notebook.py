"""Seal audited PR69 numerical inputs into a Run-all Colab; no location GT."""
import argparse,ast,base64,io,json,textwrap
from pathlib import Path
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED
import run_missing_ig_numerics as exp

SOURCE_SHA='d9170b124d97f6c7cf2bc5d626bf397cb4167b69d9a296d3e63c4ca7d8248d7d'
PR69_SOURCE='66e7e6f5209af548428653e5c0c1a70c1eb4aa4a883f3e07704265bca1d34834'

def package(args):
    if exp.old.old.probe.sha_file(args.source_run)!=SOURCE_SHA:raise ValueError('audited native PR69 ZIP required')
    with ZipFile(args.source_run) as z:
        read=lambda n:json.loads(z.read(n))
        meta=read('run/run_metadata.json');status=read('run/run_status.json');manifest=read('source_manifest.json')
        if meta['fixture'] or meta['backend']!='hf' or not status['complete'] or status['attribution_errors'] or not status['canary_passed']:raise ValueError('successful native source required')
        source=z.read('source_bundle.zip')
        if exp.sha(source)!=PR69_SOURCE:raise ValueError('wrong PR69 source')
        with ZipFile(io.BytesIO(source)) as prior:
            if set(prior.namelist())!=set(manifest):raise ValueError('source manifest coverage changed')
            for n,h in manifest.items():
                if exp.sha(prior.read(n))!=h:raise ValueError('source file changed')
            tools=read('run/tools.json');records=read('run/attributions.json');data=read('run/dataset.json');seeds=read('run/seeds.json')
            head=z.read('run/head.json')
            if exp.sha(head)!='d9ff0610d0b74752a2dfc5b53f1cdc1db5e916e17b4260172f1c7abab8c6e7f1':raise ValueError('frozen head changed')
            positives=sorted(c for c,t in tools.items() if t['positive'])
            if len(positives)!=26 or set(positives)!=set(records):raise ValueError('positive routes changed')
            selected=exp.num.choose(positives);rows={r['core_id']:r for r in data['rows']}
            seed_ids={c+'/original/0' for c in selected}
            seed_ids.update(next(k for k,s in sorted(seeds.items()) if s['kind']==kind) for kind in ('original','blur'))
            chosen={k:dict(seeds[k],image=rows[seeds[k]['core_id']]['image']) for k in sorted(seed_ids)}
            paths=exp.old.old.prior.old.verified_images(args.sup_zip,[s['image'] for s in chosen.values()],args.evidence_dir/'_images')
            payloads={'inputs/head.json':head};requests=[];old32={}
            for s in chosen.values():
                b=prior.read(s['feature_path'])
                if exp.sha(b)!=s['feature_sha256']:raise ValueError('source feature SHA changed')
                exp.old.old.check_vector(b,768);payloads[s['feature_path']]=b
                if exp.old.old.probe.sha_file(paths[s['image']])!=s['image_sha256']:raise ValueError('source image changed')
                im=exp.old.old.oc.original_image(paths[s['image']])
                if s['kind']!='original':im=exp.old.old.oc.masked(im,s['cell'],s['kind'])
                if exp.sha(im.tobytes())!=s['pixel_sha256']:raise ValueError('source RGB changed')
            for c in selected:
                s=chosen[c+'/original/0'];rec=records[c]
                if rec['error']:raise ValueError('source attribution failed')
                requests.append(dict(core_id=c,image=s['image'],image_sha256=s['image_sha256']))
                old32[c]={}
                for kind,m in rec['raw_maps'].items():
                    b=z.read('run/'+m['path'])
                    if exp.sha(b)!=m['sha256']:raise ValueError('source signed map changed')
                    a=exp.np.load(io.BytesIO(b),allow_pickle=False)
                    patch=exp.old.ig.patch_map(a,rec['geometry'])[0];buf=io.BytesIO();exp.np.save(buf,patch,allow_pickle=False)
                    name='inputs/old_patches/'+c+'_'+kind+'.npy';payloads[name]=buf.getvalue()
                    old32[c][kind]=dict(patch_path=name,signed_map_sha256=m['sha256'],diagnostics=rec['diagnostics'][kind],
                        target_logit=rec['target_logit'],original_input_sha256=rec['original_input_sha256'],
                        baseline_input_sha256=rec['diagnostics'][kind]['baseline_input_sha256'])
            values={'requests':requests,'selection':dict(positive_cores=positives,chosen_cores=selected,rule=exp.num.CONFIG['selection']),
                'seeds':chosen,'previous32':old32,'provenance':dict(source_run_sha256=SOURCE_SHA,source_bundle_sha256=PR69_SOURCE,
                    source_identity=meta['identity'],source_software=meta['software'],no_location_gt=True)}
            for n,v in values.items():payloads['inputs/'+n+'.json']=(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode()
    payloads.update({n:(exp.ROOT/n).read_bytes() for n in exp.old.runtime_hashes()})
    for n in ('scripts/build_missing_ig_numerics_notebook.py','tests/test_missing_ig_numerics.py'):
        payloads[n]=(exp.ROOT/n).read_bytes()
    return payloads,selected

def build(args):
    args.evidence_dir.mkdir(parents=True,exist_ok=True)
    payloads,selected=package(args);manifest={n:exp.sha(b) for n,b in payloads.items()};buf=io.BytesIO()
    with ZipFile(buf,'w',ZIP_DEFLATED) as z:
        for n,b in sorted(payloads.items()):
            info=ZipInfo(n,date_time=(2026,10,23,0,0,0));info.compress_type=ZIP_DEFLATED;z.writestr(info,b)
    payload=buf.getvalue();source_sha=exp.sha(payload);cells=[]
    def cell(kind,text):
        c=dict(id='ig-numerics-'+str(len(cells)),cell_type=kind,metadata={},source=textwrap.dedent(text).strip().splitlines(keepends=True))
        if kind=='code':ast.parse(''.join(c['source']));c.update(execution_count=None,outputs=[])
        cells.append(c)
    cell('markdown','''
    # IG 계산 오차 진단 — 4장, 가중치 고정

    **`lvlm_missing_ig_numerics_colab.ipynb` — L4/A100 → 기본값 → 모두 실행 → Drive 연결 허용.**
    이전 코랩 재실행/추가 표시/토큰 입력은 불필요합니다. 기존 supervised 원본 ZIP 1개를 자동 검색합니다.

    이전 실험은 실행 오류0이지만 IG 적분 경고가49/52개였습니다. 이번은 **기능 제작 완료, 실제 원인 진단은 아직 실행 전**입니다.
    - 정답 위치나 오차 크기를 보고 고르지 않은 고정4장, 흐림/평균색 두 기준을 사용합니다.
    - Midpoint32/128/512와 Gauss-Legendre128 적분, 동일 기울기의 FP32/FP64 누적을 비교합니다.
    - 경로의3점에서 autograd와 중앙차분(크기3종)을 비교하고 기본 attention/SDPA MATH 계산을 기록합니다.
    - FP64는 누적 정밀도만 높입니다. 네트워크는 기존 FP32 그대로이고 학습하지 않습니다.
    - 실행 완료와 수치 정확성은 별개입니다. 경고/오류/원시 맵을 모두 보존합니다. 위치 정확도나 최종 Agent 성능을 평가하는 코랩은 아닙니다.

    최대6,810forward/6,448backward, batch1. 중단 후 같은 설정으로 모두 실행하면 완료된 조건은 재사용합니다.
    결과: **`missing_ig_numerics_results.zip`**. 경고를 없애려고 설정을 바꾸지 말고 그대로 전달하세요.
    ''')
    cell('code','''
    import subprocess,sys
    subprocess.check_call([sys.executable,'-m','pip','install','-q','torch==2.11.0',
        'transformers==5.14.0','numpy==2.1.3','Pillow==11.3.0','timm==1.0.30'])
    ''')
    cell('code','''
    from google.colab import drive
    drive.mount('/content/drive')
    ''')
    cell('code',f'''
    import base64,hashlib,json,zipfile,io
    from pathlib import Path
    SUP_ZIP_PATH='' # 자동 검색, 필요할 때만 원본 ZIP 경로 입력
    RUN_FOLDER='missing_ig_numerics_v1'
    DRIVE=Path('/content/drive/MyDrive')
    SOURCE_SHA256={source_sha!r}
    SOURCE_MANIFEST={manifest!r}
    SOURCE_BYTES=base64.b64decode({base64.b64encode(payload).decode()!r})
    assert hashlib.sha256(SOURCE_BYTES).hexdigest()==SOURCE_SHA256
    CODE=Path('/content/ig_numerics_'+SOURCE_SHA256[:12])
    with zipfile.ZipFile(io.BytesIO(SOURCE_BYTES)) as archive:
        assert set(archive.namelist())==set(SOURCE_MANIFEST)
        for name in archive.namelist():
            content=archive.read(name);assert hashlib.sha256(content).hexdigest()==SOURCE_MANIFEST[name]
            path=(CODE/name).resolve();assert path.is_relative_to(CODE.resolve())
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():assert path.read_bytes()==content
            else:path.write_bytes(content)
    (CODE/'source_manifest.json').write_text(json.dumps(SOURCE_MANIFEST,indent=2))
    OUT=DRIVE/'lvlm_insplad_results'/RUN_FOLDER
    if not OUT.name.startswith('missing_ig_numerics'):raise ValueError('별도 실행 폴더 이름을 유지하세요.')
    RETURN_CODE=98;LOG=DRIVE/'missing_ig_numerics_console.log'
    print('SOURCE',SOURCE_SHA256,'고정4장 수치진단. 가중치고정/학습0/LVLM0/gold0')
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
    command=[sys.executable,str(CODE/'scripts/run_missing_ig_numerics.py'),
        '--capsule-dir',str(CODE),'--sup-zip',str(SUP),'--out-dir',str(OUT)]
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    with LOG.open('a',encoding='utf-8') as log:
        for line in process.stdout:print(line,end='');log.write(line);log.flush()
    RETURN_CODE=process.wait();print('RUNNER_RETURN_CODE',RETURN_CODE)
    ''')
    cell('code','''
    from google.colab import files
    RESULT=DRIVE/'missing_ig_numerics_results.zip'
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
    print('실행완료는수치통과가아닙니다. 경고와원시결과를함께분석합니다.')
    if (OUT/'integration_summary.csv').exists():print((OUT/'integration_summary.csv').read_text())
    print('ZIP',RESULT,'SHA256',hashlib.sha256(RESULT.read_bytes()).hexdigest())
    files.download(str(RESULT))
    ''')
    nb=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'colab':{'provenance':[]}},nbformat=4,nbformat_minor=5)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    (args.evidence_dir/'source_bundle.zip').write_bytes(payload)
    receipt=dict(source_bundle_sha256=source_sha,notebook_sha256=exp.old.old.probe.sha_file(args.output),source_files=manifest,
        selected_cores=selected,code_cells=5,max_forward_passes=6810,max_backward_passes=6448,actual_pretrained_neural_calls=0,native_result='UNRUN')
    exp.write(args.evidence_dir/'notebook_build.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_files'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','sup-zip','output','evidence-dir'):p.add_argument('--'+name,type=Path,required=True)
    build(p.parse_args())

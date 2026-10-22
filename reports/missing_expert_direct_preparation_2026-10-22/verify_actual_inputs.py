"""CPU wiring checks on actual images/cached vectors; unknown features are explicitly fixtures."""
import argparse
import ast
import base64
import io
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import run_missing_expert_direct_dev as exp


def main():
    p=argparse.ArgumentParser();p.add_argument('--sup-zip',type=Path,required=True)
    p.add_argument('--old-run',type=Path,required=True);args=p.parse_args()
    here=Path(__file__).resolve().parent
    receipt=exp.read(here/'notebook_build.json'); notebook=exp.read(ROOT/'notebooks/lvlm_missing_expert_direct_dev_colab.ipynb')
    code=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
    assert len(code)==5
    for c in code:ast.parse(c)
    extract_code=next(c for c in code if 'SOURCE_BYTES=base64.b64decode' in c)
    constants={}
    for statement in ast.parse(extract_code).body:
        if isinstance(statement,ast.Assign) and isinstance(statement.targets[0],ast.Name):
            name=statement.targets[0].id
            if name in ('SOURCE_SHA256','SOURCE_MANIFEST'):constants[name]=ast.literal_eval(statement.value)
            if name=='SOURCE_BYTES':constants[name]=base64.b64decode(ast.literal_eval(statement.value.args[0]))
    assert exp.sha(constants['SOURCE_BYTES'])==receipt['source_bundle_sha256']==constants['SOURCE_SHA256']
    root=here/('extracted_'+receipt['source_bundle_sha256'][:12])
    with ZipFile(io.BytesIO(constants['SOURCE_BYTES'])) as z:
        assert set(z.namelist())==set(constants['SOURCE_MANIFEST'])
        for name in z.namelist():
            content=z.read(name);assert exp.sha(content)==constants['SOURCE_MANIFEST'][name]
            target=exp.checked_path(root,name);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    exp.write(root/'source_manifest.json',constants['SOURCE_MANIFEST'])
    seeds=exp.read(root/'inputs/feature_seeds.json');vectors={}
    for seed in seeds.values():
        vectors[seed['pixel_sha256']]=exp.check_vector((root/seed['feature_path']).read_bytes(),768)
    # GT-blind deterministic stand-in only for unseen original/mask requests.
    # This checks plumbing, NOT real DINO outputs or effectiveness of new routes.
    fallback=exp.check_vector((root/next(iter(seeds.values()))['feature_path']).read_bytes(),768)
    counts={'seed_callbacks':0,'fixture_callbacks':0}
    def fixture(im):
        key=exp.pixel_sha(im)
        counts['seed_callbacks' if key in vectors else 'fixture_callbacks']+=1
        return vectors[key] if key in vectors else fallback
    out=here/(exp.PREFIX+'_actual_input_fixture_'+receipt['source_bundle_sha256'][:12])
    command=['--capsule-dir',str(root),'--sup-zip',str(args.sup_zip),'--out-dir',str(out),'--n-boot','30']
    assert exp.main(command,extractor=fixture)==0
    status=exp.read(out/'run_status.json');metadata=exp.read(out/'run_metadata.json')
    assert status['fixture'] and metadata['fixture'] and metadata['lvlm_calls']==0
    assert status['images']==240 and status['expert_images']==50
    maps=exp.read(out/'occlusion_maps.json');old=exp.read(args.old_run/'occlusion_maps.json')
    max_score_error=0.
    for core,record in old.items():
        new=maps[core]
        for key in ('regions','abstained','is_verified_location'):assert new['proposal'][key]==record['proposal'][key]
        assert exp.np.isclose(new['original_logit'],record['original_logit'],atol=1e-12,rtol=1e-12)
        for key in ('agreement_rank','logit_drops'):
            assert exp.np.allclose(new['proposal'][key],record['proposal'][key],atol=1e-12,rtol=1e-12)
        for kind in exp.oc.CONFIG['replacements']:
            delta=float(exp.np.max(exp.np.abs(exp.np.array(new['occluded_logits'][kind])-record['occluded_logits'][kind])))
            assert delta<1e-12;max_score_error=max(max_score_error,delta)
    assert len(old)==23
    resume_before=dict(counts)
    assert exp.main(command,extractor=lambda im:(_ for _ in ()).throw(AssertionError('resume must not load extractor')))==0
    assert counts==resume_before
    resumed=exp.read(out/'run_metadata.json');assert resumed['new_features']==resumed['new_canary_features']==0
    # Execute the exact export cell against a mock Colab download bridge.
    import types
    google=types.ModuleType('google');colab=types.ModuleType('google.colab');downloads=[]
    colab.files=types.SimpleNamespace(download=lambda path:downloads.append(path));google.colab=colab
    old_modules={k:sys.modules.get(k) for k in ('google','google.colab')}
    sys.modules['google']=google;sys.modules['google.colab']=colab
    try:
        import zipfile,hashlib
        ns=dict(DRIVE=here,OUT=out,SOURCE_BYTES=constants['SOURCE_BYTES'],SOURCE_MANIFEST=constants['SOURCE_MANIFEST'],
                LOG=here/'fixture_console.log',RETURN_CODE=0,zipfile=zipfile,json=json,hashlib=hashlib)
        exec(compile(code[-1],'<exact-export-cell>','exec'),ns)
    finally:
        for k,v in old_modules.items():
            if v is None:sys.modules.pop(k,None)
            else:sys.modules[k]=v
    assert len(downloads)==1
    with ZipFile(downloads[0]) as z:
        names=set(z.namelist());assert not any('/_images/' in n for n in names)
        assert {'run/features.json','run/canary.json','run/predictions.json','run/run_status.json','source_bundle.zip','source_manifest.json'}<=names
        assert json.loads(z.read('run/run_status.json'))['fixture'] is True
    # Run source-extracted unit tests in a clean subprocess, never local-module fallbacks.
    result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_missing_expert_direct_dev.py',
        '--basetemp=.pytest-extracted'],cwd=root,text=True,capture_output=True)
    (here/'extracted_tests.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
    assert result.returncode==0,result.stdout+result.stderr
    verification=dict(source_bundle_sha256=receipt['source_bundle_sha256'],source_files=len(constants['SOURCE_MANIFEST']),
        real_originals_verified=50,source_original_vectors=49,source_mask_vectors=414,source_candidate_sets_exact=23,
        source_logits_max_abs_error=max_score_error,source_logits_atol=1e-12,
        code_cells_compiled=5,raw_fixture_features=metadata['features'],fixture_new_feature_calls=metadata['new_features'],
        fixture_canary_calls=metadata['new_canary_features'],actual_dino_calls=0,actual_lvlm_calls=0,
        resume_new_calls=0,export_originals_excluded=True,extracted_tests=result.stdout.strip(),
        effect='UNRUN; fixture answers do not establish new route performance',callbacks=counts)
    exp.write(here/'verification.json',verification);print(json.dumps(verification,indent=2))


if __name__=='__main__':main()

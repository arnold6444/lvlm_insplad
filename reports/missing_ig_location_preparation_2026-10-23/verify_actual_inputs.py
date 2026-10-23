"""Verify exact capsule/real-image wiring with explicitly synthetic attribution maps."""
import argparse
import ast
import base64
import io
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType,SimpleNamespace
from zipfile import ZipFile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import run_missing_ig_location_dev as exp


def main():
    p=argparse.ArgumentParser();p.add_argument('--sup-zip',type=Path,required=True);args=p.parse_args()
    here=Path(__file__).resolve().parent;receipt=exp.read(here/'notebook_build.json')
    nb=exp.read(ROOT/'notebooks/lvlm_missing_ig_location_dev_colab.ipynb')
    code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'];assert len(code)==5
    for c in code:ast.parse(c)
    constants={}
    for node in ast.parse(code[2]).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ('SOURCE_SHA256','SOURCE_MANIFEST'):constants[name]=ast.literal_eval(node.value)
            if name=='SOURCE_BYTES':constants[name]=base64.b64decode(ast.literal_eval(node.value.args[0]))
    assert exp.sha(constants['SOURCE_BYTES'])==receipt['source_bundle_sha256']==constants['SOURCE_SHA256']
    root=here/('extracted_'+constants['SOURCE_SHA256'][:12])
    with ZipFile(io.BytesIO(constants['SOURCE_BYTES'])) as z:
        assert set(z.namelist())==set(constants['SOURCE_MANIFEST'])
        for name in z.namelist():
            b=z.read(name);assert exp.sha(b)==constants['SOURCE_MANIFEST'][name]
            p=exp.old.checked_path(root,name);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    exp.write(root/'source_manifest.json',constants['SOURCE_MANIFEST'])
    seeds=exp.read(root/'inputs/seeds.json');head=exp.read(root/'inputs/head.json')
    vectors={s['pixel_sha256']:exp.old.check_vector((root/s['feature_path']).read_bytes(),768) for s in seeds.values()}
    class Fixture:
        def __init__(self,fail=False):self.stats=dict(forward_passes=0,backward_passes=0,model_loads=0);self.calls=0;self.fail=fail
        def feature(self,im):
            if self.fail:raise RuntimeError('intentional canary failure fixture')
            self.stats['forward_passes']+=1
            return vectors[exp.sha(im.tobytes())]
        def attribute(self,im,expected):
            self.calls+=1;self.stats['forward_passes']+=67;self.stats['backward_passes']+=64
            raw={};diag={};target=exp.old.score_feature(head,expected)['logit']
            for kind in ('blur','mean'):
                a=exp.np.zeros((518,518),exp.np.float32);a[28:42,28:42]=3;a[420:434,252:266]=4
                total=float(a.astype(exp.np.float64).sum());raw[kind]=a
                diag[kind]=dict(baseline_logit=target-total,target_minus_baseline=total,attribution_sum=total,
                    completeness_residual=0.,relative_residual=0.,completeness_warning=False,baseline_input_sha256='fixture')
            return dict(raw_maps=raw,target_feature=expected.copy(),geometry=exp.ig.geometry(im.size),target_logit=target,
                diagnostics=diag,original_input_sha256=exp.sha(im.tobytes()),forward_passes=67,backward_passes=64)
    out=here/(exp.PREFIX+'_actual_input_fixture_'+constants['SOURCE_SHA256'][:12])
    command=['--capsule-dir',str(root),'--sup-zip',str(args.sup_zip),'--out-dir',str(out),'--n-boot','30']
    fixture=Fixture();assert exp.main(command,attributor=fixture)==0 and fixture.calls==26
    status=exp.read(out/'run_status.json');assert status['fixture'] and status['presence_frozen']
    records=exp.read(out/'attributions.json');finals=exp.read(out/'predictions.json');baseline=exp.read(root/'inputs/baseline.json')
    assert len(records)==26 and len(finals)==8
    for policy,preds in finals.items():assert all(preds[c]['present']==baseline[c]['present'] for c in baseline)
    for core in records:
        k=len(finals['ig_consensus_multi'][core]['regions'] or [])
        for policy in ('occlusion_matched_k','center_matched_k','random_matched_k'):assert len(finals[policy][core]['regions'] or [])==k
    again=Fixture();assert exp.main(command,attributor=again)==0 and again.calls==0 and again.stats['forward_passes']==0
    meta=exp.read(out/'run_metadata.json');assert meta['model_loads']==0 and meta['new_attributions']==0
    old_modules={k:sys.modules.get(k) for k in ('google','google.colab')};google=ModuleType('google');colab=ModuleType('google.colab')
    downloads=[];colab.files=SimpleNamespace(download=lambda p:downloads.append(p));google.colab=colab
    sys.modules['google']=google;sys.modules['google.colab']=colab
    def export(folder,drive,ret):
        import hashlib,zipfile
        drive.mkdir(parents=True,exist_ok=True)
        scope=dict(DRIVE=drive,OUT=folder,LOG=here/'fixture_console.log',RETURN_CODE=ret,SOURCE_BYTES=constants['SOURCE_BYTES'],
                   SOURCE_MANIFEST=constants['SOURCE_MANIFEST'],json=json,hashlib=hashlib,zipfile=zipfile)
        exec(compile(code[-1],'<exact-export-cell>','exec'),scope)
        with ZipFile(downloads[-1]) as z:
            names=set(z.namelist());assert not any('/_images/' in n for n in names)
            assert {'source_bundle.zip','run/run_status.json','run/attributions.json','runner_return_code.json'}<=names
            assert json.loads(z.read('run/run_status.json'))['fixture']
    try:
        export(out,here/'successful_fixture_export',0)
        partial=here/(exp.PREFIX+'_failed_fixture_'+constants['SOURCE_SHA256'][:12]);failed=list(command);failed[failed.index('--out-dir')+1]=str(partial)
        assert exp.main(failed,attributor=Fixture(fail=True))==1
        assert not exp.read(partial/'run_status.json')['complete']
        export(partial,here/'failed_fixture_export',1)
    finally:
        for k,v in old_modules.items():
            if v is None:sys.modules.pop(k,None)
            else:sys.modules[k]=v
    result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_missing_ig_location_dev.py','--basetemp=.pytest-extracted'],
                          cwd=root,text=True,capture_output=True)
    (here/'extracted_tests.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
    assert result.returncode==0,result.stdout+result.stderr
    evidence=dict(source_bundle_sha256=constants['SOURCE_SHA256'],source_files=len(constants['SOURCE_MANIFEST']),
        original_vectors_verified=50,canary_seed_vectors=1,source_actual_archive='bad35ced5b0b',routed_images=26,
        actual_input_maps_fixture=52,policies=8,presence_frozen_all240=True,matched_cardinality_all_routes=True,
        resume_new_calls=0,resume_model_loads=0,exact_export_success_and_failure=True,originals_excluded=True,
        extracted_tests=result.stdout.strip(),actual_pretrained_neural_calls=0,
        attribution_effect='UNRUN; fixed synthetic maps are fixture plumbing only',
        cpu_gradient_tests='analytic linear/cubic and tiny random HF Dinov2Model; no pretrained weights')
    exp.write(here/'verification.json',evidence);print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()

"""Real source/image wiring with explicitly synthetic maps, plus exact export cells."""
import argparse,ast,base64,importlib.util,io,json,sys
from pathlib import Path
from types import ModuleType,SimpleNamespace
from zipfile import ZipFile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import run_missing_ig_numerics as exp

def main():
    p=argparse.ArgumentParser();p.add_argument('--sup-zip',type=Path,required=True);args=p.parse_args()
    here=Path(__file__).resolve().parent;receipt=exp.read(here/'notebook_build.json')
    nb=exp.read(ROOT/'notebooks/lvlm_missing_ig_numerics_colab.ipynb')
    code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'];assert len(code)==5
    for c in code:ast.parse(c)
    constants={}
    for node in ast.parse(code[2]).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            n=node.targets[0].id
            if n in ('SOURCE_SHA256','SOURCE_MANIFEST'):constants[n]=ast.literal_eval(node.value)
            if n=='SOURCE_BYTES':constants[n]=base64.b64decode(ast.literal_eval(node.value.args[0]))
    assert exp.sha(constants['SOURCE_BYTES'])==receipt['source_bundle_sha256']==constants['SOURCE_SHA256']
    root=here/('extracted_'+constants['SOURCE_SHA256'][:12])
    with ZipFile(io.BytesIO(constants['SOURCE_BYTES'])) as z:
        assert set(z.namelist())==set(constants['SOURCE_MANIFEST'])
        assert not any('evaluation_gt' in n or 'annotations' in n for n in z.namelist())
        for n in z.namelist():
            b=z.read(n);assert exp.sha(b)==constants['SOURCE_MANIFEST'][n]
            path=exp.old.old.checked_path(root,n);path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():assert path.read_bytes()==b
            else:path.write_bytes(b)
    exp.write(root/'source_manifest.json',constants['SOURCE_MANIFEST'])
    seeds=exp.read(root/'inputs/seeds.json')
    vectors={s['pixel_sha256']:exp.old.old.check_vector((root/s['feature_path']).read_bytes(),768) for s in seeds.values()}
    spec=importlib.util.spec_from_file_location('fixture_test',ROOT/'tests/test_missing_ig_numerics.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class RealInputFixture(module.Fixture):
        def feature(self,im):
            self.stats['forward_passes']+=1
            return vectors[exp.sha(im.tobytes())].copy()
    out=here/('missing_ig_numerics_actual_fixture_'+constants['SOURCE_SHA256'][:12])
    argv=['--capsule-dir',str(root),'--sup-zip',str(args.sup_zip),'--out-dir',str(out)]
    engine=RealInputFixture();assert exp.main(argv,engine=engine)==0 and engine.calls==40
    m=exp.read(out/'run_metadata.json');assert m['fixture'] and m['forward_passes']==6810 and m['backward_passes']==6448
    assert exp.read(out/'run_status.json')['saved']==40
    again=RealInputFixture();assert exp.main(argv,engine=again)==0 and again.calls==0
    assert exp.read(out/'run_metadata.json')['model_loads']==exp.read(out/'run_metadata.json')['forward_passes']==0
    bad=here/('missing_ig_numerics_failure_fixture_'+constants['SOURCE_SHA256'][:12])
    failed=RealInputFixture(fail=True);argv[-1]=str(bad);assert exp.main(argv,engine=failed)==1
    assert exp.read(bad/'run_status.json')['condition_errors']==40
    old_modules={k:sys.modules.get(k) for k in ('google','google.colab')};google=ModuleType('google');colab=ModuleType('google.colab')
    downloads=[];colab.files=SimpleNamespace(download=lambda p:downloads.append(p));google.colab=colab
    sys.modules['google']=google;sys.modules['google.colab']=colab
    def export(folder,drive,ret):
        import hashlib,zipfile
        drive.mkdir(parents=True,exist_ok=True)
        scope=dict(DRIVE=drive,OUT=folder,LOG=here/'fixture_console.log',RETURN_CODE=ret,
            SOURCE_BYTES=constants['SOURCE_BYTES'],SOURCE_MANIFEST=constants['SOURCE_MANIFEST'],json=json,hashlib=hashlib,zipfile=zipfile)
        exec(compile(code[-1],'<exact-export-cell>','exec'),scope)
        with ZipFile(downloads[-1]) as z:
            assert not any('/_images/' in n for n in z.namelist())
            assert {'source_bundle.zip','source_manifest.json','run/records.json','run/run_status.json','runner_return_code.json'}<=set(z.namelist())
            assert json.loads(z.read('run/run_status.json'))['fixture']
            assert json.loads(z.read('runner_return_code.json'))['return_code']==ret
            assert exp.sha(z.read('source_bundle.zip'))==receipt['source_bundle_sha256']
    try:export(out,here/'fixture_success_export',0);export(bad,here/'fixture_failure_export',1)
    finally:
        for k,v in old_modules.items():
            if v is None:sys.modules.pop(k,None)
            else:sys.modules[k]=v
    exp.write(here/'actual_input_verification.json',dict(source_bundle_sha256=receipt['source_bundle_sha256'],
        selected_cores=receipt['selected_cores'],source_files=len(receipt['source_files']),code_cells=5,
        actual_image_wiring=True,fixture=True,conditions=40,forward_passes=6810,backward_passes=6448,
        resume_model_loads=0,resume_forward_passes=0,success_failure_exact_export=True,pretrained_neural_calls=0,
        extracted_root=str(root),native_diagnostic='UNRUN'))

if __name__=='__main__':main()

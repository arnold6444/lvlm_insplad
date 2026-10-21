"""Extract the exact notebook capsule and wire all20 actual images with a constant fixture model."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--notebook',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sup-zip',type=Path,required=True);p.add_argument('--unsup-zip',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    notebook=json.loads(args.notebook.read_text(encoding='utf-8'));codes=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
    assert len(codes)==4
    for code in codes:compile(code,'<notebook>','exec')
    # Only execute the literal capsule assignments; never mount/install/download or run a model.
    import base64
    scope={'base64':base64}
    tree=ast.parse(codes[1])
    for node in tree.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['SOURCE_SHA256','SOURCE_MANIFEST','SOURCE_BYTES']:
            exec(compile(ast.Module([node],type_ignores=[]),'<sealed-literal>','exec'),scope)
    source=scope['SOURCE_BYTES'];assert hashlib.sha256(source).hexdigest()==scope['SOURCE_SHA256']
    import io
    code=args.out/'extracted'
    with ZipFile(io.BytesIO(source)) as z:
        assert set(z.namelist())==set(scope['SOURCE_MANIFEST'])
        for n in z.namelist():
            b=z.read(n);assert hashlib.sha256(b).hexdigest()==scope['SOURCE_MANIFEST'][n]
            dest=code/n;assert dest.resolve().is_relative_to(code.resolve());dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
    sys.path[:0]=[str(code/'scripts'),str(code/'src')]
    import run_q3_gt_crop_diagnostic_dev as run
    data=run.read(code/'inputs/dataset.json');assert len(data['rows'])==20 and len(data['excluded'])==5
    assert run.sha((code/'inputs/annotations.json').read_bytes())==data['annotation_sha256']
    run.validate(data)
    calls=[];loads=[]
    def infer(adapter,images,prompt):
        calls.append((str(images[-1]),prompt))
        assert len(images)==3 and adapter['image_tile_budgets']==[3,3,2]
        assert run.Image.open(images[-1]).size==(1024,548)
        return '{"regions":[]}'
    def loader(_):loads.append(1);return {'kind':'internvl_chat'}
    argv=['--dataset',str(code/'inputs/dataset.json'),'--sup-zip',str(args.sup_zip),'--unsup-zip',str(args.unsup_zip),
          '--out-dir',str(args.out/(run.PREFIX+'_actual_images_fixture')),'--n-boot','30']
    assert run.main(argv,loader=loader,infer=infer)==0 and len(calls)==60 and len(loads)==1
    assert len({p for _,p in calls})==1
    out=Path(argv[7]);man=run.read(out/'crop_manifest.json');assert len(man)==60
    # Independent rectangular geometry and full-panel pixel replay for every actual crop.
    from PIL import Image,ImageOps
    import numpy as np
    byid={r['core_id']:r for r in data['rows']};verified_pixels=0
    with ZipFile(args.sup_zip) as z:
        for m in man:
            r=byid[m['core_id']];original=Image.open(io.BytesIO(z.read(r['image']))).convert('RGB')
            path=out/'crops'/(r['core_id']+'_'+m['arm']+'.png')
            assert run.sha(path.read_bytes())==m['crop_sha256']
            sheet=Image.open(path).convert('RGB')
            for i,b in enumerate(m['boxes']):
                if m['arm']=='oracle_box':
                    x,y,w,h=r['boxes'][i];iw,ih=r['size']
                    import math
                    assert b==[max(0,math.floor((x-.25*w)*iw)),max(0,math.floor((y-.25*h)*ih)),min(iw,math.ceil((x+1.25*w)*iw)),min(ih,math.ceil((y+1.25*h)*ih))]
                else:
                    cell=m['cells'][i];rr,cc=divmod(cell-1,3);iw,ih=r['size']
                    assert b==[cc*iw//3,rr*ih//3,(cc+1)*iw//3,(rr+1)*ih//3]
                expected=ImageOps.contain(original.crop(b),(512,512),Image.Resampling.BICUBIC)
                area=(i*512+(512-expected.width)//2,36+(512-expected.height)//2)
                actual=sheet.crop((*area,area[0]+expected.width,area[1]+expected.height))
                assert np.array_equal(np.asarray(expected),np.asarray(actual));verified_pixels+=1
    calls.clear();loads.clear();assert run.main(argv,loader=loader,infer=infer)==0 and not calls and not loads
    # Execute the notebook's final export cell with a local fake download callback only.
    import types
    google=types.ModuleType('google');colab=types.ModuleType('google.colab');downloaded=[]
    colab.files=types.SimpleNamespace(download=lambda path:downloaded.append(path));google.colab=colab
    sys.modules['google']=google;sys.modules['google.colab']=colab
    drive=args.out/'fake_drive';drive.mkdir(exist_ok=True);log=drive/'fixture.log';log.write_text('constant fixture; neural0\n')
    scope.update(DRIVE=drive,OUT=out,RETURN_CODE=0,LOG=log,zipfile=__import__('zipfile'),json=json,hashlib=hashlib)
    exec(compile(codes[3],'<export-cell>','exec'),scope)
    assert len(downloaded)==1
    with ZipFile(downloaded[0]) as z:
        assert not any('/_images/' in n for n in z.namelist())
        assert len(json.loads(z.read('run/calls.json')))==60
        assert json.loads(z.read('run/run_status.json'))['fixture'] is True
        assert z.read('source_bundle.zip')==source
    receipt=dict(source_bundle_sha256=scope['SOURCE_SHA256'],source_files=len(scope['SOURCE_MANIFEST']),
        compiled_code_cells=4,paired_actual_images=20,excluded=5,fixture_calls=60,fixture_loader_calls=1,
        resume_new_calls=0,resume_model_loads=0,verified_crop_sheets=60,verified_crop_panels=verified_pixels,
        exact_prompt_count=1,neural_calls=0,actual_model_effect='unrun',normal_support=0,
        final_export_cell_checked=True,exported_raw_calls=60,
        source_image_checks='all used target/reference SHA; original target dimensions/EXIF',
        warning='constant empty-response fixture; not actual LVLM evidence')
    (args.out/'actual_input_verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

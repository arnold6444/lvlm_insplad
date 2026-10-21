"""Human-GT crop diagnosis on a fixed dev failure cohort; never an Agent score."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import math
import re
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
import run_q3_missing_occlusion_dev as old
import dev_missing_location_annotation as annotation

PROTOCOL = 'q3_gt_crop_diagnostic_dev_v1'
PREFIX = 'q3_gt_crop_diagnostic_dev'
ARMS = ('candidate_grid', 'oracle_grid', 'oracle_box')
CONFIG = dict(box_padding_per_side=.25, panel_size=512, slots=2, sheet_size=[1024,548],
              image_tile_budgets=[3,3,2], max_new_tokens=80, do_sample=False,
              selection='previously reviewed missing-cap AND confirmed spatial GT',
              purpose='oracle diagnosis only; not an Agent or holdout score')
sha, digest, read, write = old.sha, old.digest, old.read, old.write


def validate(data, fixture=False):
    if data.get('protocol') != PROTOCOL or data.get('purpose') != 'oracle_diagnosis_only' or data.get('new_holdout') is not False:
        raise ValueError('separate diagnostic dev capsule required')
    rows = data['rows']
    if not rows or len({r['core_id'] for r in rows}) != len(rows):
        raise ValueError('empty/duplicate diagnostic cohort')
    if not fixture and len(rows) != 20:
        raise ValueError('frozen paired20 required')
    expected = {(r['role'],r['member']): r['sha256'] for r in data['image_manifest']}
    if len(expected) != len(data['image_manifest']):
        raise ValueError('duplicate image identity')
    for r in rows:
        if not isinstance(r['core_id'],str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',r['core_id']):
            raise ValueError('unsafe image ID')
        if not isinstance(r['size'],list) or len(r['size'])!=2 or any(type(v) is not int or v<3 for v in r['size']):
            raise ValueError('invalid image size')
        if r['protocol'] != PROTOCOL or r['source_protocol'] != 'gold_v2_dev' or r['asset_category'] != 'glass-insulator' or r['defect_gt'] != 'missing-cap' or r['anomaly_gt'] != 'anomalous':
            raise ValueError('gold/non-dev/nonmissing input forbidden')
        if r['annotation_status'] != 'boxed' or r['cached_first'] != []:
            raise ValueError('only confirmed prior-empty reviewed cases')
        if not 1 <= len(r['boxes']) <= CONFIG['slots']:
            raise ValueError('unsupported site count; never truncate GT')
        annotation.clean_boxes(r['boxes'],r['size'])
        if r['regions_gt'] != annotation.box_regions(r['boxes']):
            raise ValueError('box center GT mismatch')
        if old.prior.spatial.parse_json(old.json.dumps({'regions':r['regions_gt']}),'regions')!=r['regions_gt']:
            raise ValueError('invalid spatial GT')
        previous=r['previous_candidate_prediction']
        if previous is not None and old.prior.spatial.parse_json(old.json.dumps({'regions':previous}),'regions')!=previous:
            raise ValueError('invalid descriptive cached prediction')
        p = r['candidate_cells']
        if not p or len(p)>2 or old.prior.spatial.parse_json(old.json.dumps({'regions':p}),'regions') != sorted(p):
            raise ValueError('invalid frozen candidate cells')
        ref = r['normal_reference_image']
        if not ref.startswith('glass-insulator/train/') or not any('/train/'+v+'/' in ref for v in ['good','normal']):
            raise ValueError('same-asset train-normal reference required')
        for role,member in [('target',r['image']),('reference',ref)]:
            h=expected.get((role,member))
            if not h or len(h)!=64 or any(c not in '0123456789abcdef' for c in h):
                raise ValueError('missing image SHA')
        if expected['target',r['image']] != r['image_sha256']:
            raise ValueError('annotation image SHA mismatch')
    return expected


def padded_box(size, b):
    annotation.clean_boxes([b],size)
    iw,ih=size; x,y,w,h=b; p=CONFIG['box_padding_per_side']
    return [max(0,math.floor((x-p*w)*iw)),max(0,math.floor((y-p*h)*ih)),
            min(iw,math.ceil((x+w+p*w)*iw)),min(ih,math.ceil((y+h+p*h)*ih))]


def geometry(row, arm):
    if arm == 'candidate_grid':
        cells = row['candidate_cells']
        bounds = [list(old.box(row['size'],c)) for c in cells]
    elif arm == 'oracle_grid':
        cells = row['regions_gt']
        bounds = [list(old.box(row['size'],c)) for c in cells]
    elif arm == 'oracle_box':
        cells = [annotation.box_regions([b])[0] for b in row['boxes']]
        bounds = [padded_box(row['size'],b) for b in row['boxes']]
    else:
        raise ValueError('unknown arm')
    return cells,bounds


def make_sheet(image, cells, bounds):
    if not 1 <= len(bounds) <= 2 or len(cells) != len(bounds):
        raise ValueError('invalid crop count')
    canvas=Image.new('RGB',tuple(CONFIG['sheet_size']),'white'); draw=ImageDraw.Draw(canvas)
    for i,b in enumerate(bounds):
        crop=ImageOps.contain(image.crop(b),(512,512),Image.Resampling.BICUBIC)
        canvas.paste(crop,(i*512+(512-crop.width)//2,36+(512-crop.height)//2))
        draw.text((i*512+12,10),f'Target detail {i+1}',fill='black')
    return canvas


def prompt():
    # Identical prompt in all arms. GT cell numbers/boxes are never printed as an answer cue.
    return old.prior.spatial.prompt('regions',True)+(
        '\nImage 3 contains unmodified enlarged detail crops from Image 2. '
        'A white unused panel contains no additional crop. The crop selection is not proof of a defect. '
        'Use the full original Image 2 to determine the original grid coordinates of any defect. '
        'Your earlier answer to THIS Q3 question was {"regions":[]}. '
        'Inspect the full TARGET and crop details independently. You may report original target cells '
        'outside these crops, or an empty list. Perspective and occlusion alone do not prove loss. '
        'Return only JSON {"regions":[1,9]} or {"regions":[]}, without explanation.')


def metrics(rows,predictions):
    report=old.prior.spatial.reports(rows,predictions)
    report['overall'].update(any_location_hit=sum(bool(set(r['regions_gt']) & set(predictions.get(r['core_id']) or [])) for r in rows),
        all_gt_locations_hit=sum(predictions.get(r['core_id']) is not None and set(r['regions_gt']) <= set(predictions[r['core_id']]) for r in rows),
        diagnostic_only=True,normal_fpr=None,normal_support=0)
    return report


def paired(rows, finals, eligible, n_boot):
    result=[]; rng=np.random.default_rng(42)
    for baseline,condition in [('candidate_grid','oracle_grid'),('candidate_grid','oracle_box'),('oracle_grid','oracle_box')]:
        if not eligible[baseline] or not eligible[condition]:
            continue
        for metric in ['presence','any_location','exact_location']:
            def value(r,p):
                v=finals[p][r['core_id']]
                if metric=='presence':return int(bool(v))
                if metric=='any_location':return int(bool(set(r['regions_gt'])&set(v or [])))
                return int(v is not None and set(v)==set(r['regions_gt']))
            delta=np.array([value(r,condition)-value(r,baseline) for r in rows],float)
            dist=delta[rng.integers(0,len(rows),size=(n_boot,len(rows)))].mean(1)
            result.append(dict(baseline=baseline,condition=condition,metric=metric,n=len(rows),
                delta=float(delta.mean()),ci_low=float(np.quantile(dist,.025)),ci_high=float(np.quantile(dist,.975)),
                diagnostic_only=True,interpretation='paired fixed-dev diagnostic; oracle arms use human GT'))
    return result


def run(args, *, loader=None, infer=None):
    started=time.time()
    if (loader is None)!=(infer is None):raise ValueError('both fixture callbacks required')
    fixture=loader is not None; data=read(args.dataset); expected=validate(data,fixture)
    if not 1<=args.n_boot<=20000:raise ValueError('invalid bootstrap count')
    out=args.out_dir
    if not out.name.startswith(PREFIX) or out.resolve().is_relative_to(args.dataset.parent.resolve()) or any(p.resolve().is_relative_to(out.resolve()) for p in [args.dataset,args.sup_zip,args.unsup_zip]):
        raise ValueError('separate diagnostic output required')
    if not fixture and (not args.model_path or args.model_path.name!=old.prior.MODEL_REVISION or not args.model_path.is_dir()):
        raise ValueError('pinned native model snapshot required')
    config=dict(protocol=PROTOCOL,input_sha256=old.prior.probe.sha_file(args.dataset),
                code_sha256=old.prior.shared.code_hashes(),config=CONFIG,fixture=fixture,n_boot=args.n_boot,
                model_id=old.prior.MODEL_ID,model_revision=old.prior.MODEL_REVISION,model_path=str(args.model_path))
    identity=digest(config); meta_path=out/'run_metadata.json'
    prev=read(meta_path) if meta_path.exists() else {}
    if prev and prev['identity']!=identity:raise ValueError('changed input/code/model; use new folder')
    if not prev and out.exists() and any(out.iterdir()):raise ValueError('unknown output directory')
    rows=data['rows']
    targets=old.prior.old.verified_images(args.sup_zip,[r['image'] for r in rows],out/'_images/target')
    normals=old.prior.old.verified_images(args.unsup_zip,[r['normal_reference_image'] for r in rows],out/'_images/normal')
    for role,paths in [('target',targets),('reference',normals)]:
        for member,path in paths.items():
            if old.prior.probe.sha_file(path)!=expected[role,member]:raise ValueError('original image SHA changed')
    # All source orientation/dimensions are checked before any model call.
    for r in rows:
        if list(old.original_image(targets[r['image']]).size)!=r['size']:raise ValueError('image dimensions changed')
    out.mkdir(parents=True,exist_ok=True)
    meta={**config,'identity':identity,'complete':False,'backend':'fixture' if fixture else 'hf',
          'images':len(rows),'normal_support':0,'diagnostic_only':True,'oracle_uses_human_gt':True,
          'new_holdout':False,'agent_score':None,'q1_calls':0,'q2_calls':0,'gold_used':False,
          'training_calls':0,'dino_calls':0,'label_sha256':data['annotation_sha256'],
          'source_run_sha256':data['source_run_sha256'],
          'claim_limit':'fixed previously examined positive dev20; oracle input is not deployable Agent evidence'}
    write(meta_path,meta);write(out/'run_status.json',dict(complete=False,phase='crop_review',diagnostic_only=True))
    (out/'dataset_snapshot.json').write_bytes(args.dataset.read_bytes())
    cached=read(out/'calls.json') if (out/'calls.json').exists() else {}
    calls={}; crop_manifest=[]; finals={p:{} for p in ARMS}; counts=Counter()
    spec=dict(old.prior.agent.base.MODEL_REGISTRY[old.prior.MODEL])
    if not fixture:
        spec['model_id']=str(args.model_path)
        def loader(model_spec):
            import torch
            if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<20*1024**3:
                raise RuntimeError('native L4/A100 VRAM >=20GB required')
            torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
            return old.prior.hf.load_hf_adapter(model_spec)
        infer=old.prior.hf.infer_hf
    adapter=old.prior.agent.base.LazyHFAdapter(spec,loader=loader); load_failure=''
    try:
        for row in rows:
            core=row['core_id']; image=old.original_image(targets[row['image']])
            shift=int(hashlib.sha256(core.encode()).hexdigest(),16)%3
            order=ARMS[shift:]+ARMS[:shift]
            for arm in order:
                cells,bounds=geometry(row,arm)
                crop=out/'crops'/(core+'_'+arm+'.png'); crop.parent.mkdir(exist_ok=True)
                payload=old.png(make_sheet(image,cells,bounds))
                if crop.exists() and crop.read_bytes()!=payload:raise ValueError('changed crop bytes')
                crop.write_bytes(payload)
                crop_manifest.append(dict(core_id=core,arm=arm,cells=cells,boxes=bounds,source_sha256=row['image_sha256'],
                    crop_sha256=sha(payload),sheet_size=CONFIG['sheet_size'],oracle_uses_gt=arm!='candidate_grid'))
                images=[normals[row['normal_reference_image']],targets[row['image']],crop]
                request=dict(identity=identity,prompt=prompt(),images=list(map(str,images)),
                    image_sha256=[expected['reference',row['normal_reference_image']],row['image_sha256'],sha(payload)],
                    image_tile_budgets=CONFIG['image_tile_budgets'],generation={'max_new_tokens':80,'do_sample':False})
                key=core+'/'+arm;rec=cached.get(key); req_sha=digest(request)
                if rec and (rec['request_sha256']!=req_sha or any(rec[k]!=v for k,v in request.items()) or rec['core_id']!=core or rec['arm']!=arm or rec['id']!=key):
                    raise ValueError('changed cached request')
                if rec and not rec['error']:
                    if old.prior.spatial.parse_json(rec['output'],'regions')!=rec['prediction']:raise ValueError('changed cached parsing')
                    counts['reused_lvlm_calls']+=1
                else:
                    if rec:write(out/'attempt_history'/(digest(rec)+'.json'),rec)
                    output='';error=load_failure;start=time.time()
                    if not error:
                        try:model=adapter.get()
                        except Exception as exc:load_failure=error=f'{type(exc).__name__}: {exc}'
                        if not error:
                            counts['new_lvlm_calls']+=1
                            try:output=str(infer({**model,'image_tile_budgets':CONFIG['image_tile_budgets']},images,request['prompt']))
                            except Exception as exc:
                                error=f'{type(exc).__name__}: {exc}'
                                if 'out of memory' in error.lower():load_failure=error
                    rec=dict(id=key,core_id=core,arm=arm,**request,request_sha256=req_sha,output=output,
                        prediction=old.prior.spatial.parse_json(output,'regions'),error=error,latency_s=round(time.time()-start,4))
                calls[key]=rec;write(out/'calls.json',{**cached,**calls});finals[arm][core]=rec['prediction']
                write(out/'crop_manifest.json',crop_manifest)
                write(out/'run_status.json',dict(complete=False,phase='crop_review',saved=len(calls),expected_calls=3*len(rows),diagnostic_only=True))
            print('diagnostic saved',len(calls),'/',3*len(rows),flush=True)
    finally:
        adapter.release()
    if set(cached)-set(calls):raise ValueError('unexpected cached keys')
    errors={p:sum(bool(r['error']) for r in calls.values() if r['arm']==p) for p in ARMS}
    eligible={p:errors[p]==0 for p in ARMS}; leaderboard=[]
    for arm in ARMS:
        report=metrics(rows,finals[arm]);write(out/('metrics_'+arm+'.json'),{'eligible':eligible[arm],**report})
        if eligible[arm]:leaderboard.append(dict(arm=arm,**report['overall']))
    # Earlier candidate answers have different prompt/sheet and are descriptive only.
    old_answers={r['core_id']:r['previous_candidate_prediction'] for r in rows}
    write(out/'metrics_cached_previous_descriptive_only.json',metrics(rows,old_answers))
    old.prior.agent.cons.write_csv(out/'leaderboard.csv',leaderboard)
    old.prior.agent.cons.write_csv(out/'paired_diagnostic.csv',paired(rows,finals,eligible,args.n_boot))
    write(out/'predictions.json',finals)
    status=dict(complete=all(eligible.values()),eligible=eligible,execution_errors_by_arm=errors,
                invalid_new_outputs=sum(r['prediction'] is None for r in calls.values()),
                saved_calls=len(calls),expected_calls=3*len(rows),diagnostic_only=True,fixture=fixture)
    meta.update(**counts,complete=status['complete'],raw_new_lvlm_calls=len(calls),
                runner_wall_time_s=round(time.time()-started,4),gpu=old.prior.agent.gpu_info())
    meta['software']={}
    for name in ['torch','transformers','accelerate','numpy','pillow','timm']:
        try:meta['software'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:meta['software'][name]=None
    write(meta_path,meta);write(out/'run_status.json',status);print(old.json.dumps(status),flush=True)
    return 0 if status['complete'] else 1


def main(argv=None,**kwargs):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['dataset','sup-zip','unsup-zip','out-dir']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model-path',type=Path);p.add_argument('--n-boot',type=int,default=2000)
    return run(p.parse_args(argv),**kwargs)


if __name__=='__main__':raise SystemExit(main())

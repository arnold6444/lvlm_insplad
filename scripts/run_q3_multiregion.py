"""Direct/staged Q3 set inference with immutable inputs and per-call checkpoints."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import q3_multiregion as q3
import run_gold_baselines as base

POLICIES = ('direct', 'staged')
SOURCES = ('scripts/q3_multiregion.py', 'scripts/run_q3_multiregion.py',
           'scripts/run_gold_baselines.py', 'scripts/build_explainable_qa_gold.py',
           'src/insplad_benchmark/hf_adapters.py')


def validate_rows(rows):
    if not rows or len({r['core_id'] for r in rows}) != len(rows) or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('empty/duplicate dataset')
    for r in rows:
        if r['protocol'] != q3.PROTOCOL or r['anomaly_gt'] not in ('normal','anomalous'):
            raise ValueError('wrong dataset protocol/label')
        gt = r['regions_gt']
        if (r['anomaly_gt']=='normal') != (r['defect_gt']=='normal') or r['location_gt_known'] != (gt is not None):
            raise ValueError('inconsistent binary/defect/known-location labels')
        if r['anomaly_gt']=='normal':
            if gt != [] or r['normalized_bboxes_xywh'] or r['label_basis']!='normal_source_label':
                raise ValueError('invalid normal spatial GT')
        elif gt is None:
            if r['label_basis']!='manual_unlocatable' or r['normalized_bboxes_xywh']:
                raise ValueError('unknown spatial GT must be explicit')
        elif gt != q3.boxes_to_regions(r['normalized_bboxes_xywh']):
            raise ValueError('spatial GT differs from independent box centers')


def extract(zip_path, members, out):
    paths, hashes = {}, {}
    with ZipFile(zip_path) as z:
        names = {name.replace('\\','/').lstrip('/'):name for name in z.namelist()}
        for member in sorted(set(members)):
            path = out / member
            if '\\' in member or not path.resolve().is_relative_to(out.resolve()) or ':' in member:
                raise ValueError('unsafe image member')
            if member not in names:
                raise ValueError('image absent from archive: '+member)
            payload=z.read(names[member])
            if path.exists() and path.read_bytes()!=payload:
                raise ValueError('cached image changed; use new output folder')
            if not path.exists():
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(payload)
            paths[member]=path
            hashes[member]=hashlib.sha256(payload).hexdigest()
    return paths,hashes


def run(rows, sup_zip, unsup_zip, out, *, model_snapshot, policies=POLICIES,
        normal_reference=True, infer=None, loader=None, checkpoint_every=20):
    validate_rows(rows)
    if not policies or len(set(policies))!=len(policies) or set(policies)-set(POLICIES) or checkpoint_every<=0:
        raise ValueError('invalid policies/checkpoint interval')
    out=Path(out)
    model_snapshot=Path(model_snapshot)
    fixture = infer is not None and loader is not None
    if not fixture and (not model_snapshot.is_dir() or not re.fullmatch('[0-9a-f]{40}',model_snapshot.name)):
        raise ValueError('use an existing pinned HF snapshot directory ending in its 40-character revision')
    spec={**base.MODEL_REGISTRY['internvl3_5_8b'],'model_id':str(model_snapshot.resolve())}
    config={'protocol':q3.PROTOCOL,'dataset_hash':q3.digest(rows),'policies':list(policies),
            'normal_reference':normal_reference,'model_spec':spec,'model_revision':model_snapshot.name,
            'fixture':fixture,'prompts':{s:q3.prompt(s,normal_reference) for s in ('presence','regions')},
            'code_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}}
    lockpath=out/'input_lock.json'
    oldlock=json.loads(lockpath.read_text(encoding='utf-8')) if lockpath.exists() else None
    if oldlock is not None and oldlock['config']!=config:
        raise ValueError('code/data/model/prompt/condition changed; use a new output folder')
    if oldlock is None and any((out/name).exists() for name in ('calls.json','run_status.json','predictions.json')):
        raise ValueError('orphan output without input lock; use a new output folder')
    targets, target_hashes=extract(sup_zip,[r['image'] for r in rows],out/'_images'/'target')
    refs,ref_hashes=extract(unsup_zip,[r['normal_reference_image'] for r in rows],out/'_images'/'reference') if normal_reference else ({},{})
    lock={'config':config,'target_sha256':target_hashes,'reference_sha256':ref_hashes}
    if oldlock is not None and oldlock!=lock:
        raise ValueError('archive bytes changed; use a new output folder')
    q3.atomic_json(lockpath,lock)
    q3.atomic_json(out/'dataset.json',rows)
    calls_path=out/'calls.json'
    calls=json.loads(calls_path.read_text(encoding='utf-8')) if calls_path.exists() else {}
    called=reused=0
    holder=base.LazyHFAdapter(spec,loader=loader)
    metapath=out/'run_metadata.json'
    runtime=json.loads(metapath.read_text(encoding='utf-8'))['runtime'] if metapath.exists() else {'fixture':fixture}

    def call(core,policy,stage,paths):
        nonlocal called,reused,runtime
        key=f'{core}/{policy}/{stage}'
        text=q3.prompt(stage,normal_reference)
        request=q3.digest({'lock':lock,'key':key,'prompt':text})
        old=calls.get(key)
        if old:
            if old.get('request_hash')!=request or old.get('prompt')!=text:
                raise ValueError('stale/corrupt call cache')
            if not old.get('error'):
                reused+=1
                return q3.parse_json(old['output'],'present' if stage=='presence' else 'regions')
        if not fixture and not holder.loaded:
            import torch
            if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<20*1024**3:
                raise RuntimeError('L4/A100 with VRAM >=20GB required; no local CPU inference')
            runtime={'fixture':False,'gpu':torch.cuda.get_device_name(0),
                     'vram_bytes':torch.cuda.get_device_properties(0).total_memory}
        adapter=holder.get()
        generation=infer
        if generation is None:
            from insplad_benchmark.hf_adapters import infer_hf
            generation=infer_hf
        started=time.time()
        output,error='',''
        try:
            output=str(generation(adapter,paths,text))
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
        parsed=q3.parse_json(output,'present' if stage=='presence' else 'regions') if not error else None
        record={'core_id':core,'policy':policy,'stage':stage,'prompt':text,
                'request_hash':request,'output':output,'parsed':parsed,'error':error,
                'latency_s':time.time()-started,'image_sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]}
        if old:
            record['previous_attempts']=old.get('previous_attempts',[])+[{k:v for k,v in old.items() if k!='previous_attempts'}]
        calls[key]=record
        q3.atomic_json(calls_path,calls)  # Save every call, not only at the progress interval.
        called+=1
        if called%checkpoint_every==0:
            print(f'{called} new calls saved; errors={sum(bool(c["error"]) for c in calls.values())}',flush=True)
        return parsed

    statuspath=out/'run_status.json'
    q3.atomic_json(statuspath,{'status':'running','fixture':fixture})
    predictions={}
    try:
        for policy in policies:
            predicted={}
            for r in rows:
                target=targets[r['image']]
                paths=[refs[r['normal_reference_image']],target] if normal_reference else [target]
                if policy=='direct':
                    value=call(r['core_id'],policy,'regions',paths)
                else:
                    present=call(r['core_id'],policy,'presence',paths)
                    value=[] if present is False else call(r['core_id'],policy,'regions',paths) if present is True else None
                predicted[r['core_id']]=value
            predictions[policy]=predicted
            q3.atomic_json(out/'predictions.json',predictions)
    except BaseException as exc:
        q3.atomic_json(statuspath,{'status':'interrupted','error':f'{type(exc).__name__}: {exc}','fixture':fixture})
        raise
    finally:
        holder.release()
    for policy,predicted in predictions.items():
        q3.atomic_json(out/f'metrics_{policy}.json',q3.reports(rows,predicted))
    errors=sum(bool(c['error']) for c in calls.values())
    invalid=sum(pred is None for values in predictions.values() for pred in values.values())
    policy_calls=dict(Counter(c['policy'] for c in calls.values()))
    total_attempts=sum(1+len(c.get('previous_attempts',[])) for c in calls.values())
    status={'status':'complete' if not errors else 'has_errors','fixture':fixture,
            'called_this_run':called,'reused_calls':reused,'row_errors':errors,'invalid_final_outputs':invalid,
            'calls_by_policy':policy_calls,'total_attempts_including_retries':total_attempts,
            'expected_image_policy_pairs':len(rows)*len(policies)}
    q3.atomic_json(statuspath,status)
    q3.atomic_json(out/'run_metadata.json',{'protocol':q3.PROTOCOL,'runtime':runtime,'model_revision':model_snapshot.name,
                   'backend':'fixture' if fixture else 'hf','model_id':'OpenGVLab/InternVL3_5-8B',
                   'benchmark_commit':None,'benchmark_code_identity':config['code_sha256'],
                   'code_state':'source-pinned runtime bundle; identify exact code by hashes',
                   'inference_is_fixture':fixture,'native_generation':{'max_new_tokens':80,'do_sample':False,'max_total_tiles':6},
                   'reference_kind':'master_fixed_train_normal' if normal_reference else 'none',
                   'interpretation':'exploratory previously examined gold; unequal call budgets; no structure-only causal claim',
                   'input_lock_sha256':hashlib.sha256(lockpath.read_bytes()).hexdigest(),**status})
    print(json.dumps(status,ensure_ascii=False,indent=2))
    return status


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','sup-zip','unsup-zip','model-snapshot','out-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--policies',nargs='+',choices=POLICIES,default=list(POLICIES))
    p.add_argument('--image-only',action='store_true')
    args=p.parse_args(argv)
    result=run(json.loads(args.dataset.read_text(encoding='utf-8')),args.sup_zip,args.unsup_zip,args.out_dir,
               model_snapshot=args.model_snapshot,policies=args.policies,normal_reference=not args.image_only)
    return 0 if result['status']=='complete' else 1


if __name__=='__main__':
    raise SystemExit(main())

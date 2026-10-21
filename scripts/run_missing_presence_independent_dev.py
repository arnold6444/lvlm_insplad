"""Fresh missing-component presence only; no previous answer, location question or history."""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.metadata
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_q3_gt_crop_diagnostic_dev as crop
import verify_internvl_tokenizer as gate

PROTOCOL='missing_presence_independent_dev_v1'
PREFIX='missing_presence_independent_dev'
ARMS=crop.ARMS
read,write,digest,sha=crop.read,crop.write,crop.digest,crop.sha
CONFIG=dict(crop.CONFIG,task='missing_presence_only',previous_answer_in_prompt=False)


def prompt():
    criterion=crop.old.prior.refined.DEFINITIONS[crop.old.prior.MISSING]
    return ('Image 1 is a normal reference of the same asset category. Image 2 is the TARGET. '
            'Image 3 contains unmodified enlarged details of Image 2; a white unused panel contains no extra detail. '
            'Judge the TARGET only. Reference viewpoint and lighting can differ. '
            'Determine only whether a required insulator component is missing or visibly broken. '
            'Visual criterion: '+criterion+' '
            'Does the TARGET contain this missing/broken-insulator condition? '
            'Return only JSON {"present":true} or {"present":false}, without explanation.')


def parse(text):
    return crop.old.prior.spatial.parse_json(text,'present')


def validate(data,fixture=False):
    if data.get('experiment_protocol')!=PROTOCOL or data.get('source_presence_gt')!='all20_missing_cap':
        raise ValueError('independent missing-presence capsule required')
    crop.validate(data,fixture)
    return {(x['role'],x['member']):x['sha256'] for x in data['image_manifest']}


def metrics(rows, predictions):
    values=[predictions[r['core_id']] for r in rows];n=len(rows)
    if any(x is not None and type(x) is not bool for x in values):raise ValueError('strict bool predictions required')
    tp=sum(x is True for x in values);invalid=sum(x is None for x in values)
    return dict(images=n,positive_support=n,normal_support=0,tp=tp,fn=n-tp,fp=0,tn=0,
        valid_false=sum(x is False for x in values),invalid_anomalous=invalid,invalid_normal=0,
        missing_recall=tp/n,invalid_rate=invalid/n,normal_fpr=None,specificity=None,
        balanced_accuracy=None,location_accuracy=None,diagnostic_only=True)


def comparisons(rows,pred,eligible,n_boot):
    rng=np.random.default_rng(42);result=[]
    for a,b in [('candidate_grid','oracle_grid'),('candidate_grid','oracle_box'),('oracle_grid','oracle_box')]:
        if not eligible[a] or not eligible[b]:continue
        delta=np.array([int(pred[b][r['core_id']] is True)-int(pred[a][r['core_id']] is True) for r in rows],float)
        boot=delta[rng.integers(0,len(rows),size=(n_boot,len(rows)))].mean(1)
        result.append(dict(baseline=a,condition=b,metric='missing_recall',n=len(rows),delta=float(delta.mean()),
            ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),diagnostic_only=True))
    return result


def run(args,*,loader=None,infer=None):
    started=time.time()
    if (loader is None)!=(infer is None):raise ValueError('both fixture callbacks required')
    fixture=loader is not None;data=read(args.dataset);expected=validate(data,fixture);out=args.out_dir
    if not 1<=args.n_boot<=20000:raise ValueError('invalid bootstrap count')
    if not out.name.startswith(PREFIX) or out.resolve().is_relative_to(args.dataset.parent.resolve()) or any(p.resolve().is_relative_to(out.resolve()) for p in [args.dataset,args.sup_zip,args.unsup_zip]):
        raise ValueError('separate output required')
    if not fixture and (not args.model_path or args.model_path.name!=gate.REVISION or not args.model_path.is_dir()):
        raise ValueError('pinned native snapshot required')
    config=dict(protocol=PROTOCOL,input_sha256=crop.old.prior.probe.sha_file(args.dataset),
        code_sha256=crop.old.prior.shared.code_hashes(),config=CONFIG,fixture=fixture,n_boot=args.n_boot,
        model_id=crop.old.prior.MODEL_ID,model_revision=gate.REVISION,model_path=str(args.model_path))
    identity=digest(config);meta_path=out/'run_metadata.json';previous=read(meta_path) if meta_path.exists() else {}
    if previous and previous['identity']!=identity:raise ValueError('changed code/input/model; new folder required')
    if not previous and out.exists() and any(out.iterdir()):raise ValueError('unknown output folder')
    rows=data['rows']
    targets=crop.old.prior.old.verified_images(args.sup_zip,[r['image'] for r in rows],out/'_images/target')
    normals=crop.old.prior.old.verified_images(args.unsup_zip,[r['normal_reference_image'] for r in rows],out/'_images/normal')
    for role,paths in [('target',targets),('reference',normals)]:
        for name,path in paths.items():
            if crop.old.prior.probe.sha_file(path)!=expected[role,name]:raise ValueError('source image SHA mismatch')
    for r in rows:
        if list(crop.old.original_image(targets[r['image']]).size)!=r['size']:raise ValueError('source dimensions changed')
    out.mkdir(parents=True,exist_ok=True)
    meta=dict(**config,identity=identity,complete=False,backend='fixture' if fixture else 'hf',images=len(rows),
        normal_support=0,diagnostic_only=True,new_holdout=False,agent_score=None,
        q1_calls=0,q2_calls=0,gold_used=False,training_calls=0,dino_calls=0,
        source_result_sha256=data['source_result_sha256'],annotation_sha256=data['annotation_sha256'],
        claim_limit='explored positive dev20; oracle inputs; task and prior-answer cue both changed')
    write(meta_path,meta);write(out/'run_status.json',dict(complete=False,phase='tokenizer_gate',fixture=fixture,diagnostic_only=True))
    (out/'dataset_snapshot.json').write_bytes(args.dataset.read_bytes())
    native_receipt=None
    if not fixture:
        _,native_receipt=gate.verify(args.model_path,prompt())
        write(out/'tokenizer_audit.json',native_receipt)
        print('Native tokenizer matches publisher checkpoint; original warnings retained; no Mistral patch.',flush=True)
    cached=read(out/'calls.json') if (out/'calls.json').exists() else {}
    calls={};finals={a:{} for a in ARMS};crops=[];counts=Counter()
    spec=dict(crop.old.prior.agent.base.MODEL_REGISTRY[crop.old.prior.MODEL]);model_gate_status=['unrun']
    if not fixture:
        spec['model_id']=str(args.model_path)
        def loader(model_spec):
            model_gate_status[0]='failed_model_load_or_gate'
            import torch
            if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<20*1024**3:
                raise RuntimeError('native L4/A100 VRAM>=20GB required')
            torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
            model=crop.old.prior.hf.load_hf_adapter(model_spec)
            _,loaded=gate.verify(args.model_path,prompt(),tokenizer=model['tokenizer'])
            for k in ['backend_sha256','pre_tokenizer_sha256','probe_ids_sha256']:
                if loaded[k]!=native_receipt[k]:raise ValueError('actual loaded tokenizer differs from preflight')
            write(out/'tokenizer_loaded_audit.json',loaded)
            model_gate_status[0]='pass'
            return model
        infer=crop.old.prior.hf.infer_hf
    adapter=crop.old.prior.agent.base.LazyHFAdapter(spec,loader=loader);load_failure=''
    try:
        for row in rows:
            core=row['core_id'];im=crop.old.original_image(targets[row['image']])
            shift=int(crop.hashlib.sha256(core.encode()).hexdigest(),16)%3
            for arm in ARMS[shift:]+ARMS[:shift]:
                cells,bounds=crop.geometry(row,arm)
                path=out/'crops'/(core+'_'+arm+'.png');path.parent.mkdir(exist_ok=True)
                payload=crop.old.png(crop.make_sheet(im,cells,bounds))
                if path.exists() and path.read_bytes()!=payload:raise ValueError('changed crop bytes')
                path.write_bytes(payload)
                crops.append(dict(core_id=core,arm=arm,cells=cells,boxes=bounds,source_sha256=row['image_sha256'],
                    crop_sha256=sha(payload),oracle_uses_gt=arm!='candidate_grid'))
                images=[normals[row['normal_reference_image']],targets[row['image']],path]
                request=dict(identity=identity,prompt=prompt(),images=list(map(str,images)),
                    image_sha256=[expected['reference',row['normal_reference_image']],row['image_sha256'],sha(payload)],
                    image_tile_budgets=[3,3,2],generation={'max_new_tokens':80,'do_sample':False})
                key=core+'/'+arm;rec=cached.get(key);req_sha=digest(request)
                if rec and (rec['request_sha256']!=req_sha or any(rec[k]!=v for k,v in request.items()) or rec['id']!=key or rec['core_id']!=core or rec['arm']!=arm):
                    raise ValueError('changed cached request')
                if rec and not rec['error']:
                    if parse(rec['output'])!=rec['prediction'] or (rec['prediction'] is not None and type(rec['prediction']) is not bool):
                        raise ValueError('changed cached parsing')
                    counts['reused_lvlm_calls']+=1
                else:
                    if rec:write(out/'attempt_history'/(digest(rec)+'.json'),rec)
                    output='';error=load_failure;start=time.time()
                    if not error:
                        try:model=adapter.get()
                        except Exception as exc:load_failure=error=f'{type(exc).__name__}: {exc}'
                        if not error:
                            counts['new_lvlm_calls']+=1
                            try:output=str(infer({**model,'image_tile_budgets':[3,3,2]},images,request['prompt']))
                            except Exception as exc:
                                error=f'{type(exc).__name__}: {exc}'
                                if 'out of memory' in error.lower():load_failure=error
                    rec=dict(id=key,core_id=core,arm=arm,**request,request_sha256=req_sha,output=output,
                        prediction=parse(output),error=error,latency_s=round(time.time()-start,4))
                calls[key]=rec;write(out/'calls.json',{**cached,**calls});finals[arm][core]=rec['prediction']
                write(out/'crop_manifest.json',crops)
                write(out/'run_status.json',dict(complete=False,phase='missing_presence',saved=len(calls),expected_calls=3*len(rows),diagnostic_only=True))
            print('presence saved',len(calls),'/',3*len(rows),flush=True)
    finally:adapter.release()
    if set(cached)-set(calls):raise ValueError('unexpected cached entries')
    errors={a:sum(bool(x['error']) for x in calls.values() if x['arm']==a) for a in ARMS}
    eligible={a:errors[a]==0 for a in ARMS};leader=[]
    for arm in ARMS:
        score=metrics(rows,finals[arm]);write(out/('metrics_'+arm+'.json'),dict(eligible=eligible[arm],**score))
        if eligible[arm]:leader.append(dict(arm=arm,**score))
    crop.old.prior.agent.cons.write_csv(out/'leaderboard.csv',leader)
    crop.old.prior.agent.cons.write_csv(out/'paired_diagnostic.csv',comparisons(rows,finals,eligible,args.n_boot))
    write(out/'predictions.json',finals)
    write(out/'previous_joint_q3_descriptive.json',data['previous_joint_q3'])
    status=dict(complete=all(eligible.values()),eligible=eligible,execution_errors_by_arm=errors,
        invalid_new_outputs=sum(x['prediction'] is None for x in calls.values()),saved_calls=len(calls),
        expected_calls=3*len(rows),diagnostic_only=True,fixture=fixture,
        tokenizer_gate='fixture_unrun' if fixture else 'preflight_pass_no_model_load' if model_gate_status[0]=='unrun' else model_gate_status[0])
    meta.update(**counts,complete=status['complete'],raw_new_lvlm_calls=len(calls),
        runner_wall_time_s=round(time.time()-started,4),gpu=crop.old.prior.agent.gpu_info())
    meta['software']={}
    for n in ['torch','transformers','tokenizers','accelerate','numpy','pillow','timm']:
        try:meta['software'][n]=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:meta['software'][n]=None
    write(meta_path,meta);write(out/'run_status.json',status);print(crop.old.json.dumps(status),flush=True)
    return 0 if status['complete'] else 1


def main(argv=None,**kwargs):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['dataset','sup-zip','unsup-zip','out-dir']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--model-path',type=Path);p.add_argument('--n-boot',type=int,default=2000)
    return run(p.parse_args(argv),**kwargs)


if __name__=='__main__':raise SystemExit(main())

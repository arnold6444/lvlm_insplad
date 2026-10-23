"""Four-image IG convergence/gradient diagnostic. No location GT or model selection."""
from collections import Counter
import argparse, importlib.metadata, io, json, time
from pathlib import Path
import numpy as np
import run_missing_ig_location_dev as old
import missing_ig_numerics as num

ROOT=Path(__file__).resolve().parents[1]
PREFIX='missing_ig_numerics';PROTOCOL='missing_ig_numerics_v1'
read,sha,digest=old.read,old.sha,old.digest

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)

def array(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);buf=io.BytesIO();np.save(buf,value,allow_pickle=False)
    b=buf.getvalue();tmp=path.with_suffix('.tmp');tmp.write_bytes(b);tmp.replace(path)
    return dict(path=path.name,sha256=sha(b))

def check_record(rec,request,out):
    if any(rec.get(k)!=v for k,v in request.items()):raise ValueError('changed diagnostic request')
    payload={k:v for k,v in rec.items() if k!='payload_sha256'}
    if digest(payload)!=rec['payload_sha256']:raise ValueError('changed record payload')
    if rec['error']:return
    for kind,info in rec.get('raw_maps',{}).items():
        b=old.old.checked_path(out,info['path']).read_bytes()
        if sha(b)!=info['sha256']:raise ValueError('corrupt diagnostic map')
        a=np.load(io.BytesIO(b),allow_pickle=False)
        if a.shape!=(518,518) or a.dtype!=np.dtype('float32' if kind=='fp32' else 'float64') or not np.isfinite(a).all():raise ValueError('invalid diagnostic map')
        if num.completeness(a,rec['target_logit'],rec['baseline_logit'])!=rec['diagnostics'][kind]:raise ValueError('changed diagnostic arithmetic')
    info=rec['target_feature'];b=old.old.checked_path(out,info['path']).read_bytes()
    if sha(b)!=info['sha256']:raise ValueError('corrupt target feature')
    old.old.check_vector(b,768)

def validate(root,fixture=False):
    manifest=read(root/'source_manifest.json')
    for n,h in manifest.items():
        if old.old.probe.sha_file(old.old.checked_path(root,n))!=h:raise ValueError('source changed: '+n)
    for n,h in old.runtime_hashes().items():
        if manifest.get(n)!=h:raise ValueError('unsealed runtime: '+n)
    requests=read(root/'inputs/requests.json');head=read(root/'inputs/head.json');seeds=read(root/'inputs/seeds.json')
    if not fixture and (old.old.probe.sha_file(root/'inputs/head.json')!='d9ff0610d0b74752a2dfc5b53f1cdc1db5e916e17b4260172f1c7abab8c6e7f1' or len(requests)!=4):raise ValueError('wrong native cohort/head')
    selection=read(root/'inputs/selection.json')
    if [r['core_id'] for r in requests]!=num.choose(selection['positive_cores']):raise ValueError('cohort selection changed')
    for r in requests:
        seed=seeds[r['core_id']+'/original/0'];b=(root/seed['feature_path']).read_bytes()
        if sha(b)!=seed['feature_sha256'] or seed['image_sha256']!=r['image_sha256']:raise ValueError('source feature changed')
        old.old.check_vector(b,len(head['coef']))
    if any('gt' in k.lower() or 'box' in k.lower() for r in requests for k in r):raise ValueError('diagnostic requests must exclude GT')
    return manifest,requests,head,seeds

def summarize(records,out,root):
    rows=[];stability=[]
    old32=read(root/'inputs/previous32.json')
    for key,rec in records.items():
        if rec['error'] or rec['method']=='derivatives':continue
        d=rec['diagnostics'];rows.append(dict(core_id=rec['core_id'],baseline=rec['baseline'],method=rec['method'],steps=rec['steps'],
            residual_fp32=d['fp32']['residual'],residual_fp64=d['fp64']['residual'],relative_fp64=d['fp64']['relative_residual'],
            warning_fp64=d['fp64']['warning'],accumulation_relative_l1=rec['accumulation_comparison']['relative_l1']))
        if rec['method']=='midpoint' and rec['steps']==32:
            prior=old32[rec['core_id']][rec['baseline']];raw=np.load(out/rec['raw_maps']['fp32']['path'],allow_pickle=False)
            patch=old.ig.patch_map(raw,rec['geometry'])[0]
            stability.append(dict(core_id=rec['core_id'],baseline=rec['baseline'],comparison='replay32_vs_PR69',
                **num.similarity(patch,np.load(root/prior['patch_path'],allow_pickle=False)),
                original_input_identical=rec['original_input_sha256']==prior['original_input_sha256'],
                baseline_input_identical=rec['baseline_input_sha256']==prior['baseline_input_sha256'],
                signed_fp32_npy_identical=rec['raw_maps']['fp32']['sha256']==prior['signed_map_sha256']))
    for core in sorted({r['core_id'] for r in records.values()}):
        for baseline in num.CONFIG['baselines']:
            for a,b in [('midpoint32','midpoint128'),('midpoint128','midpoint512'),('gausslegendre128','midpoint512')]:
                ra=records.get(core+'/'+baseline+'/'+a);rb=records.get(core+'/'+baseline+'/'+b)
                if not ra or not rb or ra['error'] or rb['error']:continue
                ma=np.load(out/ra['raw_maps']['fp64']['path'],allow_pickle=False);mb=np.load(out/rb['raw_maps']['fp64']['path'],allow_pickle=False)
                stability.append(dict(core_id=core,baseline=baseline,comparison=a+'_vs_'+b,**num.similarity(ma,mb)))
    fd=[]
    for rec in records.values():
        if rec['error'] or rec['method']!='derivatives':continue
        for backend,checks in rec['checks'].items():
            for point in checks:
                fd.append(dict(core_id=rec['core_id'],baseline=rec['baseline'],backend=backend,alpha=point['alpha'],
                    autograd=point['autograd'],epsilon_passes=sum(v['within_tolerance'] for v in point['finite_differences']),
                    min_relative_error=min(v['relative_error'] for v in point['finite_differences'])))
    write(out/'integration_summary.json',rows);write(out/'map_stability.json',stability);write(out/'derivative_summary.json',fd)
    old.old.prior.agent.cons.write_csv(out/'integration_summary.csv',rows)
    return dict(integration_records=len(rows),derivative_points=len(fd),
                warnings_fp64=sum(r['warning_fp64'] for r in rows),
                derivative_points_without_matching_epsilon=sum(r['epsilon_passes']==0 for r in fd))

def run(args,engine=None):
    start=time.time();root=args.capsule_dir;out=args.out_dir;fixture=engine is not None
    manifest,requests,head,seeds=validate(root,fixture)
    if not out.name.startswith(PREFIX) or out.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(out.resolve()) or args.sup_zip.resolve().is_relative_to(out.resolve()):raise ValueError('separate output required')
    software={}
    for name in ('torch','transformers','numpy','pillow','timm'):
        try:software[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:software[name]=None
    identity_data=dict(protocol=PROTOCOL,manifest_sha256=digest(manifest),config=num.CONFIG,software=software,fixture=fixture)
    identity=digest(identity_data)
    if (out/'run_metadata.json').exists():
        if read(out/'run_metadata.json')['identity']!=identity:raise ValueError('source/config/software changed; new output required')
    elif out.exists() and any(out.iterdir()):raise ValueError('unknown nonempty output')
    out.mkdir(parents=True,exist_ok=True)
    meta=dict(**identity_data,identity=identity,complete=False,backend='fixture' if fixture else 'hf',
        model_id=old.old.probe.MODEL,model_revision=old.old.probe.REVISION,
        training_calls=0,lvlm_calls=0,q1_calls=0,q2_calls=0,gold_used=False,
        claim_limit='Numerical diagnostic on four explored-dev images; no localization accuracy/model selection.')
    write(out/'run_metadata.json',meta);write(out/'run_status.json',dict(complete=False,phase='inputs',fixture=fixture))
    paths=old.old.prior.old.verified_images(args.sup_zip,[r['image'] for r in requests]+[s['image'] for s in seeds.values()],out/'_images')
    images={}
    for r in requests:
        p=paths[r['image']]
        if old.old.probe.sha_file(p)!=r['image_sha256']:raise ValueError('original image SHA changed')
        images[r['core_id']]=old.old.oc.original_image(p)
        if sha(images[r['core_id']].tobytes())!=seeds[r['core_id']+'/original/0']['pixel_sha256']:raise ValueError('original RGB changed')
    for name in ('requests.json','selection.json','head.json','seeds.json','previous32.json'):(out/name).write_bytes((root/'inputs'/name).read_bytes())
    holder=[engine];counts=Counter(new_conditions=0,resumed_conditions=0,new_canaries=0,resumed_canaries=0)
    def get():
        if holder[0] is None:holder[0]=num.NativeNumerics(head)
        return holder[0]
    canary_path=out/'canary.json';canary=read(canary_path) if canary_path.exists() else {};failure=''
    canary_seeds=[next(s for _,s in sorted(seeds.items()) if s['kind']==k) for k in ('original','blur')]
    if canary:
        if canary['identity']!=identity:raise ValueError('canary identity changed')
        if canary.get('passed'):
            if len(canary['checks'])!=2:raise ValueError('canary coverage changed')
            for check,s in zip(canary['checks'],canary_seeds):
                b=(out/check['path']).read_bytes()
                if sha(b)!=check['sha256'] or check['seed_id']!=s['id']:raise ValueError('canary corrupted')
                if not np.allclose(old.old.check_vector(b,len(head['coef'])),old.old.check_vector((root/s['feature_path']).read_bytes(),len(head['coef'])),atol=1e-5,rtol=1e-4):raise ValueError('canary mismatch')
            counts['resumed_canaries']=2
    if not canary.get('passed'):
        checks=[]
        try:
            for s in canary_seeds:
                counts['new_canaries']+=1
                im=old.old.oc.original_image(paths[s['image']])
                if old.old.probe.sha_file(paths[s['image']])!=s['image_sha256']:raise ValueError('canary original changed')
                if s['kind']!='original':im=old.old.oc.masked(im,s['cell'],s['kind'])
                if sha(im.tobytes())!=s['pixel_sha256']:raise ValueError('canary pixels changed')
                x=np.asarray(get().feature(im),dtype=np.float32);expected=old.old.check_vector((root/s['feature_path']).read_bytes(),len(head['coef']))
                if x.shape!=expected.shape or not np.isfinite(x).all() or not np.allclose(x,expected,atol=1e-5,rtol=1e-4):raise ValueError('native canary failed')
                info=array(out/'canary_features'/(s['kind']+'.npy'),x);info['path']='canary_features/'+info['path']
                checks.append(dict(**info,seed_id=s['id'],max_abs_error=float(np.abs(x-expected).max())))
            canary=dict(identity=identity,passed=True,error='',checks=checks)
        except Exception as exc:failure=type(exc).__name__+': '+str(exc);canary=dict(identity=identity,passed=False,error=failure,checks=checks)
        write(canary_path,canary)
    cache=read(out/'records.json') if (out/'records.json').exists() else {};records={}
    for r in requests:
        core=r['core_id'];expected=old.old.check_vector((root/seeds[core+'/original/0']['feature_path']).read_bytes(),len(head['coef']))
        for baseline in num.CONFIG['baselines']:
            for method,steps in [*num.CONFIG['conditions'],['derivatives',0]]:
                key=core+'/'+baseline+'/'+method+str(steps) if method!='derivatives' else core+'/'+baseline+'/derivatives'
                request=dict(identity=identity,core_id=core,image_sha256=r['image_sha256'],baseline=baseline,method=method,steps=steps)
                rec=cache.get(key)
                if rec:check_record(rec,request,out)
                if rec and not rec['error']:counts['resumed_conditions']+=1
                else:
                    if rec:write(out/'attempt_history'/(digest(rec)+'.json'),rec)
                    rec=dict(**request,error=failure)
                    if not failure:
                        try:
                            counts['new_conditions']+=1;result=get().execute(images[core],expected,baseline,method,steps)
                            feature=result.pop('target_feature')
                            if not np.allclose(feature,expected,atol=1e-5,rtol=1e-4):raise ValueError('condition target feature mismatch')
                            raw=result.pop('raw_maps',{})
                            rec.update(**result,raw_maps={})
                            for kind,a in raw.items():
                                info=array(out/'maps'/(digest(key)+'_'+kind+'.npy'),a);info['path']='maps/'+info['path'];rec['raw_maps'][kind]=info
                            info=array(out/'target_features'/(digest(key)+'.npy'),feature);info['path']='target_features/'+info['path'];rec['target_feature']=info
                        except Exception as exc:
                            rec['error']=type(exc).__name__+': '+str(exc)
                            if 'out of memory' in str(exc).lower():failure=rec['error']
                    rec['payload_sha256']=digest(rec)
                records[key]=rec;write(out/'records.json',{**cache,**records})
                write(out/'run_status.json',dict(complete=False,fixture=fixture,phase='diagnostic',saved=len(records),expected=len(requests)*10))
                print('DIAGNOSTIC',len(records),'/',len(requests)*10,key,'ERROR',rec['error'],flush=True)
    if set(cache)-set(records):raise ValueError('unexpected cached conditions')
    errors=sum(bool(r['error']) for r in records.values());summary=summarize(records,out,root)
    status=dict(complete=canary['passed'] and errors==0,fixture=fixture,canary_passed=canary['passed'],condition_errors=errors,
                saved=len(records),expected=len(requests)*10,numerical_validation='inspect diagnostics; execution complete is not numerical pass',**summary)
    stats=dict(holder[0].stats) if holder[0] is not None else dict(forward_passes=0,backward_passes=0,model_loads=0)
    meta.update(complete=status['complete'],**counts,**stats,runner_wall_time_s=round(time.time()-start,4),gpu=old.old.prior.agent.gpu_info())
    write(out/'run_metadata.json',meta);write(out/'run_status.json',status);print(json.dumps(status),flush=True)
    return 0 if status['complete'] else 1

def main(argv=None,**kwargs):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capsule-dir','sup-zip','out-dir'):p.add_argument('--'+name,type=Path,required=True)
    return run(p.parse_args(argv),**kwargs)

if __name__=='__main__':raise SystemExit(main())

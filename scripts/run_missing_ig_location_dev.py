"""Fixed-presence dev experiment: input IG location proposals and matched-count controls."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_missing_expert_direct_dev as old
import missing_ig_location as ig

read,write,sha,digest=old.read,old.write,old.sha,old.digest
PROTOCOL='missing_ig_location_dev_v1'
PREFIX='missing_ig_location_dev'
POLICIES=('cached_pr68','ig_blur_top1','ig_mean_top1','ig_consensus_top1','ig_consensus_multi',
          'occlusion_matched_k','center_matched_k','random_matched_k')


def runtime_hashes():
    # Same scripts/src boundary as the shared implementation; avoid resolving every
    # external torch/transformers module through the Windows filesystem.
    prefixes=tuple((str(ROOT/d).replace('\\','/').casefold()+'/') for d in ('scripts','src'))
    paths=set()
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename or not str(filename).endswith('.py'):continue
        lexical=os.path.abspath(filename).replace('\\','/').casefold()
        if not lexical.startswith(prefixes):continue
        path=Path(filename).resolve()
        if path.is_relative_to(ROOT) and path.relative_to(ROOT).parts[0] in ('scripts','src'):paths.add(path)
    return {p.relative_to(ROOT).as_posix():old.probe.sha_file(p) for p in sorted(paths)}


def validate(root,fixture=False):
    manifest=read(root/'source_manifest.json')
    for name,expected_sha in manifest.items():
        if old.probe.sha_file(old.checked_path(root,name))!=expected_sha:raise ValueError('capsule changed: '+name)
    for name,value in runtime_hashes().items():
        if manifest.get(name)!=value:raise ValueError('unsealed runtime code: '+name)
    data=read(root/'inputs/dataset.json');head=read(root/'inputs/head.json')
    expected=old.oc.validate(data,head,old.probe.sha_file(root/'inputs/head.json'),fixture)
    baseline=read(root/'inputs/baseline.json');tools=read(root/'inputs/tools.json');seeds=read(root/'inputs/seeds.json')
    cores={r['core_id'] for r in data['rows']}
    if set(baseline)!=cores:raise ValueError('baseline coverage changed')
    for core,p in baseline.items():
        if p['present'] is not None and type(p['present']) is not bool:raise ValueError('invalid baseline presence')
        if p['regions'] is not None and old.spatial.parse_json(json.dumps({'regions':p['regions']}),'regions')!=p['regions']:raise ValueError('invalid baseline regions')
    rows=[r for r in data['rows'] if r['asset_category']==old.probe.ASSET]
    if set(tools)!={r['core_id'] for r in rows}:raise ValueError('expert coverage changed')
    for row in rows:
        core=row['core_id'];seed=seeds[core+'/original/0'];content=(root/seed['feature_path']).read_bytes()
        if sha(content)!=seed['feature_sha256'] or seed['image_sha256']!=expected['target',row['image']]:raise ValueError('source original changed')
        x=old.check_vector(content,len(head['coef']));score=old.score_feature(head,x)['score'];t=tools[core]
        if t['error'] or not np.isclose(score,t['score'],rtol=1e-12,atol=1e-12) or t['positive']!=(score>=head['threshold']):raise ValueError('frozen expert score changed')
        if t['positive'] and baseline[core]['present'] is not True:raise ValueError('positive presence must be frozen')
    active=[r for r in rows if tools[r['core_id']]['positive']]
    if not fixture and (len(data['rows'])!=240 or len(rows)!=50 or len(active)!=26):raise ValueError('changed actual cohort')
    return manifest,data,head,expected,baseline,seeds,active


def assemble(baseline,records,old_maps):
    """No rows, GT, boxes or defect labels accepted. Presence never changes."""
    final={p:copy.deepcopy(baseline) for p in POLICIES}
    for core,rec in records.items():
        chosen={p:None for p in POLICIES[1:]}
        if not rec['error']:
            proposal=rec['proposals']
            for mode in ('blur_top1','mean_top1','consensus_top1','consensus_multi'):chosen['ig_'+mode]=proposal[mode]
            k=len(proposal['consensus_multi'] or [])
            chosen.update(ig.matched(core,k,old_maps[core]))
        for policy in POLICIES[1:]:
            pred=final[policy][core]
            pred.update(regions=chosen[policy],source=policy,location_abstained=chosen[policy] is None,error=rec['error'])
    assert all(final[p][c]['present']==baseline[c]['present'] for p in POLICIES for c in baseline)
    return final


def eval_rows(data,gt,expected):
    bygt={r['core_id']:r for r in gt}
    if len(bygt)!=len(gt):raise ValueError('duplicate evaluation label')
    rows=[]
    for row in data['rows']:
        label=bygt.get(row['core_id'])
        if label:
            if label['image']!=row['image'] or label['image_sha256']!=expected['target',row['image']]:raise ValueError('label identity mismatch')
            boxes=label['normalized_bboxes_xywh'];cells=label['regions_gt']
            if cells is not None and old.spatial.boxes_to_regions(boxes)!=cells:raise ValueError('label box centers changed')
            if cells is None and boxes:raise ValueError('unknown GT has boxes')
            row=dict(row,regions_gt=cells,label_basis='human_dev_box_centers' if cells is not None else 'human_dev_unlocatable')
        rows.append(row)
    return rows


def paired(rows,finals,n_boot):
    comparisons=[('cached_pr68','ig_consensus_top1'),('cached_pr68','ig_consensus_multi'),
        ('ig_consensus_top1','ig_consensus_multi'),('ig_blur_top1','ig_consensus_top1'),
        ('ig_mean_top1','ig_consensus_top1'),('occlusion_matched_k','ig_consensus_multi'),
        ('center_matched_k','ig_consensus_multi'),('random_matched_k','ig_consensus_multi')]
    rng=np.random.default_rng(42);known=[r for r in rows if r['anomaly_gt']=='anomalous' and r['regions_gt'] is not None];result=[]
    for a,b in comparisons:
        for metric in ('exact','any'):
            def val(r,p):
                pred=finals[p][r['core_id']]['regions'];truth=set(r['regions_gt'])
                return int(pred is not None and set(pred)==truth) if metric=='exact' else int(bool(set(pred or [])&truth))
            if not known:continue
            delta=np.array([val(r,b)-val(r,a) for r in known]);dist=delta[rng.integers(0,len(delta),size=(n_boot,len(delta)))].mean(1)
            result.append(dict(baseline=a,condition=b,metric=metric,n=len(known),delta=float(delta.mean()),
                ci_low=float(np.quantile(dist,.025)),ci_high=float(np.quantile(dist,.975)),n_boot=n_boot,
                scope='explored dev localization; presence frozen; no holdout/automatic selection'))
    return result


def save_array(path,x):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.tmp').open('wb') as stream:np.save(stream,x,allow_pickle=False)
    path.with_suffix('.tmp').replace(path)
    return old.probe.sha_file(path)


def run(args,*,attributor=None):
    started=time.time();root=args.capsule_dir;fixture=attributor is not None
    manifest,data,head,expected,baseline,seeds,active=validate(root,fixture)
    if not 1<=args.n_boot<=20000:raise ValueError('invalid bootstrap count')
    out=args.out_dir
    if not out.name.startswith(PREFIX) or out.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(out.resolve()) or args.sup_zip.resolve().is_relative_to(out.resolve()):raise ValueError('separate output required')
    software={}
    for name in ('torch','transformers','numpy','pillow','timm'):
        try:software[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:software[name]=None
    config=dict(protocol=PROTOCOL,manifest_sha256=digest(manifest),config=ig.CONFIG,software=software,fixture=fixture,
                n_boot=args.n_boot,model_id=old.probe.MODEL,model_revision=old.probe.REVISION)
    identity=digest(config);meta_path=out/'run_metadata.json'
    if meta_path.exists():
        if read(meta_path)['identity']!=identity:raise ValueError('changed source/input/software; use a new folder')
    elif out.exists() and any(out.iterdir()):raise ValueError('unknown nonempty output folder')
    canary_seeds=[next(v for k,v in sorted(seeds.items()) if v['kind']==kind) for kind in ('original','blur')]
    targets={r['core_id']:r for r in data['rows'] if r['asset_category']==old.probe.ASSET}
    required={r['image'] for r in active}|{targets[s['core_id']]['image'] for s in canary_seeds}
    paths=old.prior.old.verified_images(args.sup_zip,sorted(required),out/'_images')
    images={}
    for core,row in targets.items():
        if row['image'] not in paths:continue
        if old.probe.sha_file(paths[row['image']])!=expected['target',row['image']]:raise ValueError('original image changed')
        images[core]=old.oc.original_image(paths[row['image']])
        if sha(images[core].tobytes())!=seeds[core+'/original/0']['pixel_sha256']:raise ValueError('original RGB changed')
    out.mkdir(parents=True,exist_ok=True)
    meta=dict(**config,identity=identity,complete=False,backend='fixture' if fixture else 'hf',
              expert_routes=len(active),presence_frozen=True,lvlm_calls=0,training_calls=0,q1_calls=0,q2_calls=0,gold_used=False,
              claim_limit='IG evidence is not defect segmentation; custom component proposal; explored dev; no detection gain by design')
    write(meta_path,meta);write(out/'run_status.json',dict(complete=False,phase='canary',fixture=fixture))
    for name in ('dataset.json','baseline.json','head.json','evaluation_gt.json','seeds.json','tools.json','old_maps.json'):(out/name).write_bytes((root/'inputs'/name).read_bytes())
    holder=[attributor];failure=[''];counts=Counter(new_canaries=0,resumed_canaries=0,new_attributions=0,resumed_attributions=0,blocked_attributions=0)
    def get():
        if holder[0] is None:holder[0]=ig.NativeAttributor(head)
        return holder[0]
    def seed_image(seed):
        im=images[seed['core_id']]
        im=im if seed['kind']=='original' else old.oc.masked(im,seed['cell'],seed['kind'])
        if sha(im.tobytes())!=seed['pixel_sha256']:raise ValueError('canary mask pixels changed')
        return im
    canary_path=out/'canary.json';canary=read(canary_path) if canary_path.exists() else {}
    if canary and canary.get('identity')!=identity:raise ValueError('changed canary identity')
    if canary.get('passed'):
        if len(canary['checks'])!=2:raise ValueError('incomplete canary')
        for check,seed in zip(canary['checks'],canary_seeds):
            b=old.checked_path(out,check['path']).read_bytes()
            if check['seed_id']!=seed['id'] or sha(b)!=check['sha256']:raise ValueError('canary cache changed')
            x=old.check_vector(b,len(head['coef']));source=old.check_vector((root/seed['feature_path']).read_bytes(),len(head['coef']))
            if not np.allclose(x,source,atol=ig.CONFIG['canary_atol'],rtol=ig.CONFIG['canary_rtol']):raise ValueError('canary vector mismatch')
        counts['resumed_canaries']=2
    else:
        checks=[]
        try:
            for seed in canary_seeds:
                counts['new_canaries']+=1;x=np.asarray(get().feature(seed_image(seed)),dtype=np.float32)
                source=old.check_vector((root/seed['feature_path']).read_bytes(),len(head['coef']))
                if x.shape!=source.shape or not np.isfinite(x).all() or not np.allclose(x,source,atol=ig.CONFIG['canary_atol'],rtol=ig.CONFIG['canary_rtol']):raise ValueError('native feature canary failed')
                path=out/'canary_features'/(seed['kind']+'.npy');h=save_array(path,x)
                checks.append(dict(seed_id=seed['id'],path=path.relative_to(out).as_posix(),sha256=h,max_abs_error=float(np.max(np.abs(x-source)))))
            canary=dict(identity=identity,passed=True,error='',checks=checks)
        except Exception as exc:
            failure[0]=f'{type(exc).__name__}: {exc}';canary=dict(identity=identity,passed=False,error=failure[0],checks=checks)
        write(canary_path,canary)
    cache_path=out/'attributions.json';cache=read(cache_path) if cache_path.exists() else {};records={}
    for row in active:
        core=row['core_id'];image=images[core]
        request=dict(identity=identity,core_id=core,image_sha256=expected['target',row['image']],geometry=ig.geometry(image.size))
        rec=cache.get(core);raw={}
        if rec and any(rec.get(k)!=v for k,v in request.items()):raise ValueError('attribution request changed')
        if rec and not rec['error']:
            for kind,m in rec['raw_maps'].items():
                path=old.checked_path(out,m['path']);b=path.read_bytes()
                if sha(b)!=m['sha256']:raise ValueError('attribution map cache corrupted')
                raw[kind]=np.load(path,allow_pickle=False)
            p,maps=ig.proposals(raw,request['geometry'])
            if p!=rec['proposals']:raise ValueError('changed proposal cache')
            target=rec['target_feature'];b=old.checked_path(out,target['path']).read_bytes()
            if sha(b)!=target['sha256']:raise ValueError('target feature cache corrupted')
            native=old.check_vector(b,len(head['coef']));source=old.check_vector((root/seeds[core+'/original/0']['feature_path']).read_bytes(),len(head['coef']))
            if not np.allclose(native,source,atol=ig.CONFIG['canary_atol'],rtol=ig.CONFIG['canary_rtol']):raise ValueError('changed cached target feature')
            if not np.isclose(old.score_feature(head,native)['logit'],rec['target_logit'],atol=1e-10,rtol=1e-10):raise ValueError('changed cached target logit')
            for kind,a in raw.items():
                d=rec['diagnostics'][kind];total=float(a.astype(np.float64).sum())
                if not np.isclose(total,d['attribution_sum'],atol=1e-10,rtol=1e-10):raise ValueError('changed completeness sum')
                if not np.isclose(total-d['target_minus_baseline'],d['completeness_residual'],atol=1e-10,rtol=1e-10):raise ValueError('changed completeness residual')
            counts['resumed_attributions']+=1
        else:
            if rec:write(out/'attempt_history'/(digest(rec)+'.json'),rec)
            rec=dict(**request,error=failure[0],raw_maps={},proposals=None)
            start=time.time()
            if not failure[0]:
                try:
                    counts['new_attributions']+=1
                    source=old.check_vector((root/seeds[core+'/original/0']['feature_path']).read_bytes(),len(head['coef']))
                    result=get().attribute(image,source)
                    if result['geometry']!=request['geometry']:raise ValueError('native preprocessing geometry changed')
                    raw=result.pop('raw_maps');feature=result.pop('target_feature')
                    if not np.allclose(feature,source,atol=ig.CONFIG['canary_atol'],rtol=ig.CONFIG['canary_rtol']):raise ValueError('target feature mismatch')
                    p,maps=ig.proposals(raw,request['geometry'])
                    for kind,a in raw.items():
                        path=out/'signed_maps'/(core+'_'+kind+'.npy');h=save_array(path,a)
                        rec['raw_maps'][kind]=dict(path=path.relative_to(out).as_posix(),sha256=h)
                    path=out/'target_features'/(core+'.npy');h=save_array(path,feature.astype(np.float32))
                    rec.update(**result,proposals=p,target_feature=dict(path=path.relative_to(out).as_posix(),sha256=h))
                except Exception as exc:
                    rec['error']=f'{type(exc).__name__}: {exc}'
                    if 'out of memory' in rec['error'].lower():failure[0]=rec['error']
            else:counts['blocked_attributions']+=1
            rec['latency_s']=round(time.time()-start,4)
        records[core]=rec;write(cache_path,{**cache,**records})
        write(out/'run_status.json',dict(complete=False,phase='attribution',fixture=fixture,saved=len(records),expected=len(active)))
        print('IG saved',len(records),'/',len(active),dict(counts),flush=True)
    if set(cache)-set(records):raise ValueError('unexpected cached attribution')
    finals=assemble(baseline,records,read(root/'inputs/old_maps.json'));write(out/'predictions.json',finals)
    gt=read(root/'inputs/evaluation_gt.json');rows=eval_rows(data,gt,expected)
    if not fixture and (len(gt)!=25 or sum(r['regions_gt'] is not None for r in gt)!=24):raise ValueError('changed location support')
    errors=sum(bool(r['error']) for r in records.values());eligible=canary['passed'] and not errors
    metrics={};groups=[];leader=[]
    for policy,pred in finals.items():
        m=old.metric(rows,pred);metrics[policy]=m
        leader.append(dict(policy=policy,eligible=eligible if policy!='cached_pr68' else True,**m['presence'],
                           location_exact=m['location']['anomaly_exact_set_accuracy'],location_any=m['location']['any_location_hits'],location_abstentions=m['location_abstentions']))
        for field in ('asset_category','defect_gt'):
            for value in sorted({r[field] for r in rows}):groups.append(dict(policy=policy,group=field,value=value,**old.metric([r for r in rows if r[field]==value],pred)))
    write(out/'metrics.json',metrics);write(out/'metrics_by_group.json',groups)
    write(out/'paired.json',paired(rows,finals,args.n_boot) if eligible else [])
    old.prior.agent.cons.write_csv(out/'leaderboard.csv',leader)
    macro={}
    for policy in POLICIES:
        assets=[r['presence']['balanced_accuracy'] for r in groups if r['policy']==policy and r['group']=='asset_category']
        supported=[v for v in assets if v is not None]
        macro[policy]=dict(asset_macro_presence_balanced_accuracy=float(np.mean(supported)) if supported else None,
                           assets_with_both_classes=len(supported),assets_total=len(assets))
    write(out/'macro_metrics.json',macro)
    warnings=sum(bool(d['completeness_warning']) for r in records.values() if not r['error'] for d in r['diagnostics'].values())
    stats=dict(holder[0].stats) if holder[0] is not None else dict(forward_passes=0,backward_passes=0,model_loads=0)
    status=dict(complete=bool(eligible),fixture=fixture,attribution_errors=errors,canary_passed=canary['passed'],
                saved=len(records),expected=len(active),presence_frozen=True,completeness_warnings=warnings)
    meta.update(**counts,**stats,complete=status['complete'],runner_wall_time_s=round(time.time()-started,4),gpu=old.prior.agent.gpu_info())
    write(meta_path,meta);write(out/'run_status.json',status);print(json.dumps(status),flush=True)
    return 0 if status['complete'] else 1


def main(argv=None,**kwargs):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capsule-dir','sup-zip','out-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--n-boot',type=int,default=2000)
    return run(p.parse_args(argv),**kwargs)


if __name__=='__main__':raise SystemExit(main())

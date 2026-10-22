"""Frozen missing expert + direct top-one location, explored dev only; no LVLM calls."""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
import run_q3_missing_occlusion_dev as oc

prior, probe, spatial = oc.prior, oc.prior.probe, oc.prior.spatial
read, write, digest, sha = oc.read, oc.write, oc.digest, oc.sha
PROTOCOL = 'missing_expert_direct_dev_v1'
PREFIX = 'missing_expert_direct_dev'
POLICIES = ('cached_lvlm', 'same_route_top1', 'all_positive_top1',
            'same_route_center', 'all_positive_center', 'same_route_random', 'all_positive_random')
CONFIG = {'occlusion': oc.CONFIG, 'top_k': 1, 'center': 5, 'random_namespace': 'matched-random42/',
          'canary_atol': 1e-5, 'canary_rtol': 1e-4, 'threshold_selection': False}


def pixel_sha(image):
    return sha(image.tobytes())


def check_vector(content, dim):
    import io
    x = np.load(io.BytesIO(content), allow_pickle=False)
    if x.dtype != np.float32 or x.shape != (dim,) or not np.isfinite(x).all() or not np.isclose(np.linalg.norm(x), 1, atol=1e-5):
        raise ValueError('invalid normalized float32 feature')
    return x


def checked_path(root, name):
    p = root/name
    if not p.resolve().is_relative_to(root.resolve()):
        raise ValueError('unsafe capsule path')
    return p


def verify_capsule(root):
    manifest = read(root/'source_manifest.json')
    for name, expected in manifest.items():
        if probe.sha_file(checked_path(root, name)) != expected:
            raise ValueError('capsule changed: '+name)
    # Every imported local runtime module is sealed, including this runner.
    for name, expected in prior.shared.code_hashes().items():
        if manifest.get(name) != expected:
            raise ValueError('unsealed runtime code: '+name)
    return manifest


def request(core, kind, cell=0):
    return core+'/'+kind+'/'+str(cell)


def score_feature(head, x):
    return {'score': float(probe.predict(head, x[None, :])[0]), 'logit': oc.logit(head, x)}


def choose_predictions(rows, baseline, tools, maps):
    """Inference accepts no GT. All-positive routing ignores initial answer/validity."""
    out = {p: {} for p in POLICIES}
    for row in rows:
        core = row['core_id']; first = baseline[core]
        original = {'present': None if first is None else bool(first), 'regions': first,
                    'source': 'cached_lvlm', 'location_abstained': False, 'error': ''}
        tool = tools.get(core)
        for policy in POLICIES:
            pred = dict(original)
            active = (policy != 'cached_lvlm' and row['asset_category'] == probe.ASSET
                      and (policy.startswith('all_positive') or first == []))
            if active and tool and tool['error']:
                pred.update(present=None, regions=None, error=tool['error'], source='expert_error')
            elif active and tool and tool['positive']:
                record = maps[core]; cells = record['proposal']['regions'] if record['proposal'] else []
                selected = None
                if cells and not record['error']:
                    selected = ([cells[0]] if policy.endswith('top1') else [5] if policy.endswith('center') else
                                [1+int(sha((CONFIG['random_namespace']+core).encode()), 16) % 9])
                pred.update(present=True, regions=selected, source='frozen_missing_expert',
                            location_abstained=selected is None, error=record['error'])
            out[policy][core] = pred
    return out


def metric(rows, predictions):
    """Presence is scored separately from a positive location abstention."""
    counts = Counter(tp=0, fp=0, tn=0, fn=0, invalid_normal=0, invalid_anomalous=0)
    for r in rows:
        positive = r['anomaly_gt'] == 'anomalous'; value = predictions[r['core_id']]['present']
        if value is None:
            counts['invalid_anomalous' if positive else 'invalid_normal'] += 1
            counts['fn' if positive else 'fp'] += 1
        else:
            counts['tp' if positive and value else 'fn' if positive else 'fp' if value else 'tn'] += 1
    location = spatial.score(rows, {k: v['regions'] for k, v in predictions.items()})
    pos = sum(r['anomaly_gt'] == 'anomalous' for r in rows); normal = len(rows)-pos
    recall, specificity = spatial.ratio(counts['tp'], pos), spatial.ratio(counts['tn'], normal)
    known = [r for r in rows if r['anomaly_gt'] == 'anomalous' and r['regions_gt'] is not None]
    location['any_location_hits'] = sum(bool(set(r['regions_gt']) & set(predictions[r['core_id']]['regions'] or [])) for r in known)
    return {'presence': dict(images=len(rows), anomalous_images=pos, normal_images=normal, **counts,
                            recall=recall, specificity=specificity, normal_fpr=spatial.ratio(counts['fp'], normal),
                            precision=spatial.ratio(counts['tp'],counts['tp']+counts['fp']),
                            f1=spatial.ratio(2*counts['tp'],2*counts['tp']+counts['fp']+counts['fn']),
                            invalid_rate=spatial.ratio(counts['invalid_normal']+counts['invalid_anomalous'],len(rows)),
                            balanced_accuracy=(recall+specificity)/2 if pos and normal else None),
            'location': location,
            'location_abstentions': sum(predictions[r['core_id']]['location_abstained'] for r in rows),
            'note': 'location scorer invalid counts include abstentions; use separate presence counts for detection'}


def paired(rows, finals, n_boot):
    comparisons = [('cached_lvlm','same_route_top1'), ('cached_lvlm','all_positive_top1'),
                   ('same_route_top1','all_positive_top1'), ('all_positive_center','all_positive_top1'),
                   ('all_positive_random','all_positive_top1')]
    rng = np.random.default_rng(42); result = []
    for a, b in comparisons:
        for name, selected in [('presence_recall',[r for r in rows if r['anomaly_gt']=='anomalous']),
                               ('normal_fpr',[r for r in rows if r['anomaly_gt']=='normal']),
                               ('location_exact',[r for r in rows if r['anomaly_gt']=='anomalous' and r['regions_gt'] is not None]),
                               ('location_any',[r for r in rows if r['anomaly_gt']=='anomalous' and r['regions_gt'] is not None])]:
            if not selected: continue
            def value(r, policy):
                p = finals[policy][r['core_id']]
                if name=='presence_recall': return int(p['present'] is True)
                if name=='normal_fpr': return int(p['present'] is not False)
                if name=='location_exact': return int(p['regions'] is not None and set(p['regions'])==set(r['regions_gt']))
                return int(bool(set(p['regions'] or []) & set(r['regions_gt'])))
            delta = np.array([value(r,b)-value(r,a) for r in selected])
            dist = delta[rng.integers(0,len(delta),size=(n_boot,len(delta)))].mean(1)
            result.append(dict(baseline=a,condition=b,metric=name,n=len(delta),delta=float(delta.mean()),
                               ci_low=float(np.quantile(dist,.025)),ci_high=float(np.quantile(dist,.975)),
                               n_boot=n_boot,scope='explored dev; descriptive cached baseline; no selection/holdout claim'))
    return result


def run(args, *, extractor=None):
    started = time.time(); root = args.capsule_dir; fixture = extractor is not None
    manifest = verify_capsule(root)
    data = read(root/'inputs/dataset.json'); head_path = root/'inputs/frozen_detector.json'; head = read(head_path)
    expected = oc.validate(data, head, probe.sha_file(head_path), fixture)
    seeds = read(root/'inputs/feature_seeds.json')
    if not 1 <= args.n_boot <= 20000: raise ValueError('invalid bootstrap count')
    out = args.out_dir
    if (not out.name.startswith(PREFIX) or out.resolve().is_relative_to(root.resolve()) or
            root.resolve().is_relative_to(out.resolve()) or args.sup_zip.resolve().is_relative_to(out.resolve())):
        raise ValueError('separate output directory required')
    software = {}
    for name in ('torch','transformers','numpy','pillow','timm'):
        try: software[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: software[name] = None
    config = dict(protocol=PROTOCOL,manifest_sha256=digest(manifest),fixture=fixture,config=CONFIG,
                  n_boot=args.n_boot,model_id=probe.MODEL,model_revision=probe.REVISION,software=software)
    identity = digest(config); meta_path = out/'run_metadata.json'
    if meta_path.exists():
        if read(meta_path)['identity'] != identity: raise ValueError('changed source/input/software; use new folder')
    elif out.exists() and any(out.iterdir()): raise ValueError('unknown nonempty output directory')
    rows = data['rows']; targets_rows = [r for r in rows if r['asset_category']==probe.ASSET]
    paths = prior.old.verified_images(args.sup_zip, [r['image'] for r in targets_rows], out/'_images')
    images = {}
    for r in targets_rows:
        if probe.sha_file(paths[r['image']]) != expected['target',r['image']]: raise ValueError('changed original image')
        images[r['core_id']] = oc.original_image(paths[r['image']])
    dim = len(head['coef'])
    def image_for(core, kind, cell):
        return images[core] if kind=='original' else oc.masked(images[core],cell,kind)
    # Validate every reusable vector, score, original and mask operation before neural work.
    for key, rec in seeds.items():
        if key != request(rec['core_id'],rec['kind'],rec['cell']): raise ValueError('seed key mismatch')
        im = image_for(rec['core_id'],rec['kind'],rec['cell'])
        if list(im.size)!=rec['size'] or pixel_sha(im)!=rec['pixel_sha256']: raise ValueError('changed source mask pixels')
        row = next(r for r in targets_rows if r['core_id']==rec['core_id'])
        if expected['target',row['image']]!=rec['image_sha256']: raise ValueError('seed image mismatch')
        content = checked_path(root,rec['feature_path']).read_bytes()
        if sha(content)!=rec['feature_sha256']: raise ValueError('seed feature checksum mismatch')
        x = check_vector(content,dim)
        if any(not np.isclose(score_feature(head,x)[k],rec[k],rtol=1e-12,atol=1e-12) for k in ('score','logit')):
            raise ValueError('seed score mismatch')
    out.mkdir(parents=True,exist_ok=True)
    meta = dict(**config, identity=identity, complete=False, backend='fixture' if fixture else 'hf',
                lvlm_calls=0,q1_calls=0,q2_calls=0,gold_used=False,training_calls=0,
                claim_limit='explored dev; supervised frozen expert; cached historical LVLM; not final online Agent score')
    write(meta_path,meta); write(out/'run_status.json',{'complete':False,'phase':'features'})
    for name in ('dataset.json','frozen_detector.json','evaluation_gt.json','feature_seeds.json'):
        (out/name).write_bytes((root/'inputs'/name).read_bytes())
    cache_path = out/'features.json'; cache = read(cache_path) if cache_path.exists() else {}
    counts = Counter({k:0 for k in ('new_features','new_canary_features','reused_source_features',
                                   'resumed_features','resumed_canaries','blocked_features')})
    used = set(); holder = [extractor]; failure = ['']
    def extract(im):
        if holder[0] is None: holder[0] = probe.DinoFeatures()
        return oc.vector(holder[0](im),dim)
    def obtain(core,kind,cell=0):
        key = request(core,kind,cell); used.add(key); im = image_for(core,kind,cell)
        req = dict(core_id=core,kind=kind,cell=cell,pixel_sha256=pixel_sha(im),size=list(im.size))
        rec = cache.get(key); path = out/'feature_cache'/(digest(key)+'.npy')
        if rec and any(rec.get(k)!=v for k,v in dict(req,identity=identity).items()): raise ValueError('changed cached feature request')
        if rec and not rec['error']:
            if probe.sha_file(path)!=rec['feature_sha256']: raise ValueError('corrupt output feature')
            x = check_vector(path.read_bytes(),dim)
            if any(not np.isclose(score_feature(head,x)[k],rec[k],rtol=1e-12,atol=1e-12) for k in ('score','logit')): raise ValueError('corrupt output score')
            counts['resumed_features'] += 1
            return rec
        if rec: write(out/'attempt_history'/(digest(rec)+'.json'),rec)
        rec = dict(req,identity=identity,error='',feature_sha256=None,score=None,logit=None,
                   source='sealed_prior_cache' if key in seeds else 'new_dino')
        if key in seeds:
            x = check_vector((root/seeds[key]['feature_path']).read_bytes(),dim); counts['reused_source_features']+=1
            rec['seed_provenance']=seeds[key]
        elif failure[0]:
            rec['error']=failure[0]; counts['blocked_features']+=1; x=None
        else:
            feature_started=time.time()
            try:
                counts['new_features']+=1; x=extract(im)
            except Exception as exc:
                rec['error']=failure[0]=f'{type(exc).__name__}: {exc}'; x=None
            rec['neural_latency_s']=round(time.time()-feature_started,6)
        if x is not None:
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.with_suffix('.tmp').open('wb') as stream: np.save(stream,x,allow_pickle=False)
            path.with_suffix('.tmp').replace(path)
            rec.update(**score_feature(head,x),feature_sha256=probe.sha_file(path))
        cache[key]=rec; write(cache_path,cache)
        return rec
    # Two GT-blind runtime canaries: one cached original and one cached mask.
    # Resume verifies receipt and raw vector hashes; it does not reload the model.
    canary_path=out/'canary.json'; canary=read(canary_path) if canary_path.exists() else {}
    if canary and canary.get('identity')!=identity: raise ValueError('changed canary identity')
    if canary.get('passed'):
        if len(canary['checks'])!=2: raise ValueError('incomplete canary receipt')
        if {seeds[r['seed_id']]['kind'] for r in canary['checks']}!={'original','blur'}: raise ValueError('wrong canary kinds')
        for rec in canary['checks']:
            path=checked_path(out,rec['path'])
            if probe.sha_file(path)!=rec['sha256']: raise ValueError('corrupt canary vector')
            seed=seeds[rec['seed_id']]
            x=check_vector(path.read_bytes(),dim); old=check_vector((root/seed['feature_path']).read_bytes(),dim)
            if not np.allclose(x,old,atol=CONFIG['canary_atol'],rtol=CONFIG['canary_rtol']): raise ValueError('changed resumed canary')
        counts['resumed_canaries']=len(canary['checks'])
    else:
        checks=[]
        try:
            for kind in ('original','blur'):
                rec=next(v for k,v in sorted(seeds.items()) if v['kind']==kind)
                canary_started=time.time()
                counts['new_canary_features']+=1; x=extract(image_for(rec['core_id'],rec['kind'],rec['cell']))
                old=check_vector((root/rec['feature_path']).read_bytes(),dim)
                if not np.allclose(x,old,atol=CONFIG['canary_atol'],rtol=CONFIG['canary_rtol']): raise ValueError('native DINO feature canary mismatch; cannot mix caches')
                path=out/'canary_features'/(kind+'.npy'); path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('wb') as stream: np.save(stream,x,allow_pickle=False)
                checks.append(dict(seed_id=rec['id'],path=path.relative_to(out).as_posix(),sha256=probe.sha_file(path),
                                   max_abs_error=float(np.max(np.abs(x-old))),neural_latency_s=round(time.time()-canary_started,6)))
            canary=dict(identity=identity,passed=True,checks=checks,error='')
        except Exception as exc:
            failure[0]=f'{type(exc).__name__}: {exc}'
            canary=dict(identity=identity,passed=False,checks=checks,error=failure[0])
        write(canary_path,canary)
    tools={}; maps={}
    for row in targets_rows:
        core=row['core_id']; rec=obtain(core,'original')
        tool=dict(score=rec['score'],positive=None if rec['error'] else rec['score']>=head['threshold'],error=rec['error'])
        tools[core]=tool
        if tool['positive']:
            occluded={kind:[] for kind in oc.CONFIG['replacements']}; errors=[]
            for kind in occluded:
                for cell in range(1,10):
                    masked=obtain(core,kind,cell); occluded[kind].append(masked['logit'])
                    if masked['error']: errors.append(masked['error'])
            maps[core]=dict(original_logit=rec['logit'],occluded_logits=occluded,error='; '.join(sorted(set(errors))),
                            proposal=None if errors else oc.candidates(rec['logit'],occluded))
        write(out/'tools.json',tools); write(out/'occlusion_maps.json',maps)
        print('saved',len(tools),'/',len(targets_rows),'features',dict(counts),flush=True)
    if set(cache)-used: raise ValueError('unexpected output feature cache entries')
    # Label-free payload only is passed to prediction selection.
    finals=choose_predictions([{k:r[k] for k in ('core_id','asset_category')} for r in rows],data['cached_direct'],tools,maps)
    write(out/'predictions.json',finals)
    labels=read(root/'inputs/evaluation_gt.json'); bygt={r['core_id']:r for r in labels}
    if len(bygt)!=len(labels): raise ValueError('duplicate evaluation label')
    if not fixture and (set(bygt)!={r['core_id'] for r in rows if r['defect_gt']=='missing-cap'} or
                        len(rows)!=240 or len(targets_rows)!=50 or sum(r['regions_gt'] is not None for r in labels)!=24):
        raise ValueError('changed actual dev evaluation support')
    eval_rows=[]
    for r in rows:
        gt=bygt.get(r['core_id'])
        if gt:
            if gt['image']!=r['image'] or gt['image_sha256']!=expected['target',r['image']]: raise ValueError('evaluation label identity mismatch')
            cells=gt['regions_gt']; boxes=gt['normalized_bboxes_xywh']
            if cells is not None and spatial.boxes_to_regions(boxes)!=cells: raise ValueError('label boxes/cells disagree')
            if cells is None and boxes: raise ValueError('unknown label has boxes')
            r=dict(r,regions_gt=cells,label_basis='human_dev_box_centers' if cells is not None else 'human_dev_unlocatable')
        eval_rows.append(r)
    errors=sum(bool(rec['error']) for rec in cache.values()); eligible=not errors and canary['passed']
    metrics={}; leaderboard=[]; groups=[]
    for policy,pred in finals.items():
        metrics[policy]=metric(eval_rows,pred)
        leaderboard.append(dict(policy=policy,eligible=True if policy=='cached_lvlm' else eligible,
                                **metrics[policy]['presence'],location_exact=metrics[policy]['location']['anomaly_exact_set_accuracy'],
                                location_abstentions=metrics[policy]['location_abstentions']))
        for field in ('asset_category','defect_gt'):
            for value in sorted({r[field] for r in eval_rows}): groups.append(dict(policy=policy,group=field,value=value,
                **metric([r for r in eval_rows if r[field]==value],pred)))
    write(out/'metrics.json',metrics); write(out/'metrics_by_group.json',groups)
    macro={}
    for policy in POLICIES:
        assets=[r for r in groups if r['policy']==policy and r['group']=='asset_category']
        supported=[r['presence']['balanced_accuracy'] for r in assets if r['presence']['balanced_accuracy'] is not None]
        macro[policy]=dict(asset_macro_presence_balanced_accuracy=float(np.mean(supported)) if supported else None,
                           assets_with_both_classes=len(supported),assets_total=len(assets))
    write(out/'macro_metrics.json',macro)
    write(out/'paired.json',paired(eval_rows,finals,args.n_boot) if eligible else [])
    prior.agent.cons.write_csv(out/'leaderboard.csv',leaderboard)
    status=dict(complete=bool(eligible),fixture=fixture,feature_errors=errors,canary_passed=canary['passed'],
                images=len(rows),expert_images=len(tools),expert_positive=sum(t['positive'] is True for t in tools.values()),
                lvlm_calls=0,new_dino_features=counts['new_features'],new_canary_features=counts['new_canary_features'])
    meta.update(**counts,complete=status['complete'],features=len(cache),runner_wall_time_s=round(time.time()-started,4),
                gpu=prior.agent.gpu_info(),original_seed_features=sum(v['kind']=='original' for v in seeds.values()),
                masked_seed_features=sum(v['kind']!='original' for v in seeds.values()))
    write(meta_path,meta); write(out/'run_status.json',status)
    print(json.dumps(status),flush=True)
    return 0 if status['complete'] else 1


def main(argv=None,**kwargs):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('capsule-dir','sup-zip','out-dir'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--n-boot',type=int,default=2000)
    return run(p.parse_args(argv),**kwargs)


if __name__=='__main__': raise SystemExit(main())

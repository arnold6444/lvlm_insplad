"""Frozen missing-classifier occlusion proposals + matched crop Q3 review, dev only."""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import hashlib
import io
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'),str(ROOT/'src')]
import run_selective_missing_expert_agent_dev as prior

PROTOCOL = 'q3_missing_occlusion_dev_v1'
PREFIX = 'q3_missing_occlusion_dev'
POLICIES = ('cached_direct','cached_cue_review','fixed_crop','occlusion_crop')
NEW = ('fixed_crop','occlusion_crop')
CONFIG = {'grid':3,'replacements':['blur','local_mean'],'blur_radius_divisor':6,
          'max_candidates':2,'positive_agreement':True,'fixed_cells':[5,2],
          'rank':'minimum L2-normalized positive logit drop across both replacements',
          'image_tile_budgets':[3,3,2],'max_new_tokens':80,'do_sample':False}
sha = lambda b:hashlib.sha256(b).hexdigest()
digest = prior.digest
read = lambda p:json.loads(p.read_text(encoding='utf-8'))
write = prior.old.atomic_json


def box(size,cell):
    if type(cell) is not int or not 1<=cell<=9 or min(size)<3:raise ValueError('invalid grid geometry')
    w,h=size;r,c=divmod(cell-1,3)
    return (c*w//3,r*h//3,(c+1)*w//3,(r+1)*h//3)


def masked(image,cell,kind):
    im=image.copy();bounds=box(im.size,cell);patch=im.crop(bounds)
    if kind=='blur':replacement=patch.filter(ImageFilter.GaussianBlur(max(2,min(patch.size)/6)))
    elif kind=='local_mean':
        color=tuple(np.asarray(patch,dtype=np.float64).mean((0,1)).round().astype(int).tolist())
        replacement=Image.new('RGB',patch.size,color)
    else:raise ValueError('unknown replacement')
    im.paste(replacement,bounds[:2]);return im


def png(image):
    stream=io.BytesIO();image.save(stream,format='PNG');return stream.getvalue()


def original_image(path):
    with Image.open(path) as im:
        if im.getexif().get(274,1)!=1:raise ValueError('nonidentity EXIF orientation needs a separate coordinate protocol')
        return im.convert('RGB')


def vector(x,dim):
    x=np.asarray(x,dtype=np.float32)
    if x.shape!=(dim,) or not np.isfinite(x).all() or np.linalg.norm(x)<=0:raise ValueError('invalid feature')
    return x/np.linalg.norm(x)


def logit(head,x):
    z=((np.asarray(x,dtype=np.float64)-head['mean'])/head['scale'])@np.asarray(head['coef'])+head['intercept']
    if not np.isfinite(z):raise ValueError('nonfinite logit')
    return float(z)


def candidates(original,occluded):
    drops=np.asarray([[original-occluded[k][i] for i in range(9)] for k in CONFIG['replacements']],float)
    if drops.shape!=(2,9) or not np.isfinite(drops).all():raise ValueError('invalid occlusion scores')
    rank=np.minimum(*(row/max(np.linalg.norm(row),1e-12) for row in drops))
    eligible=[i+1 for i in range(9) if (drops[:,i]>0).all()]
    selected=sorted(eligible,key=lambda c:(-rank[c-1],c))[:CONFIG['max_candidates']]
    return {'regions':selected,'logit_drops':drops.tolist(),'agreement_rank':rank.tolist(),
            'abstained':not selected,'is_verified_location':False}


def sheet(image,cells):
    if not 1<=len(cells)<=2 or len(set(cells))!=len(cells):raise ValueError('invalid crop cells')
    canvas=Image.new('RGB',(512*len(cells),548),'white');draw=ImageDraw.Draw(canvas)
    for i,c in enumerate(cells):
        crop=image.crop(box(image.size,c))
        crop=ImageOps.contain(crop,(512,512),Image.Resampling.BICUBIC)
        canvas.paste(crop,(512*i+(512-crop.width)//2,36+(512-crop.height)//2))
        draw.text((512*i+12,10),f'Original target cell {c} (row {(c-1)//3+1}, col {(c-1)%3+1})',fill='black')
    return canvas


def prompt(first,cells):
    # Deliberately identical text for both new arms, apart from original crop coordinates.
    return prior.spatial.prompt('regions',True)+(
        '\nImage 3 contains enlarged, UNMODIFIED crops from Image 2, labeled with their ORIGINAL '
        'target grid cell numbers '+json.dumps(cells)+'. It is not another scene or another normal reference. '
        'A frozen supervised glass-insulator classifier flags a POSSIBLE missing/broken component. '
        'The crop selector is experimental and may highlight background or normal parts. '
        'Neither the classifier nor the crop labels are proof of a defect. '
        'Your earlier answer to THIS Q3 question was '+json.dumps({'regions':first})+'. '
        'Examine the full TARGET and the crop details. Report the original target cells of any visible defects; '
        'you may select cells OUTSIDE these crops, or an empty list. Do not invent a missing part because a '
        'crop is highlighted; perspective and occlusion alone do not prove loss. '
        'Return only JSON {"regions":[1,9]} or {"regions":[]}, without explanation.')


def validate(data,head,head_sha,fixture=False):
    if data.get('protocol')!=PROTOCOL or not data.get('rows'):raise ValueError('expected Q3 dev capsule')
    if head_sha!=data['detector_sha256'] or (not fixture and head_sha!=prior.DETECTOR_SHA256):raise ValueError('changed frozen head')
    if head['model_id']!=prior.probe.MODEL or head['model_revision']!=prior.probe.REVISION or head['feature']!=prior.probe.FEATURE:
        raise ValueError('changed feature contract')
    if head['gold_scored'] is not False or head['selection_uses']!=['fit','calibration']:raise ValueError('invalid head provenance')
    dim=len(head['coef'])
    if dim<2 or any(len(head[k])!=dim for k in ['mean','scale']) or min(head['scale'])<=0:raise ValueError('invalid head dimensions')
    if not np.isfinite([*head['coef'],*head['mean'],*head['scale'],head['intercept'],head['threshold']]).all():raise ValueError('invalid head')
    if len({r['core_id'] for r in data['rows']})!=len(data['rows']):raise ValueError('duplicate cores')
    for r in data['rows']:
        if r['protocol']!=prior.PROTOCOL or r['question_type']!=prior.agent.Q3 or r['label_basis']!='dev_image_label_only':raise ValueError('gold or non-dev row forbidden')
        if r['regions_gt']!=([] if r['anomaly_gt']=='normal' else None):raise ValueError('unexpected spatial labels')
        for p in ['cached_direct','cached_cue_review']:
            value=data[p][r['core_id']]
            if value is not None and prior.spatial.parse_json(json.dumps({'regions':value}),'regions')!=value:raise ValueError('invalid cached location')
        if not data['references'][r['image']].startswith(r['asset_category']+'/train/'):raise ValueError('wrong asset reference')
        if not any('/train/'+n+'/' in data['references'][r['image']] for n in ['good','normal']):raise ValueError('non-normal reference')
    expected={(x['role'],x['member']):x['sha256'] for x in data['image_manifest']}
    if len(expected)!=len(data['image_manifest']):raise ValueError('duplicate image identity')
    for r in data['rows']:
        if ('target',r['image']) not in expected or ('reference',data['references'][r['image']]) not in expected:raise ValueError('missing image SHA')
    return expected


def routed(row,data):
    # No anomaly_gt, defect_gt, Q1 or Q2 read here.
    tool=data['experts'].get(row['core_id'])
    return row['asset_category']==prior.probe.ASSET and data['cached_direct'][row['core_id']]==[] and bool(tool and not tool['error'] and tool['positive'] is True)


def bootstrap(rows,finals,eligible,n_boot):
    raw=prior.q3_bootstrap(rows,{'lvlm_only':finals['cached_direct'],'expert_q2_override':finals['cached_cue_review'],
        'selective_agent':finals['occlusion_crop'],'matched_reask':finals['fixed_crop']},
        {'lvlm_only':eligible['cached_direct'],'expert_q2_override':eligible['cached_cue_review'],
        'selective_agent':eligible['occlusion_crop'],'matched_reask':eligible['fixed_crop']},n_boot)
    names={'lvlm_only':'cached_direct','expert_q2_override':'cached_cue_review','selective_agent':'occlusion_crop','matched_reask':'fixed_crop'}
    for r in raw:
        r['baseline']=names[r['baseline']];r['condition']=names[r['condition']]
        r['comparison_note']='matched new context' if r['baseline']=='fixed_crop' else 'descriptive cached comparator; context/budget differs'
    return raw


def run(args,*,extractor=None,loader=None,infer=None):
    started=time.time();supplied=(extractor is not None,loader is not None,infer is not None)
    if any(supplied) and not all(supplied):raise ValueError('all three fixture callbacks required')
    fixture=all(supplied);data=read(args.dataset);head=read(args.detector)
    expected=validate(data,head,prior.probe.sha_file(args.detector),fixture)
    if not 1<=args.n_boot<=20000:raise ValueError('invalid bootstrap count')
    out=args.out_dir
    inputs=[args.dataset,args.detector,args.sup_zip,args.unsup_zip]
    if not out.name.startswith(PREFIX) or out.resolve().is_relative_to(args.dataset.parent.resolve()) or any(p.resolve().is_relative_to(out.resolve()) for p in inputs):raise ValueError('separate output required')
    if not fixture and (not args.model_path or args.model_path.name!=prior.MODEL_REVISION or not args.model_path.is_dir()):raise ValueError('pinned native model snapshot required')
    config={'protocol':PROTOCOL,'input_sha256':prior.probe.sha_file(args.dataset),'head_sha256':prior.probe.sha_file(args.detector),
        'code_sha256':prior.shared.code_hashes(),'config':CONFIG,'fixture':fixture,'n_boot':args.n_boot,
        'model_id':prior.MODEL_ID,'model_revision':prior.MODEL_REVISION,'model_path':str(args.model_path)}
    identity=digest(config);meta_path=out/'run_metadata.json';previous=read(meta_path) if meta_path.exists() else {}
    if previous and previous['identity']!=identity:raise ValueError('changed input/code/model; new folder required')
    if not previous and out.exists() and any(out.iterdir()):raise ValueError('unknown output directory')
    rows=data['rows'];active=[r for r in rows if routed(r,data)]
    # Check every original byte used by the experiment, before any neural call.
    targets=prior.old.verified_images(args.sup_zip,[r['image'] for r in active],out/'_images/target')
    refs={r['image']:data['references'][r['image']] for r in active}
    normals=prior.old.verified_images(args.unsup_zip,list(refs.values()),out/'_images/normal')
    for role,paths in [('target',targets),('reference',normals)]:
        for name,path in paths.items():
            if prior.probe.sha_file(path)!=expected[role,name]:raise ValueError('original image SHA changed')
    for path in targets.values():original_image(path)
    out.mkdir(parents=True,exist_ok=True)
    meta={**config,'identity':identity,'complete':False,'backend':'fixture' if fixture else 'hf',
        'images':len(rows),'routed_images':len(active),'q1_calls':0,'q2_calls':0,'gold_used':False,'training_calls':0,
        'positive_spatial_accuracy':None,'positive_spatial_support':0,'source_run_sha256':data['source_run_sha256'],
        'claim_limit':'weak occlusion proposals; masking artifacts possible; no positive location GT; already examined dev'}
    write(meta_path,meta);write(out/'run_status.json',{'complete':False,'phase':'occlusion'})
    (out/'dataset_snapshot.json').write_bytes(args.dataset.read_bytes());(out/'frozen_detector.json').write_bytes(args.detector.read_bytes())
    feature_path=out/'features.json';cache=read(feature_path) if feature_path.exists() else {};used={};maps={};counts=Counter()
    holder=[extractor];feature_failure=''
    try:
        for row in active:
            core=row['core_id'];tool=data['experts'][core];origin=args.dataset.parent/'original_features'/(core+'.npy')
            if prior.probe.sha_file(origin)!=tool['feature_sha256']:raise ValueError('original feature changed')
            x=np.load(origin,allow_pickle=False)
            if x.dtype!=np.float32 or x.shape!=(len(head['coef']),) or not np.isfinite(x).all() or not np.isclose(np.linalg.norm(x),1,atol=1e-5):raise ValueError('invalid original feature')
            score=float(prior.probe.predict(head,x[None,:])[0])
            if not np.isclose(score,tool['score'],rtol=1e-12,atol=1e-14) or not score>=head['threshold']:raise ValueError('original tool score mismatch')
            if tool['image']!=row['image'] or tool['image_sha256']!=expected['target',row['image']]:raise ValueError('tool image identity mismatch')
            original=logit(head,x);image=original_image(targets[row['image']]);occluded={k:[] for k in CONFIG['replacements']};error=''
            for kind in CONFIG['replacements']:
                for cell in range(1,10):
                    key=core+'/'+kind+'/'+str(cell);variant=masked(image,cell,kind);variant_sha=sha(png(variant));rec=cache.get(key)
                    featurefile=out/'feature_cache'/(digest(key)+'.npy')
                    if rec and (rec['identity']!=identity or rec['variant_sha256']!=variant_sha):raise ValueError('changed feature request')
                    if rec and not rec['error']:
                        if not featurefile.is_file() or prior.probe.sha_file(featurefile)!=rec['feature_sha256']:raise ValueError('feature cache corruption')
                        v=np.load(featurefile,allow_pickle=False)
                        if v.dtype!=np.float32 or v.shape!=x.shape or not np.isfinite(v).all() or not np.isclose(np.linalg.norm(v),1,atol=1e-5):raise ValueError('invalid cached vector')
                        if not np.isclose(logit(head,v),rec['logit'],rtol=1e-12,atol=1e-12):raise ValueError('feature score corruption')
                        counts['reused_features']+=1
                    else:
                        if rec:write(out/'attempt_history'/(digest(rec)+'.json'),rec)
                        rec={'id':key,'identity':identity,'variant_sha256':variant_sha,'image_sha256':tool['image_sha256'],
                             'kind':kind,'cell':cell,'box':list(box(image.size,cell)),'error':feature_failure,'logit':None,'feature_sha256':None}
                        start=time.time()
                        if not feature_failure:
                            try:
                                if holder[0] is None:holder[0]=prior.probe.DinoFeatures()
                                counts['new_features']+=1;v=vector(holder[0](variant),len(head['coef']))
                                featurefile.parent.mkdir(parents=True,exist_ok=True)
                                with featurefile.with_suffix('.tmp').open('wb') as f:np.save(f,v,allow_pickle=False)
                                featurefile.with_suffix('.tmp').replace(featurefile)
                                rec.update(logit=logit(head,v),feature_sha256=prior.probe.sha_file(featurefile))
                            except Exception as exc:feature_failure=rec['error']=f'{type(exc).__name__}: {exc}'
                        else:counts['blocked_features']+=1
                        rec['latency_s']=round(time.time()-start,4)
                    used[key]=rec;write(feature_path,{**cache,**used});occluded[kind].append(rec['logit']);error=error or rec['error']
            maps[core]={'error':error,'original_logit':original,'occluded_logits':occluded,
                        'proposal':None if error else candidates(original,occluded)}
            write(out/'occlusion_maps.json',maps)
            print('occlusion saved',len(maps),'/',len(active),flush=True)
    finally:
        holder[0]=None;gc.collect()
        if not fixture:
            import torch
            torch.cuda.empty_cache()
    if set(cache)-set(used):raise ValueError('unexpected feature cache entries')
    finals={p:dict(data['cached_direct']) for p in POLICIES};finals['cached_cue_review']=dict(data['cached_cue_review'])
    raw_path=out/'calls.json';cached=read(raw_path) if raw_path.exists() else {};calls={};routes=[];crop_manifest=[]
    spec={**prior.agent.base.MODEL_REGISTRY[prior.MODEL]}
    if not fixture:spec['model_id']=str(args.model_path)
    if not fixture:
        def loader(model_spec):
            import torch
            if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<20*1024**3:raise RuntimeError('native L4/A100 >=20GB required')
            torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
            return prior.hf.load_hf_adapter(model_spec)
        infer=prior.hf.infer_hf
    adapter=prior.agent.base.LazyHFAdapter(spec,loader=loader);load_failure=''
    try:
        for row in active:
            core=row['core_id'];record=maps[core];proposal=record['proposal'];selected=[] if not proposal else proposal['regions']
            routes.append({'core_id':core,'first':data['cached_direct'][core],'feature_error':record['error'],
                'proposal_cells':selected,'review_requested':bool(selected) and not record['error']})
            if not selected or record['error']:continue
            image=original_image(targets[row['image']])
            order=list(NEW)
            if int(hashlib.sha256(core.encode()).hexdigest(),16)%2:order.reverse()
            for policy in order:
                cells=selected if policy=='occlusion_crop' else CONFIG['fixed_cells'][:len(selected)]
                crop=out/'crops'/(core+'_'+policy+'.png');crop.parent.mkdir(exist_ok=True)
                payload=png(sheet(image,cells));crop.write_bytes(payload)
                crop_manifest.append({'core_id':core,'policy':policy,'cells':cells,'crop_sha256':sha(payload),
                    'source_sha256':expected['target',row['image']],'boxes':[list(box(image.size,c)) for c in cells]})
                images=[normals[refs[row['image']]],targets[row['image']],crop]
                request={'identity':identity,'prompt':prompt(data['cached_direct'][core],cells),
                    'images':list(map(str,images)),'image_sha256':[expected['reference',refs[row['image']]],expected['target',row['image']],sha(payload)],
                    'image_tile_budgets':CONFIG['image_tile_budgets'],'generation':{'max_new_tokens':80,'do_sample':False}}
                key=core+'/'+policy;rec=cached.get(key);req_sha=digest(request)
                if rec and (rec['request_sha256']!=req_sha or any(rec[k]!=v for k,v in request.items())):raise ValueError('changed LVLM request')
                if rec and not rec['error']:
                    if prior.spatial.parse_json(rec['output'],'regions')!=rec['prediction']:raise ValueError('changed cached parsing')
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
                    rec={'id':key,'core_id':core,'policy':policy,**request,'request_sha256':req_sha,'output':output,
                         'prediction':prior.spatial.parse_json(output,'regions'),'error':error,'latency_s':round(time.time()-start,4)}
                calls[key]=rec;write(raw_path,{**cached,**calls});finals[policy][core]=rec['prediction']
            write(out/'run_status.json',{'complete':False,'phase':'crop_review','saved':len(calls)})
            if len(calls)%10==0:print('crop reviews saved',len(calls),flush=True)
    finally:adapter.release()
    if set(cached)-set(calls):raise ValueError('unexpected LVLM cache entries')
    errors={p:sum(bool(r['error']) for r in maps.values())+sum(bool(r['error']) for r in calls.values() if r['policy']==p) if p in NEW else 0 for p in POLICIES}
    eligible={p:errors[p]==0 for p in POLICIES};leaderboard=[];bydefect=[];byasset=[]
    for p in POLICIES:
        report=prior.spatial.reports(rows,finals[p]);write(out/('metrics_'+p+'.json'),dict(eligible=eligible[p],**report))
        if eligible[p]:leaderboard.append({'policy':p,**report['overall'],'positive_location_accuracy':None,'new_logical_lvlm_calls':sum(r['policy']==p for r in calls.values())})
        for group,key in [(bydefect,'defect_gt'),(byasset,'asset_category')]:
            for value in sorted({r[key] for r in rows}):group.append({'policy':p,key:value,**prior.spatial.score([r for r in rows if r[key]==value],finals[p])})
    prior.agent.cons.write_csv(out/'leaderboard.csv',leaderboard)
    prior.agent.cons.write_csv(out/'significance.csv',bootstrap(rows,finals,eligible,args.n_boot))
    write(out/'predictions.json',finals);write(out/'routes.json',routes);write(out/'crop_manifest.json',crop_manifest)
    write(out/'metrics_by_defect.json',bydefect);write(out/'metrics_by_asset.json',byasset)
    status={'complete':all(eligible.values()),'eligible':eligible,'execution_errors_by_policy':errors,
        'invalid_new_outputs':sum(r['prediction'] is None for r in calls.values()),'routed_images':len(active),
        'proposal_abstentions':sum(not m['error'] and m['proposal']['abstained'] for m in maps.values()),
        'positive_location_accuracy':None,'positive_spatial_support':0,'fixture':fixture}
    meta.update(**counts,complete=status['complete'],raw_masked_features=len(used),raw_new_lvlm_calls=len(calls),
                runner_wall_time_s=round(time.time()-started,4),gpu=prior.agent.gpu_info(),
                expert_model_id=prior.probe.MODEL,expert_model_revision=prior.probe.REVISION)
    meta['software']={}
    for name in ['torch','transformers','accelerate','numpy','pillow','timm']:
        try:meta['software'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:meta['software'][name]=None
    write(meta_path,meta);write(out/'run_status.json',status);print(json.dumps(status),flush=True)
    return 0 if status['complete'] else 1


def main(argv=None,**kwargs):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['dataset','detector','sup-zip','unsup-zip','out-dir']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--model-path',type=Path);p.add_argument('--n-boot',type=int,default=2000)
    return run(p.parse_args(argv),**kwargs)


if __name__=='__main__':raise SystemExit(main())

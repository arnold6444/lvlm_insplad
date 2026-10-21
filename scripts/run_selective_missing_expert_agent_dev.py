#!/usr/bin/env python3
"""Dev-only normal-reference LVLM + frozen missing-component expert, with matched controls."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_q2_conflict_evidence_dev as shared
import run_missing_component_probe as probe
import q3_multiregion as spatial

agent,old,refined,common,hf=shared.agent,shared.old,shared.refined,shared.common,shared.hf
MODEL,MODEL_ID,MODEL_REVISION=shared.MODEL,shared.MODEL_ID,shared.MODEL_REVISION
PROTOCOL='selective_missing_expert_agent_dev_v1'
PREFIX='selective_missing_expert_agent_dev'
DETECTOR_SHA256='d9ff0610d0b74752a2dfc5b53f1cdc1db5e916e17b4260172f1c7abab8c6e7f1'
POLICIES=('lvlm_only','expert_q2_override','selective_agent','matched_reask')
COMPARISONS=[('lvlm_only','expert_q2_override'),('lvlm_only','selective_agent'),
             ('expert_q2_override','selective_agent'),('matched_reask','selective_agent')]
MISSING='Missing cap / missing insulator component.'
digest=shared.digest


def validate_inputs(data,artifact,artifact_sha,fixture=False):
    if data.get('protocol')!=PROTOCOL or not data.get('rows'):
        raise ValueError('expected separate integrated dev input')
    rows=data['rows']
    if any(r.get('protocol')!='gold_v2_dev' or r['question_type'] not in (agent.Q1,agent.Q2) for r in rows):
        raise ValueError('Q1/Q2 dev only; gold forbidden')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate QA IDs')
    refined.select_rows(rows,len(rows)//2,42)
    q1=[r for r in rows if r['question_type']==agent.Q1]
    if set(data['q1_fixed'])!={r['id'] for r in q1}:raise ValueError('Q1 cache coverage mismatch')
    if any(data['q1_fixed'][r['id']] not in r['options'] for r in q1):raise ValueError('invalid Q1 cache')
    if artifact_sha!=data['detector_sha256'] or (not fixture and artifact_sha!=DETECTOR_SHA256):
        raise ValueError('wrong frozen detector SHA256; no retraining or dev tuning')
    if (artifact.get('model_id'),artifact.get('model_revision'),artifact.get('feature'))!=(probe.MODEL,probe.REVISION,probe.FEATURE):
        raise ValueError('wrong expert model/feature contract')
    if artifact.get('gold_scored') is not False or artifact.get('selection_uses')!=['fit','calibration']:
        raise ValueError('invalid frozen detector provenance')
    dim=len(artifact['coef'])
    if dim<2 or any(len(artifact[k])!=dim for k in ('mean','scale')):
        raise ValueError('detector feature dimension mismatch')
    if (not np.isfinite([*artifact['coef'],*artifact['mean'],*artifact['scale'],artifact['threshold'],artifact['intercept']]).all()
        or min(artifact['scale'])<=0 or not 0<=artifact['threshold']<=1):
        raise ValueError('invalid frozen detector coefficients')
    expected={(r['role'],r['member']):r['sha256'] for r in data['image_manifest']}
    if len(expected)!=len(data['image_manifest']):raise ValueError('duplicate image manifest')
    for r in rows:
        if r['question_type']==agent.Q2 and set(r['options'].values())!=set(refined.DEFINITIONS):
            raise ValueError('unexpected Q2 options')
        ref=data['references'].get(r['image'])
        if not ref or ('target',r['image']) not in expected or ('reference',ref) not in expected:
            raise ValueError('missing reference/image SHA')
        if not any(ref.startswith(r['asset_category']+'/train/'+normal+'/') for normal in ('good','normal')):
            raise ValueError('reference must be same-asset train normal')


def q3_row(row):
    return {**{k:row[k] for k in ('core_id','image','asset_category','anomaly_gt','defect_gt')},
        'id':row['core_id']+'-q3-multiregion-dev','question_type':agent.Q3,'protocol':PROTOCOL,
        'regions_gt':[] if row['anomaly_gt']=='normal' else None,'label_basis':'dev_image_label_only'}


def initial_prompt(row):
    return refined.definition_prompt(row,True) if row['question_type']==agent.Q2 else spatial.prompt('regions',True)


def parse(row,output):
    return agent.base.parse_choice(output,set(row['options'])) if row['question_type']==agent.Q2 else spatial.parse_json(output,'regions')


def missing_choice(row):
    return next(k for k,v in row['options'].items() if v==MISSING)


def needs_expert(asset,question_type,first,missing=None):
    if asset!=probe.ASSET or first is None:return False
    if question_type==agent.Q2:return first!=missing
    if question_type==agent.Q3:return first==[]
    raise ValueError('expert routing is Q2/Q3 only')


def review_prompt(row,first,with_expert):
    if row['question_type'] not in (agent.Q2,agent.Q3):raise ValueError('same-question review only')
    own=row['options'][first] if row['question_type']==agent.Q2 else json.dumps({'regions':first})
    instruction=('Re-examine this SAME question from the images. Its earlier answer was '+own+
        '. This answer is unverified and may be wrong. ')
    if with_expert:
        instruction+=('A separate frozen supervised image classifier for glass insulators flags a POSSIBLE '
            'missing/broken insulator component in the TARGET. This is an experimental suggestion, not a fact. '
            'It provides NO position or bounding box. Accept or reject the suggestion using visible TARGET evidence; '
            'do not infer missing parts from viewpoint, lighting or occlusion alone. ')
    else:
        instruction+='Decide again from the TARGET and the same visual definitions; do not follow the earlier answer automatically. '
    suffix=('Return exactly one option letter and nothing else.' if row['question_type']==agent.Q2 else
        'Return only JSON {"regions":[1,9]} with applicable cells, or {"regions":[]} when no visible defect exists.')
    return initial_prompt(row)+'\n\n'+instruction+'\n'+suffix


def q3_bootstrap(rows,choices,eligible,n_boot):
    n=len(rows);positive=np.array([r['anomaly_gt']=='anomalous' for r in rows],float)
    weights=np.random.default_rng(42).multinomial(n,np.full(n,1/n),size=n_boot).astype(float)
    output=[]
    def values(pred,w):
        tp=np.array([pred[r['core_id']] is not None and bool(pred[r['core_id']]) for r in rows],float)*positive
        tn=np.array([pred[r['core_id']]==[] for r in rows],float)*(1-positive)
        with np.errstate(divide='ignore',invalid='ignore'):
            recall=(w@tp)/(w@positive);specificity=(w@tn)/(w@(1-positive))
        return {'q3_presence_balanced_accuracy':(recall+specificity)/2,
            'q3_anomaly_miss_rate':1-recall,'q3_normal_fpr':1-specificity}
    for a,b in COMPARISONS:
        if not eligible[a] or not eligible[b]:continue
        va,vb=values(choices[a],weights),values(choices[b],weights)
        pa,pb=values(choices[a],np.ones((1,n))),values(choices[b],np.ones((1,n)))
        for metric in va:
            delta=vb[metric]-va[metric];finite=delta[np.isfinite(delta)]
            if not len(finite) or not np.isfinite(pa[metric][0]+pb[metric][0]):continue
            lo,hi=np.percentile(finite,[2.5,97.5])
            output.append({'baseline':a,'condition':b,'metric':metric,'delta':float(pb[metric][0]-pa[metric][0]),
                'ci_low':float(lo),'ci_high':float(hi),'valid_resamples':len(finite),'n_boot':n_boot,
                'scope':'exploratory dev presence only; no positive spatial labels'})
    return output


def main(argv=None,*,loader=None,infer=None,extractor=None):
    started=time.time()
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','detector','sup-zip','unsup-zip','out-dir'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--model-path',type=Path);p.add_argument('--n-images',type=int,default=240)
    p.add_argument('--n-boot',type=int,default=2000);args=p.parse_args(argv)
    supplied=(loader is not None,infer is not None,extractor is not None)
    if any(supplied) and not all(supplied):raise ValueError('fixtures require loader, infer AND extractor')
    fixture=all(supplied);out=args.out_dir
    if not out.name.startswith(PREFIX) or args.n_boot<1:raise ValueError('separate output prefix and positive bootstrap required')
    if (out.resolve().is_relative_to(args.dataset.parent.resolve()) or
        any(source.resolve().is_relative_to(out.resolve()) for source in (args.dataset,args.detector,args.sup_zip,args.unsup_zip))):
        raise ValueError('output overlaps input directory')
    data=json.loads(args.dataset.read_text('utf-8'));artifact=json.loads(args.detector.read_text('utf-8'))
    head_sha=probe.sha_file(args.detector);validate_inputs(data,artifact,head_sha,fixture)
    rows,cohorts=refined.select_rows(data['rows'],args.n_images,42)
    q2rows=[r for r in rows if r['question_type']==agent.Q2];regions=[q3_row(r) for r in q2rows]
    spec={**agent.base.MODEL_REGISTRY[MODEL]}
    if not fixture:
        if args.model_path is None or args.model_path.name!=MODEL_REVISION or not args.model_path.is_dir():
            raise ValueError('use the pinned native model snapshot directory')
        spec['model_id']=str(args.model_path)
    config={'protocol':PROTOCOL,'dataset_sha256':probe.sha_file(args.dataset),'detector_sha256':head_sha,
        'n_images':args.n_images,'n_boot':args.n_boot,'rows':rows,'code_sha256':shared.code_hashes(),
        'model_id':MODEL_ID,'model_revision':MODEL_REVISION,'model_path':str(args.model_path),
        'fixture':fixture,'generation':{'max_new_tokens':80,'do_sample':False,'image_tile_budgets':[3,3]},
        'expert_model_id':probe.MODEL,'expert_model_revision':probe.REVISION,'expert_feature':probe.FEATURE,
        'expert_scope':'glass-insulator missing component supervised; no position output',
        'q1':'fixed cache, zero inference','q3_spatial_positive_support':0,'gold_used':False}
    signature=digest(config);mp=out/'run_metadata.json'
    previous=json.loads(mp.read_text('utf-8')) if mp.exists() else {}
    if previous and previous.get('input_signature')!=signature:raise ValueError('inputs/code/model changed; use NEW output directory')
    if not previous and out.exists() and any(out.iterdir()):raise ValueError('unrecognized nonempty output')
    targets=old.verified_images(args.sup_zip,[r['image'] for r in q2rows],out/'_images/target')
    refs={r['image']:data['references'][r['image']] for r in q2rows}
    normals=old.verified_images(args.unsup_zip,list(refs.values()),out/'_images/normal')
    manifest=shared.previous.tile_manifest(targets,normals)
    expected={(r['role'],r['member']):r['sha256'] for r in data['image_manifest']}
    if any(expected.get((r['role'],r['member']))!=r['sha256'] for r in manifest):raise ValueError('original image SHA differs')
    identity=digest([signature,manifest])
    if previous.get('prediction_identity') and previous['prediction_identity']!=identity:raise ValueError('image identity changed')
    meta={**config,'input_signature':signature,'prediction_identity':identity,'complete':False,'backend':'fixture' if fixture else 'hf',
        'benchmark_commit':agent.git_commit(),'source_identification':'exact self-contained code_sha256; commit may be null outside a checkout',
        'selection_performed':False,'detector_training_calls':0,'claim_limit':'already examined dev; supervised tool; new model effect unverified until real run'}
    old.atomic_json(mp,meta);old.atomic_json(out/'run_status.json',{'complete':False,'status':'running'})
    (out/'dataset_snapshot.json').write_bytes(args.dataset.read_bytes());(out/'frozen_detector.json').write_bytes(args.detector.read_bytes())
    common.save_records(out/'subset_qa_q12.jsonl',rows);common.save_records(out/'q3_dev_partial.jsonl',regions)
    agent.cons.write_csv(out/'image_manifest.csv',manifest)
    callpath=out/'calls.json';existing=json.loads(callpath.read_text()) if callpath.exists() else {};used={}
    toolpath=out/'tool_calls.json';cachedtools=json.loads(toolpath.read_text()) if toolpath.exists() else {};usedtools={}
    counts=Counter(new_lvlm_calls=0,reused_lvlm_calls=0,blocked_lvlm_calls=0,new_expert_features=0,reused_expert_features=0,logical_expert_requests=0)
    load_failure='';expert_failure='';expert_holder=[extractor]
    if not fixture:
        def loader(model_spec):
            import torch
            if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<20*1024**3:
                raise RuntimeError('native L4/A100 VRAM>=20GB required')
            torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
            return hf.load_hf_adapter(model_spec)
        infer=hf.infer_hf
    holder=agent.base.LazyHFAdapter(spec,loader=loader)
    def save_calls():
        if used:old.atomic_json(callpath,{**existing,**used})
    def call(row,step,prompt):
        nonlocal load_failure
        key=row['id']+'/'+step;images=[normals[refs[row['image']]],targets[row['image']]]
        request={'identity':identity,'question_type':row['question_type'],'prompt':prompt,
            'options':row.get('options'),'images':list(map(str,images)),'image_tile_budgets':[3,3]}
        request_sha=digest(request);rec=existing.get(key)
        if rec and (rec.get('request_sha256')!=request_sha or any(rec.get(k)!=v for k,v in request.items())):
            raise ValueError('cached LVLM request differs')
        if rec and not rec['error']:
            if parse(row,rec['output'])!=rec['prediction']:raise ValueError('cached parsing changed')
            counts['reused_lvlm_calls']+=1
        else:
            if rec:old.atomic_json(out/'attempt_history'/(digest(rec)+'.json'),rec)
            output,error,start='',load_failure,time.time()
            if error:counts['blocked_lvlm_calls']+=1
            else:
                try:adapter=holder.get()
                except Exception as exc:load_failure=error=f'{type(exc).__name__}: {exc}'
                if not error:
                    try:output=str(infer({**adapter,'image_tile_budgets':[3,3]},images,prompt))
                    except Exception as exc:
                        error=f'{type(exc).__name__}: {exc}'
                        if 'out of memory' in error.lower():load_failure=error
                counts['new_lvlm_calls']+=1
            rec={'id':key,'question_id':row['id'],'step':step,**request,'request_sha256':request_sha,
                'output':output,'prediction':parse(row,output),'error':error,'latency_s':round(time.time()-start,4)}
        used[key]=rec;save_calls()
        if len(used)%20==0:
            print(f'{len(used)} LVLM calls saved',flush=True)
            old.atomic_json(out/'run_status.json',{'complete':False,'status':'running','finished_lvlm_calls':len(used),**counts})
        return rec
    def expert(row):
        nonlocal expert_failure
        counts['logical_expert_requests']+=1;core=row['core_id'];image_sha=expected['target',row['image']]
        rec=usedtools.get(core,cachedtools.get(core))
        featurepath=out/'tool_features'/(image_sha+'.npy')
        if rec and (rec.get('identity')!=identity or rec.get('image_sha256')!=image_sha or
            rec.get('detector_sha256')!=head_sha or rec.get('image')!=row['image']):
            raise ValueError('expert cache identity differs')
        if rec and not rec['error']:
            if not featurepath.is_file() or probe.sha_file(featurepath)!=rec['feature_sha256']:raise ValueError('expert feature checksum differs')
            x=np.load(featurepath,allow_pickle=False)
            if x.shape!=(len(artifact['coef']),) or not np.isfinite(x).all() or not np.isclose(np.linalg.norm(x),1,atol=1e-5):raise ValueError('invalid expert feature cache')
            score=float(probe.predict(artifact,x[None,:])[0])
            if not np.isclose(score,rec['score'],rtol=1e-12,atol=1e-14) or rec['positive']!=(score>=artifact['threshold']):raise ValueError('expert cached decision differs')
            counts['reused_expert_features']+=1
        else:
            if rec:old.atomic_json(out/'attempt_history'/(digest(rec)+'.json'),rec)
            error,start=expert_failure,time.time();score=None;feature_sha=None
            if not error:
                if expert_holder[0] is None:
                    try:expert_holder[0]=probe.DinoFeatures()
                    except Exception as exc:expert_failure=error=f'{type(exc).__name__}: {exc}'
                if not error:
                    try:
                        with Image.open(targets[row['image']]) as im:x=np.asarray(expert_holder[0](ImageOps.exif_transpose(im).convert('RGB')),dtype=np.float32)
                        if x.shape!=(len(artifact['coef']),) or not np.isfinite(x).all() or np.linalg.norm(x)<=0:raise ValueError('invalid extracted expert vector')
                        x=x/np.linalg.norm(x);featurepath.parent.mkdir(parents=True,exist_ok=True)
                        with featurepath.with_suffix('.tmp').open('wb') as handle:np.save(handle,x,allow_pickle=False)
                        featurepath.with_suffix('.tmp').replace(featurepath);feature_sha=probe.sha_file(featurepath)
                        score=float(probe.predict(artifact,x[None,:])[0]);counts['new_expert_features']+=1
                    except Exception as exc:
                        error=f'{type(exc).__name__}: {exc}'
                        if 'out of memory' in error.lower():expert_failure=error
            rec={'core_id':core,'image':row['image'],'image_sha256':image_sha,'identity':identity,'detector_sha256':head_sha,
                'score':score,'threshold':artifact['threshold'],'positive':None if error else score>=artifact['threshold'],
                'feature_sha256':feature_sha,'error':error,'latency_s':round(time.time()-start,4)}
        usedtools[core]=rec;old.atomic_json(toolpath,{**cachedtools,**usedtools});return rec
    q1={r['id']:data['q1_fixed'][r['id']] for r in rows if r['question_type']==agent.Q1}
    finals={p:dict(q1) for p in POLICIES};locations={p:{} for p in POLICIES}
    deps={p:[] for p in POLICIES};tooldeps={p:set() for p in POLICIES};routes=[]
    try:
        for row,r3 in zip(q2rows,regions):
            for task in (row,r3):
                first=call(task,'initial',initial_prompt(task));choice=first['prediction'];is_q2=task['question_type']==agent.Q2
                for policy in POLICIES:
                    deps[policy].append(first)
                    if is_q2:finals[policy][task['id']]=choice
                    else:locations[policy][task['core_id']]=choice
                need=needs_expert(task['asset_category'],task['question_type'],choice,missing_choice(task) if is_q2 else None)
                route={'id':task['id'],'question_type':task['question_type'],'first':choice,'expert_requested':need,'review_requested':False}
                if need:
                    rec=expert(task)
                    for policy in ('selective_agent','matched_reask',*(['expert_q2_override'] if is_q2 else [])):
                        tooldeps[policy].add(task['core_id'])
                    route.update(expert_positive=rec['positive'],expert_error=rec['error'])
                    if rec['positive'] is True:
                        route['review_requested']=True
                        if is_q2:finals['expert_q2_override'][task['id']]=missing_choice(task)
                        order=['selective_agent','matched_reask']
                        if int(hashlib.sha256(task['id'].encode()).hexdigest(),16)%2:order.reverse()
                        for policy in order:
                            review=call(task,policy,review_prompt(task,choice,policy=='selective_agent'));deps[policy].append(review)
                            if is_q2:finals[policy][task['id']]=review['prediction']
                            else:locations[policy][task['core_id']]=review['prediction']
                routes.append(route)
    except BaseException:
        old.atomic_json(out/'run_status.json',{'complete':False,'status':'interrupted','finished_lvlm_calls':len(used),**counts});raise
    finally:
        save_calls();expert_holder[0]=None;holder.release()
    planned=2*len(q2rows)+2*sum(r['review_requested'] for r in routes)
    if len(used)!=planned or sum(counts[k] for k in ('new_lvlm_calls','reused_lvlm_calls','blocked_lvlm_calls'))!=planned:
        raise RuntimeError('LVLM route coverage mismatch')
    answers=[];leaderboard=[];bydefect=[];byasset=[];transitions=[];statuses=[];found={};eligible={};cohort_metrics=[]
    for policy in POLICIES:
        errors=sum(bool(r['error']) for r in deps[policy])+sum(bool(usedtools[c]['error']) for c in tooldeps[policy])
        q1fixed=all(finals[policy][rid]==value for rid,value in q1.items())
        eligible[policy]=errors==0 and q1fixed
        invalids=sum(r['prediction'] is None for r in deps[policy])
        statuses.append({'policy':policy,'eligible_for_comparison':eligible[policy],'q1_fixed':q1fixed,
            'row_errors':errors,'invalid_tool_outputs':invalids,'logical_lvlm_calls':len(deps[policy]),
            'logical_expert_images':len(tooldeps[policy])})
        q12=common.q12_metrics(rows,finals[policy]);q3=spatial.reports(regions,locations[policy])
        q1_rows={r['core_id']:r for r in rows if r['question_type']==agent.Q1}
        q12['joint_q1_q2_q3_presence']=sum(
            finals[policy][q1_rows[r['core_id']]['id']]==q1_rows[r['core_id']]['answer'] and
            finals[policy][r['id']]==r['answer'] and locations[policy][r['core_id']] is not None and
            bool(locations[policy][r['core_id']])==(r['anomaly_gt']=='anomalous') for r in q2rows)/len(q2rows)
        old.atomic_json(out/('metrics_'+policy+'.json'),{'eligible_for_comparison':eligible[policy],'q12':q12,'q3':q3,
            'q3_positive_location_accuracy':None,'q3_positive_spatial_support':0})
        for r in rows:
            answers.append({'policy':policy,'id':r['id'],'core_id':r['core_id'],'question_type':r['question_type'],
                'answer':r['answer'],'prediction':finals[policy][r['id']],'eligible_for_comparison':eligible[policy]})
        for r in regions:
            answers.append({'policy':policy,'id':r['id'],'core_id':r['core_id'],'question_type':agent.Q3,
                'answer':r['regions_gt'],'prediction':locations[policy][r['core_id']],'eligible_for_comparison':eligible[policy]})
        transitions.extend({'policy':policy,**r} for r in shared.transition_report(rows,finals['lvlm_only'],
            {rid:{'predicted_choice':v} for rid,v in finals[policy].items()}))
        for cohort,cores in cohorts.items():
            selected=[r for r in rows if r['core_id'] in cores];selected3=[r for r in regions if r['core_id'] in cores]
            if selected:cohort_metrics.append({'policy':policy,'cohort':cohort,'eligible_for_comparison':eligible[policy],
                'q12':common.q12_metrics(selected,finals[policy]),'q3':spatial.score(selected3,locations[policy])})
        if not eligible[policy]:continue
        found[MODEL,policy]=finals[policy]
        q3m=q3['overall']
        leaderboard.append({'policy':policy,**q12,**{'q3_'+k:q3m[k] for k in ('tp','fp','tn','fn','presence_balanced_accuracy','invalid_rate')},
            'q3_positive_location_accuracy':None,'q3_positive_spatial_support':0,
            'logical_lvlm_calls':len(deps[policy]),'logical_expert_images':len(tooldeps[policy])})
        for asset in sorted({r['asset_category'] for r in q2rows}):
            byasset.append({'policy':policy,'asset_category':asset,**common.q12_metrics([r for r in rows if r['asset_category']==asset],finals[policy]),
                'q3':spatial.score([r for r in regions if r['asset_category']==asset],locations[policy])})
        for defect in sorted({r['defect_gt'] for r in q2rows}):
            selected=[r for r in q2rows if r['defect_gt']==defect]
            bydefect.append({'policy':policy,'defect':defect,'images':len(selected),
                'q2_correct':sum(finals[policy][r['id']]==r['answer'] for r in selected),
                'q2_false_normal':sum(r['anomaly_gt']=='anomalous' and agent.base.predicted_semantic(r,finals[policy][r['id']])=='normal' for r in selected),
                'q3':spatial.score([r for r in regions if r['defect_gt']==defect],locations[policy])})
    significance=[r for r in agent.cons.significance_rows(agent.cons.image_pairs(rows),[],found,[MODEL],COMPARISONS,args.n_boot)
        if r['metric'] not in ('q3_accuracy','q1q3_contradiction')]
    for name,table in [('answers.csv',answers),('leaderboard.csv',leaderboard),('transitions.csv',transitions),('q12_significance.csv',significance),
        ('q3_significance.csv',q3_bootstrap(regions,locations,eligible,args.n_boot))]:agent.cons.write_csv(out/name,table)
    for name,value in [('routes.json',routes),('predictions.json',{'q12':finals,'q3':locations}),('cohort_metrics.json',cohort_metrics),('metrics_by_asset.json',byasset),('metrics_by_defect.json',bydefect)]:old.atomic_json(out/name,value)
    complete=all(eligible.values())
    meta.update(complete=complete,**counts,raw_lvlm_calls=len(used),raw_expert_images=len(usedtools),planned_lvlm_calls=planned,
        q1_inference_calls=0,gpu=agent.gpu_info(),row_errors=sum(bool(r['error']) for r in used.values())+sum(bool(r['error']) for r in usedtools.values()),
        invalid_lvlm_outputs=sum(r['prediction'] is None for r in used.values()),
        latency_s_by_policy={p:sum(r['latency_s'] for r in deps[p]) for p in POLICIES},
        cost_note='same expert-positive routes and one review per question in agent/control; shared physical feature cache; cached calls are not free deployment',
        invalid_policy='retain raw invalid, count final invalid Q2 wrong and Q3 anomalous FN/normal FP; never correctness-retry; errors exclude policies')
    meta['runner_wall_time_s']=round(time.time()-started,4)
    meta['software']={}
    for package in ('torch','transformers','accelerate','numpy','pillow','timm'):
        try:meta['software'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:meta['software'][package]=None
    old.atomic_json(mp,meta);old.atomic_json(out/'run_status.json',{'complete':complete,'status':'complete' if complete else 'failed',
        'policies':statuses,'invalid_lvlm_outputs':meta['invalid_lvlm_outputs'],'selection_performed':False,'gold_used':False,'fixture':fixture})
    print(json.dumps({'complete':complete,'images':len(q2rows),'planned_lvlm_calls':planned,**counts,'new_neural_effect_claim':False}),flush=True)
    return 0 if complete else 1


if __name__=='__main__':raise SystemExit(main())

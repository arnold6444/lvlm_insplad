#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, gc, json, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts')); sys.path.insert(0,str(ROOT/'src'))
import run_gold_baselines as base
from insplad_benchmark.hf_adapters import infer_hf, load_hf_adapter
from insplad_benchmark.vision_expert import read_manifest, resolve_artifact_path

MODELS=('qwen3_5_4b','llava_onevision_7b','minicpm_v_4_5')
CONDS=('roi','heatmap')

def read_jsonl(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def write_jsonl(p,rows): base.write_jsonl(Path(p),rows)

def prompt(row, condition, expert):
    opts='\n'.join(f'{k}. {v}' for k,v in row['options'].items())
    ev='a candidate close-up selected by the vision expert' if condition=='roi' else 'an anomaly heatmap overlay produced by the vision expert'
    return (f'You are an industrial visual inspector. Image 1 is the complete TARGET image. Image 2 is {ev} ({expert}). '
            'Image 2 is only visual evidence and may highlight a normal region, so do not assume anomaly from the highlight alone. '
            'Answer the question about Image 1 using both images.\n\n'
            f"Question: {row['question']}\n{opts}\n\nReturn exactly one option letter and nothing else.")

def index_manifest(path, root, expert, core_ids):
    idx={}
    for r in read_manifest(path):
        if r.get('expert')!=expert or r.get('query_id') not in core_ids: continue
        qid=r['query_id']
        if qid in idx: raise RuntimeError(f'duplicate {expert}/{qid}')
        idx[qid]=dict(r)
    missing=sorted(core_ids-set(idx))
    if missing: raise RuntimeError(f'{expert} missing {missing[:10]}')
    for r in idx.values():
        for f in ('roi_path','overlay_path'):
            p=resolve_artifact_path(path,r[f],root)
            if not p.is_file(): raise FileNotFoundError(p)
    return idx

def main():
    p=argparse.ArgumentParser(); p.add_argument('--qa',type=Path,required=True); p.add_argument('--sup-zip',type=Path,required=True); p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--patchcore-manifest',type=Path,required=True); p.add_argument('--patchcore-root',type=Path,required=True)
    p.add_argument('--reconpatch-manifest',type=Path,required=True); p.add_argument('--reconpatch-root',type=Path,required=True)
    p.add_argument('--models',nargs='+',default=list(MODELS),choices=list(MODELS)); p.add_argument('--conditions',nargs='+',default=list(CONDS),choices=list(CONDS)); p.add_argument('--checkpoint-every',type=int,default=5)
    a=p.parse_args(); rows=read_jsonl(a.qa); core_ids={r['core_id'] for r in rows}
    expert_specs={
      'patchcore':(a.patchcore_manifest,a.patchcore_root),
      'reconpatch_lite':(a.reconpatch_manifest,a.reconpatch_root),
    }
    indexes={e:index_manifest(m,r,e,core_ids) for e,(m,r) in expert_specs.items()}
    a.out_dir.mkdir(parents=True,exist_ok=True)
    targets=base.prepare_images(rows,a.sup_zip,a.out_dir/'_images'/'target')
    leaderboard=[]; by_asset=[]; status=[]
    for model_key in a.models:
      spec=base.MODEL_REGISTRY[model_key]; adapter=load_hf_adapter(spec)
      try:
        for expert,(manifest,root) in expert_specs.items():
          for condition in a.conditions:
            pred_path=a.out_dir/'predictions'/model_key/f'{expert}__{condition}.jsonl'; existing={r['id']:r for r in read_jsonl(pred_path)} if pred_path.is_file() else {}; preds=[]; called=0; errs=0
            for i,row in enumerate(rows,1):
              old=existing.get(row['id'])
              if base.reusable_prediction(old): preds.append(old); continue
              er=indexes[expert][row['core_id']]; field='roi_path' if condition=='roi' else 'overlay_path'; ev=resolve_artifact_path(manifest,er[field],root)
              t0=time.time(); out=''; err=None
              try: out=infer_hf(adapter,[targets[row['image']],ev],prompt(row,condition,expert))
              except Exception as exc: err=f'{type(exc).__name__}: {exc}'; errs+=1
              choice=base.parse_choice(out,{str(k).upper() for k in row['options']})
              preds.append({'id':row['id'],'core_id':row['core_id'],'asset_category':row['asset_category'],'question_type':row['question_type'],'output':out,'predicted_choice':choice,'latency_s':round(time.time()-t0,3),'error':err,'expert':expert,'condition':condition})
              called+=1
              if called%a.checkpoint_every==0: write_jsonl(pred_path,preds)
            write_jsonl(pred_path,preds)
            metric, assets=base.evaluate_model(model_key,rows,pred_path); metric.update({'expert':expert,'condition':condition}); leaderboard.append(metric)
            q3=Counter(r['asset_category'] for r in rows if r['question_type']=='defect_localization')
            for x in assets:
              x.update({'expert':expert,'condition':condition,'q3_questions':q3.get(x['asset_category'],0)})
              if x['q3_questions']==0: x['q3_accuracy']=None
            by_asset.extend(assets); status.append({'model_key':model_key,'expert':expert,'condition':condition,'status':'completed','called_this_run':called,'row_errors':errs})
      finally:
        del adapter; gc.collect()
        try:
          import torch
          if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception: pass
    base.write_csv(a.out_dir/'leaderboard.csv',leaderboard); base.write_csv(a.out_dir/'metrics_by_asset.csv',by_asset); (a.out_dir/'run_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    grouped=defaultdict(list)
    for r in leaderboard: grouped[(r['expert'],r['condition'])].append(r)
    summary=[]
    for (e,c),rs in grouped.items():
      summary.append({'expert':e,'condition':c,'models':len(rs),'mean_overall_accuracy':sum(x['overall_accuracy'] for x in rs)/len(rs),'mean_anomaly_f1':sum(x['anomaly_f1'] for x in rs)/len(rs),'mean_defect_macro_f1':sum(x['defect_macro_f1'] for x in rs)/len(rs),'mean_localization_accuracy':sum(x['localization_accuracy'] for x in rs)/len(rs),'mean_primary_score':sum((x['anomaly_f1']+x['defect_macro_f1'])/2 for x in rs)/len(rs)})
    summary.sort(key=lambda x:(x['mean_primary_score'],x['mean_overall_accuracy']),reverse=True); base.write_csv(a.out_dir/'screening_summary.csv',summary)
    (a.out_dir/'run_metadata.json').write_text(json.dumps({'condition':'gold_vision_expert_smoke','qa_count':len(rows),'models':a.models,'experts':list(expert_specs),'conditions':a.conditions,'selection_rule':'mean of anomaly_f1 and defect_macro_f1 across models; localization reported separately'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())

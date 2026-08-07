#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from insplad_benchmark.metrics import balanced_accuracy,binary_f1,confusion_matrix,consistency_error,invalid_output_rate,macro_f1,per_class_prf
from insplad_benchmark.parsing import parse_anomaly_output,parse_structured_output
INVALID='<invalid>'

def read_jsonl(path:Path)->list[dict]: return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def write_csv(path:Path,rows:list[dict])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text('',encoding='utf-8'); return
    with path.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def upsert_csv(path:Path,rows:list[dict],keys:tuple[str,...])->None:
    old=[]
    if path.exists() and path.stat().st_size:
        with path.open(newline='',encoding='utf-8') as f: old=list(csv.DictReader(f))
    newkeys={tuple(str(r.get(k,'')) for k in keys) for r in rows}; old=[r for r in old if tuple(str(r.get(k,'')) for k in keys) not in newkeys]; write_csv(path,old+rows)

def eval_anomaly(gt,predictions,out_dir,regime):
    def summary(rows):
        parsed=[parse_anomaly_output(predictions.get(r['id'],'')) for r in rows]; yt=[r['answer'] for r in rows]; yp=[p.anomaly if p.valid and p.anomaly else INVALID for p in parsed]
        return {'regime':regime,'n':len(rows),'valid_n':sum(p.valid for p in parsed),'balanced_accuracy':balanced_accuracy(yt,yp,['normal','anomalous']) if yt else 0.0,'f1':binary_f1(yt,yp) if yt else 0.0,'invalid_output_rate':invalid_output_rate([p.valid for p in parsed])}
    upsert_csv(out_dir/'anomaly_overall.csv',[summary(gt)],('regime',)); groups=defaultdict(list)
    for r in gt: groups[r['asset_category']].append(r)
    rows=[]
    for asset,g in sorted(groups.items()): d=summary(g); d['asset_category']=asset; rows.append(d)
    upsert_csv(out_dir/'anomaly_by_asset.csv',rows,('regime','asset_category'))

def eval_explainability(gt,predictions,out_dir):
    parsed=[parse_structured_output(predictions.get(r['id'],''),set(r['valid_defect_classes'])) for r in gt]
    def summary(idx):
        at=[gt[i]['answer']['anomaly'] for i in idx]; dt=[gt[i]['answer']['defect_class'] for i in idx]; ap=[parsed[i].anomaly if parsed[i].valid else INVALID for i in idx]; dp=[parsed[i].defect_class if parsed[i].valid else INVALID for i in idx]; labels=sorted(set(dt))
        joint=sum(at[j]==ap[j] and dt[j]==dp[j] for j in range(len(idx)))/len(idx) if idx else 0.0
        normals=[j for j,a in enumerate(at) if a=='normal']; false=sum(parsed[idx[j]].valid and dp[j] not in {'normal',INVALID} for j in normals); fdr=false/len(normals) if normals else 0.0
        pairs=[(ap[j],dp[j]) for j in range(len(idx)) if parsed[idx[j]].valid]
        return {'n':len(idx),'valid_n':sum(parsed[i].valid for i in idx),'defect_balanced_accuracy':balanced_accuracy(dt,dp,labels) if dt else 0.0,'defect_macro_f1':macro_f1(dt,dp,labels) if dt else 0.0,'joint_accuracy':joint,'false_defect_rate':fdr,'consistency_error':consistency_error([a for a,_ in pairs],[d for _,d in pairs]) if pairs else 0.0,'invalid_output_rate':invalid_output_rate([parsed[i].valid for i in idx])}
    allidx=list(range(len(gt))); write_csv(out_dir/'explainability_overall.csv',[summary(allidx)]); groups=defaultdict(list)
    for i,r in enumerate(gt): groups[r['asset_category']].append(i)
    by=[]
    for asset,idx in sorted(groups.items()): d=summary(idx); d['asset_category']=asset; by.append(d)
    write_csv(out_dir/'explainability_by_asset.csv',by)
    dt=[r['answer']['defect_class'] for r in gt]; dp=[p.defect_class if p.valid else INVALID for p in parsed]; labels=sorted(set(dt)); write_csv(out_dir/'explainability_by_defect.csv',[{'defect_class':k,**v} for k,v in per_class_prf(dt,dp,labels).items()])
    cols=labels+([INVALID] if INVALID in dp else []); matrix=confusion_matrix(dt,dp,cols); rows=[]
    for truth in labels:
        row={'actual_defect':truth}; row.update({f'pred_{p}':matrix.get(truth,{}).get(p,0) for p in cols}); rows.append(row)
    write_csv(out_dir/'explainability_confusion_matrix.csv',rows)

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--task',choices=['anomaly','explainability'],required=True); p.add_argument('--qa',type=Path,required=True); p.add_argument('--predictions',type=Path,required=True); p.add_argument('--out-dir',type=Path,default=Path('results')); p.add_argument('--regime',choices=['zero_shot','one_shot'],default='zero_shot'); a=p.parse_args(); gt=read_jsonl(a.qa); pred={r['id']:str(r.get('output','')) for r in read_jsonl(a.predictions)}
    eval_anomaly(gt,pred,a.out_dir,a.regime) if a.task=='anomaly' else eval_explainability(gt,pred,a.out_dir); return 0
if __name__=='__main__': raise SystemExit(main())

"""Separate box-center nine-cell set protocol; never rewrites frozen gold QA."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import build_explainable_qa_gold as qa

PROTOCOL = "gold_v2_q3_box_center_set_v1"
NAMES = ("upper-left", "upper", "upper-right", "left", "center", "right", "lower-left", "lower", "lower-right")
CELL = {name: i + 1 for i, name in enumerate(NAMES)}
GRID = "1=upper-left, 2=upper-center, 3=upper-right, 4=middle-left, 5=center, 6=middle-right, 7=lower-left, 8=lower-center, 9=lower-right"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def boxes_to_regions(boxes):
    if not boxes:
        raise ValueError("anomalous spatial GT requires boxes")
    result = set()
    for box in boxes:
        if len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
            raise ValueError("invalid box coordinates")
        x, y, w, h = box
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > 1.000002 or y + h > 1.000002:
            raise ValueError("box outside normalized image")
        result.add(CELL[qa.coarse_region_from_bboxes([box])])
    return sorted(result)


def build_dataset(master, localization, annotations):
    q1, q2 = {}, {}
    for r in master:
        store = q1 if r['question_type'] == 'anomaly_detection' else q2 if r['question_type'] == 'defect_classification' else None
        if store is not None:
            if r['core_id'] in store:
                raise ValueError("duplicate master ID")
            store[r['core_id']] = r
    if not q1 or q1.keys() != q2.keys():
        raise ValueError("Q1/Q2 cohort mismatch")
    loc = {}
    for r in localization:
        core = r['core_id']
        if core in loc or core not in q2:
            raise ValueError("duplicate/unknown spatial ID")
        if r.get('protocol') != 'gold_v2_loc_mmad':
            raise ValueError("expected the preserved 133-row-style location protocol")
        loc[core] = r
    if any(core not in q2 for core in annotations):
        raise ValueError("unknown manual ID")
    out = []
    for core, row in sorted(q2.items()):
        for key in ('image', 'asset_category', 'anomaly_gt', 'defect_gt', 'normal_reference_image'):
            if row[key] != q1[core][key]:
                raise ValueError("Q1/Q2 data mismatch: " + key)
        anomalous = row['anomaly_gt'] == 'anomalous'
        if row['anomaly_gt'] not in ('normal', 'anomalous') or anomalous != (row['defect_gt'] != 'normal'):
            raise ValueError("invalid binary/defect label")
        for ref in (row['image'], row['normal_reference_image']):
            if not ref or '\\' in ref or ref.startswith('/') or '..' in Path(ref).parts or ':' in ref:
                raise ValueError("unsafe image member")
        boxes, regions, basis = [], [], 'normal_source_label'
        annotation = annotations.get(core)
        if annotation:
            for key in ('core_id', 'image', 'asset_category', 'defect_gt'):
                if annotation[key] != row[key]:
                    raise ValueError("manual record mismatch: " + key)
            if not anomalous or annotation['status'] not in ('boxed', 'unlocatable'):
                raise ValueError("invalid manual status")
        if not anomalous:
            if core in loc:
                raise ValueError("normal image has positive spatial GT")
        elif core in loc:
            spatial = loc[core]
            for key in ('image', 'asset_category', 'anomaly_gt', 'defect_gt'):
                if spatial[key] != row[key]:
                    raise ValueError("location record mismatch: " + key)
            gt = spatial['localization_gt']
            boxes, basis = gt['normalized_bboxes_xywh'], gt['basis']
            if basis not in ('manual_box_annotation', 'insplad_seg_instance_annotation'):
                raise ValueError("unknown location source")
            if basis == 'manual_box_annotation':
                if not annotation or annotation['status'] != 'boxed' or boxes != annotation['boxes_xywh']:
                    raise ValueError("manual boxes differ from frozen location set")
            elif annotation:
                raise ValueError("manual override of existing seg is not allowed")
            regions = boxes_to_regions(boxes)
        else:
            if not annotation or annotation['status'] != 'unlocatable' or annotation['boxes_xywh']:
                raise ValueError("missing annotation must not become unknown silently")
            regions, basis = None, 'manual_unlocatable'
        out.append({"id": core + '-q3-multiregion', 'core_id': core,
                    **{key: row[key] for key in ('image', 'asset_category', 'anomaly_gt', 'defect_gt', 'normal_reference_image')},
                    'protocol': PROTOCOL, 'regions_gt': regions, 'normalized_bboxes_xywh': boxes,
                    'location_gt_known': regions is not None, 'label_basis': basis})
    return out


def parse_json(output, key):
    text = str(output).strip()
    fence = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        def unique_pairs(pairs):
            result = {}
            for k, v in pairs:
                if k in result:
                    raise ValueError("duplicate JSON key")
                result[k] = v
            return result
        obj = json.loads(text, object_pairs_hook=unique_pairs)
        if not isinstance(obj, dict) or set(obj) != {key}:
            return None
        value = obj[key]
        if key == 'present':
            return value if type(value) is bool else None
        if not isinstance(value, list) or any(type(v) is not int or not 1 <= v <= 9 for v in value) or len(set(value)) != len(value):
            return None
        return sorted(value)
    except (ValueError, TypeError):
        return None


def prompt(stage='regions', normal_reference=True):
    # Deliberately accepts no row/GT argument: inference cannot see labels/boxes/type/path.
    context = ('The FIRST image is a normal reference of the same asset type. The SECOND image is the TARGET. '
               'Different lighting/viewpoint alone is not a defect. ' if normal_reference else 'The image is the TARGET. ')
    context += 'Inspect only visible defects in the TARGET: corrosion, missing insulator components, or bird nests. '
    if stage == 'presence':
        return context + 'Does the TARGET contain a visible defect? Return only JSON {"present":true} or {"present":false}.'
    if stage != 'regions':
        raise ValueError(stage)
    return (context + 'Use the entire original TARGET frame as an equal 3 by 3 grid: ' + GRID + '. '
            'Independently locate every visible defect site. For each site choose the cell containing its center, '
            'list each cell only once, and do not average the locations of separate defects. '
            'Re-check the image itself; if no visible defect is present return an empty list. '
            'Return only JSON {"regions":[1,9]} using the applicable integers, or {"regions":[]} for no defect.')


def ratio(a, b):
    return a / b if b else None


def score(rows, predictions):
    n, a = len(rows), sum(r['anomaly_gt'] == 'anomalous' for r in rows)
    counts = Counter(tp=0, fp=0, tn=0, fn=0, invalid_normal=0, invalid_anomalous=0,
                     false_none=0, normal_false_cells=0, region_tp=0, region_fp=0, region_fn=0,
                     exact_positive=0, exact_known=0)
    known = positive_known = 0
    for row in rows:
        pred = predictions.get(row['core_id'])
        if pred is not None and (not isinstance(pred, list) or any(type(v) is not int or not 1 <= v <= 9 for v in pred) or len(pred) != len(set(pred))):
            raise ValueError("unparsed prediction")
        valid, positive = pred is not None, row['anomaly_gt'] == 'anomalous'
        if not valid:
            counts['invalid_anomalous' if positive else 'invalid_normal'] += 1
            counts['fn' if positive else 'fp'] += 1
        else:
            counts['tp' if pred and positive else 'fp' if pred else 'fn' if positive else 'tn'] += 1
            counts['false_none'] += positive and not pred
            counts['normal_false_cells'] += len(pred) if not positive else 0
        gt = row['regions_gt']
        if gt is None:
            continue
        known += 1
        positive_known += positive
        pset, gset = set(pred or []), set(gt)
        counts['exact_known'] += valid and pset == gset
        if positive:
            counts['exact_positive'] += valid and pset == gset
        counts['region_tp'] += len(pset & gset)
        counts['region_fp'] += len(pset - gset)
        counts['region_fn'] += len(gset - pset)
    recall, specificity = ratio(counts['tp'], a), ratio(counts['tn'], n-a)
    tp, fp, fn = (counts[k] for k in ('region_tp', 'region_fp', 'region_fn'))
    return dict(images=n, anomalous_images=a, normal_images=n-a, **dict(counts),
                presence_recall=recall, presence_specificity=specificity,
                presence_balanced_accuracy=(recall+specificity)/2 if recall is not None and specificity is not None else None,
                invalid_rate=ratio(counts['invalid_normal']+counts['invalid_anomalous'], n),
                spatial_known_images=known, spatial_known_anomalies=positive_known, spatial_unknown_images=n-known,
                anomaly_exact_set_accuracy=ratio(counts['exact_positive'],positive_known),
                all_known_exact_set_accuracy=ratio(counts['exact_known'],known),
                region_micro_precision=ratio(tp,tp+fp), region_micro_recall=ratio(tp,tp+fn),
                region_micro_f1=ratio(2*tp,2*tp+fp+fn))


def reports(rows, predictions):
    by_group = []
    for key in ('asset_category', 'defect_gt', 'label_basis'):
        for value in sorted({r[key] for r in rows}):
            by_group.append(dict(group=key, value=value, **score([r for r in rows if r[key]==value], predictions)))
    controls = {name: score(rows, {r['core_id']: pred for r in rows})
                for name,pred in (('always_none',[]),('always_center',[5]),('always_all_cells',list(range(1,10))))}
    per_cell=[]
    for cell in range(1,10):
        tp=fp=fn=tn=invalid=0
        for row in rows:
            gt=row['regions_gt']
            if gt is None:
                continue
            pred=predictions.get(row['core_id'])
            if pred is None:
                invalid+=1
                fn+=cell in gt
            elif cell in pred:
                if cell in gt:tp+=1
                else:fp+=1
            elif cell in gt:fn+=1
            else:tn+=1
        per_cell.append(dict(cell=cell,region=NAMES[cell-1],tp=tp,fp=fp,fn=fn,tn=tn,
                             invalid_images=invalid,precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn)))
    per_image=[]
    for row in rows:
        gt,pred=row['regions_gt'],predictions.get(row['core_id'])
        per_image.append(dict(core_id=row['core_id'],image=row['image'],asset_category=row['asset_category'],
            defect_gt=row['defect_gt'],regions_gt=gt,predicted_regions=pred,valid=pred is not None,
            hit_regions=sorted(set(gt)&set(pred or [])) if gt is not None else None,
            missed_regions=sorted(set(gt)-set(pred or [])) if gt is not None else None,
            false_regions=sorted(set(pred or [])-set(gt)) if gt is not None and pred is not None else None))
    return dict(overall=score(rows,predictions), by_group=by_group, per_cell=per_cell,
                per_image=per_image,constant_baselines=controls)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--master-qa',type=Path,required=True)
    p.add_argument('--loc-qa',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args(argv)
    rows=build_dataset(read_jsonl(args.master_qa),read_jsonl(args.loc_qa),json.loads(args.annotations.read_text(encoding='utf-8')))
    lock={'protocol':PROTOCOL,'dataset_hash':digest(rows),'sources':{
        name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in
        (('master',args.master_qa),('location',args.loc_qa),('annotations',args.annotations))}}
    lockpath=args.out_dir/'input_lock.json'
    if lockpath.exists() and json.loads(lockpath.read_text(encoding='utf-8'))!=lock:
        raise ValueError('input changed; use a new output folder')
    atomic_json(lockpath,lock)
    atomic_json(args.out_dir/'dataset.json',rows)
    summary={'protocol':PROTOCOL,'images':len(rows),'by_basis':dict(Counter(r['label_basis'] for r in rows)),
             'positive_region_count_distribution':dict(Counter(len(r['regions_gt']) for r in rows if r['regions_gt'])),
             'multi_region_anomalies':sum(len(r['regions_gt'] or [])>1 for r in rows),
             'inference_performed':False}
    atomic_json(args.out_dir/'build_summary.json',summary)
    atomic_json(args.out_dir/'constant_baselines.json',reports(rows,{})['constant_baselines'])
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())

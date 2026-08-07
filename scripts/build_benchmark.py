#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from insplad_benchmark.dataset import (
    aligned_core_rows,
    all_assets,
    build_anomaly_qa,
    build_defect_qa,
    category_intersection,
    core_link_rows,
    cross_split_duplicate_candidates,
    defect_classes_by_asset,
    exact_overlap_candidates,
    load_overlap_confirmation,
    normalize_defect,
    normalize_key,
    parse_annotation_previews,
    parse_supervised_manifest,
    parse_unsupervised_manifest,
    select_reference_by_asset,
    stratified_core_sample,
    stratified_sample,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def label_counts(unsup: list[dict], sup: list[dict]) -> list[dict]:
    counts = Counter()
    for r in unsup:
        counts[('unsupervised_anomaly_detection', r['asset_category'], r['split'], r['source_label_raw'], r['label'])] += 1
    for r in sup:
        counts[('supervised_fault_classification', r['asset_category'], r['split'], r['source_defect_raw'], r['defect_class'])] += 1
    return [
        {'dataset': d, 'asset_category': a, 'split': s, 'raw_label': raw, 'normalized_label': norm, 'count': n}
        for (d, a, s, raw, norm), n in sorted(counts.items())
    ]


def annotation_section(annotation_info: dict | None) -> list[str]:
    if not annotation_info:
        return [
            '- Annotation contents were not supplied to this build.',
            '- Run `inspect_zip_manifest.py --text-preview-json ...` and pass `--annotation-previews` to verify schema.',
        ]
    lines = [
        f"- Verified line schema: `{annotation_info['line_schema']}`.",
        '- The integer is an asset-local class ID, **not** a universal binary anomaly label.',
        '- Defect folder names remain the authoritative semantic labels for benchmark GT.',
        '',
        'Observed preview mappings:',
    ]
    for asset, info in sorted(annotation_info['by_asset'].items()):
        mapping = info['raw_defect_to_class_ids']
        rendered = ', '.join(f"`{k}`→{v}" for k, v in mapping.items()) or '(no parseable lines)'
        suffix = ' (preview truncated)' if info['truncated'] else ''
        lines.append(f'- `{asset}`: {rendered}{suffix}')
    return lines


def structure_report(unsup, sup, annotations, shared_assets, eval_split, reason, annotation_info, core_count) -> str:
    u = defaultdict(lambda: defaultdict(set)); s = defaultdict(lambda: defaultdict(set))
    for r in unsup: u[r['asset_category']][r['split']].add(r['source_label_raw'])
    for r in sup: s[r['asset_category']][r['split']].add(r['source_defect_raw'])
    lines = ['# InsPLAD fault dataset structure (verified)', '', '## Track A source — `unsupervised_anomaly_detection.zip`', '', '```text', 'unsupervised_anomaly_detection/']
    for asset in sorted(u):
        lines.append(f'├─ {asset}/')
        for split in sorted(u[asset]): lines.append(f"│  ├─ {split}/  [{', '.join(sorted(u[asset][split]))}]")
    lines += ['```', '', f'- Asset categories: **{len(u)}**', f"- Train images: **{sum(r['split'] == 'train' for r in unsup):,}**", f"- Test images: **{sum(r['split'] == 'test' for r in unsup):,}**", '- Track A evaluates **all 17 assets on official test**; train normals are used only as one-shot reference candidates.', '', '## Track B source — `supervised_fault_classification.zip`', '', '```text', 'defect_supervised/']
    for asset in sorted(s):
        lines.append(f'├─ {asset}/')
        for split in sorted(s[asset]): lines.append(f"│  ├─ {split}/  [{', '.join(sorted(s[asset][split]))}]")
        lines.append('│  └─ annotations/image_list_val.txt')
    lines += ['```', '', f'- Asset categories: **{len(s)}**', f"- Train images: **{sum(r['split'] == 'train' for r in sup):,}**", f"- Val images: **{sum(r['split'] == 'val' for r in sup):,}**", '- No supervised test split was observed.', f'- Structured evaluation split: `{eval_split}` ({reason}).', f'- Annotation files: **{len(annotations)}**.', '', '## Annotation schema', '', *annotation_section(annotation_info), '', '## Shared asset categories', '', *[f'- `{x}`' for x in shared_assets], '', '## Aligned Core', '', '- Definition: `unsupervised test ∩ supervised val ∩ exact SHA256 match ∩ binary GT agreement`.', f'- Clean aligned images: **{core_count:,}**.', '- This core links zero-shot, one-shot, and structured predictions for the **same image bytes**.', '']
    return '\n'.join(lines)


def sample_review(sections: list[tuple[str, list[dict]]], seed: int) -> str:
    lines = ['# QA sample review', '', f'Deterministic stratified review sample (`seed={seed}`).', '']
    for title, rows in sections:
        lines += [f'## {title}', '']
        for i, row in enumerate(rows, 1):
            asset = row.get('asset_category', ''); row_id = row.get('id') or row.get('core_id')
            lines.append(f"{i}. `{row_id}` — **{asset}**")
            if row.get('image'): lines.append(f"   - image: `{row['image']}`")
            if row.get('unsupervised_image'):
                lines.append(f"   - unsupervised image: `{row['unsupervised_image']}`"); lines.append(f"   - supervised image: `{row['supervised_image']}`")
            if row.get('reference_image'): lines.append(f"   - reference: `{row['reference_image']}`")
            if 'answer' in row: lines.append(f"   - GT: `{json.dumps(row['answer'], ensure_ascii=False)}`")
            elif 'anomaly_answer' in row:
                lines.append(f"   - anomaly GT: `{row['anomaly_answer']}`"); lines.append(f"   - defect GT: `{row['defect_class']}`")
        lines.append('')
    return '\n'.join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--anomaly-manifest', type=Path, required=True); p.add_argument('--fault-manifest', type=Path, required=True)
    p.add_argument('--overlap-summary', type=Path, help='JSON emitted by hash_overlap.py'); p.add_argument('--annotation-previews', type=Path, help='JSON emitted by inspect_zip_manifest.py --text-preview-json')
    p.add_argument('--repo-root', type=Path, default=ROOT); p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(); root = args.repo_root
    unsup = parse_unsupervised_manifest(args.anomaly_manifest); sup, annotations = parse_supervised_manifest(args.fault_manifest)
    track_a_assets = all_assets(unsup); track_b_assets = all_assets(sup); shared_assets = category_intersection(unsup, sup)
    track_a_test_labels = defaultdict(set)
    for row in unsup:
        if row['split'] == 'test': track_a_test_labels[row['asset_category']].add(row['label'])
    track_a_both_class_assets = sorted(a for a, labels in track_a_test_labels.items() if labels == {'normal', 'anomalous'})
    track_a_normal_only_assets = sorted(a for a, labels in track_a_test_labels.items() if labels == {'normal'})
    refs = select_reference_by_asset(unsup, track_a_assets, seed=args.seed)
    zero = build_anomaly_qa(unsup, track_a_assets); one = build_anomaly_qa(unsup, track_a_assets, references=refs); defect, eval_split, reason = build_defect_qa(sup)
    overlap = exact_overlap_candidates(sup, unsup); overlap_confirmation = None; exact_confirmed = False
    if args.overlap_summary:
        overlap_confirmation = load_overlap_confirmation(args.overlap_summary, len(overlap)); exact_confirmed = bool(overlap_confirmation['all_candidates_sha256_confirmed'])
    clean_core, conflicts = aligned_core_rows(overlap, exact_sha256_confirmed=exact_confirmed); core_links = core_link_rows(clean_core, refs) if clean_core else []
    annotation_info = parse_annotation_previews(args.annotation_previews) if args.annotation_previews else None
    write_jsonl(root/'data/qa/anomaly_zero_shot.jsonl', zero); write_jsonl(root/'data/qa/anomaly_one_shot.jsonl', one); write_jsonl(root/'data/qa/defect_qa.jsonl', defect)
    zero_sample = stratified_sample(zero,20,seed=args.seed); one_by_id={r['id']:r for r in one}; one_sample=[one_by_id[r['id']] for r in zero_sample]; defect_sample=stratified_sample(defect,20,seed=args.seed); core_sample=stratified_core_sample(core_links,20,seed=args.seed) if core_links else []
    write_jsonl(root/'samples/anomaly_zero_shot_20.jsonl',zero_sample); write_jsonl(root/'samples/anomaly_one_shot_20.jsonl',one_sample); write_jsonl(root/'samples/defect_20.jsonl',defect_sample); write_jsonl(root/'samples/aligned_core_20.jsonl',core_sample)
    write_csv(root/'reports/category_counts.csv',label_counts(unsup,sup)); write_csv(root/'reports/common_categories.csv',[{'asset_category':a,'in_unsupervised':True,'in_supervised':True} for a in shared_assets])
    overlap_rows=[]
    for row in overlap:
        out=dict(row); out['exact_sha256_match']=exact_confirmed; overlap_rows.append(out)
    write_csv(root/'reports/image_overlap.csv',overlap_rows); write_csv(root/'reports/aligned_core.csv',core_links); write_csv(root/'reports/aligned_label_conflicts.csv',conflicts)
    split_counts=Counter((r['unsupervised_split'],r['supervised_split']) for r in overlap); write_csv(root/'reports/overlap_split_counts.csv',[{'unsupervised_split':u,'supervised_split':s,'count':n} for (u,s),n in sorted(split_counts.items())])
    core_counts=[]
    for asset in sorted({r['asset_category'] for r in core_links}):
        rows=[r for r in core_links if r['asset_category']==asset]; core_counts.append({'asset_category':asset,'total':len(rows),'normal':sum(r['anomaly_answer']=='normal' for r in rows),'anomalous':sum(r['anomaly_answer']=='anomalous' for r in rows)})
    write_csv(root/'reports/aligned_core_counts.csv',core_counts)
    if overlap_confirmation: (root/'reports/image_overlap_summary.json').write_text(json.dumps(overlap_confirmation,ensure_ascii=False,indent=2),encoding='utf-8')
    if annotation_info:
        lines=['# Supervised annotation schema','',f"- Line schema: `{annotation_info['line_schema']}`",'- The trailing integer is an asset-local classification class ID, not a binary anomaly label.','- Benchmark GT is derived from the defect folder name.','','## Observed class IDs from local previews','']
        for asset,info in sorted(annotation_info['by_asset'].items()):
            mapping=', '.join(f"`{k}` → {v}" for k,v in info['raw_defect_to_class_ids'].items()); suffix=' (preview truncated)' if info['truncated'] else ''; lines.append(f'- `{asset}`: {mapping}{suffix}')
        lines += ['', 'The `vari-grip` mapping (`bird-nest`→0, `good`→1, `rust`→2) proves that 0/1 cannot be interpreted globally as normal/anomalous.', '']; (root/'reports/annotation_schema.md').write_text('\n'.join(lines),encoding='utf-8')
    (root/'reports/dataset_structure.md').write_text(structure_report(unsup,sup,annotations,shared_assets,eval_split,reason,annotation_info,len(core_links)),encoding='utf-8')
    (root/'reports/sample_review.md').write_text(sample_review([('Track A — Anomaly zero-shot',zero_sample),('Track A — Anomaly one-shot',one_sample),('Track B — Structured explainability',defect_sample),('Aligned Core — same-image linkage',core_sample)],args.seed),encoding='utf-8')
    u_leak=cross_split_duplicate_candidates(unsup); s_leak=cross_split_duplicate_candidates(sup)
    issue_lines=['# Data issues and benchmark decisions','', '## Benchmark tracks','',f'- Track A uses all **{len(track_a_assets)}** unsupervised assets and official test (**{len(zero):,}** images).',f'- Track A assets with both normal/anomalous test GT: **{len(track_a_both_class_assets)}**.',f"- Track A normal-only test assets: **{len(track_a_normal_only_assets)}** — {', '.join(f'`{a}`' for a in track_a_normal_only_assets)}.",f'- Track B uses all **{len(track_b_assets)}** supervised assets and official val (**{len(defect):,}** images) because no test split exists.',f'- Shared asset categories: **{len(shared_assets)}**.','','## Exact cross-dataset overlap','',f'- Manifest overlap candidates: **{len(overlap):,}**.',f"- SHA256 confirmed matches: **{int(overlap_confirmation['sha256_match_count']):,}**." if overlap_confirmation else '- SHA256 confirmation summary not supplied.',f"- SHA256 mismatches: **{int(overlap_confirmation['sha256_mismatch_count']):,}**." if overlap_confirmation else '',f"- `unsupervised test ↔ supervised val`: **{split_counts[('test','val')]:,}** exact SHA256 pairs." if exact_confirmed else f"- `unsupervised test ↔ supervised val`: **{split_counts[('test','val')]:,}** manifest candidates.",f'- Binary GT conflicts in that aligned split: **{len(conflicts):,}**.',f'- Clean Aligned Core: **{len(core_links):,}** images.','','The Aligned Core excludes exact same images whose binary normal/anomalous GT disagrees between the two dataset variants.','','## Label policy','','- Source normal labels `good` and `normal` are normalized to benchmark `normal`.','- `corrosão` is mechanically normalized to `corrosao` and is NOT merged with `rust`.']
    if annotation_info: issue_lines += ['- `image_list_val.txt` line schema is `<relative_image_path> <integer_class_id>`.','- Annotation integer IDs are asset-local class IDs, not binary anomaly values.','- Example proving this: `vari-grip` preview maps `bird-nest→0`, `good→1`, `rust→2`.','- Therefore benchmark semantic GT is derived from defect folder names, not the numeric annotation ID.']
    issue_lines += ['','## Leakage checks','',f'- Unsupervised cross-split duplicate candidates: **{len(u_leak)}**.',f'- Supervised cross-split duplicate candidates: **{len(s_leak)}**.','- One-shot reference images come only from same-asset unsupervised train normals, seed 42.','- A train reference is excluded when its size+ZIP CRC32 content fingerprint occurs in unsupervised test.']
    if conflicts:
        issue_lines += ['','## Aligned binary GT conflicts']
        for row in conflicts: issue_lines.append(f"- `{row['filename']}` — `{row['asset_category']}`: unsupervised `{row['unsupervised_label_raw']}`→{row['unsupervised_anomaly']} vs supervised `{row['supervised_defect_raw']}`→{row['supervised_anomaly']}")
    (root/'reports/data_issues.md').write_text('\n'.join(x for x in issue_lines if x!='')+'\n',encoding='utf-8')
    (root/'configs').mkdir(parents=True,exist_ok=True)
    category_map={'raw_to_normalized':{x:normalize_key(x) for x in sorted({r['asset_category_raw'] for r in unsup}|{r['asset_category_raw'] for r in sup})},'track_a_assets':track_a_assets,'track_b_assets':track_b_assets,'shared_assets':shared_assets,'aligned_core_definition':'unsupervised test ∩ supervised val ∩ exact SHA256 match ∩ binary GT agreement'}
    defect_map={'raw_to_normalized':{x:normalize_defect(x) for x in sorted({r['source_defect_raw'] for r in sup})},'valid_by_asset':defect_classes_by_asset(sup),'normal_source_labels':sorted({r['source_defect_raw'] for r in sup if r['defect_class']=='normal'})}
    if annotation_info:
        defect_map['annotation_line_schema']=annotation_info['line_schema']; defect_map['annotation_class_ids_by_asset_preview']={a:i['raw_defect_to_class_ids'] for a,i in annotation_info['by_asset'].items()}; defect_map['annotation_class_id_policy']=annotation_info['gt_policy']
    (root/'configs/category_mapping.json').write_text(json.dumps(category_map,ensure_ascii=False,indent=2),encoding='utf-8'); (root/'configs/defect_mapping.json').write_text(json.dumps(defect_map,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={'track_a':{'dataset':'unsupervised_anomaly_detection','assets':track_a_assets,'evaluation_split':'test','zero_shot_qa_count':len(zero),'one_shot_qa_count':len(one),'assets_with_both_classes':track_a_both_class_assets,'normal_only_assets':track_a_normal_only_assets},'track_b':{'dataset':'supervised_fault_classification','assets':track_b_assets,'evaluation_split':eval_split,'structured_qa_count':len(defect)},'shared_assets':shared_assets,'exact_overlap_count':int(overlap_confirmation['sha256_match_count']) if overlap_confirmation else None,'aligned_test_val_exact_count':split_counts[('test','val')] if exact_confirmed else None,'aligned_binary_gt_conflict_count':len(conflicts) if exact_confirmed else None,'aligned_core_count':len(core_links) if exact_confirmed else None,'references':{a:r['relative_path'] for a,r in refs.items()}}
    (root/'reports/build_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0

if __name__=='__main__': raise SystemExit(main())

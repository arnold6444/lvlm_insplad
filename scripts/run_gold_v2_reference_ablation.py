#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))

import run_gold_v2_baselines as v2  # noqa: E402

base = v2.base
CONDITIONS = ('normal_reference', 'positive_reference', 'normal_positive_reference')


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def prompt_for_condition(row: dict, condition: str) -> str:
    options = row.get('options')
    if not isinstance(options, dict) or not options:
        raise ValueError(f"QA row {row.get('id')} has no options")
    option_lines = '\n'.join(f'{key}. {value}' for key, value in options.items())

    if condition == 'normal_reference':
        context = (
            'Image 1 is a verified NORMAL reference of the same asset category. '
            'Image 2 is the TARGET image. Use Image 1 only as the normal visual standard and answer about Image 2.'
        )
    elif condition == 'positive_reference':
        context = (
            'Image 1 is a retrieved verified ANOMALOUS reference from the supervised TRAIN split of the same asset category. '
            'It was selected only by visual similarity without using the target ground-truth label. '
            'Image 2 is the TARGET image. Use Image 1 only as an anomalous visual example and answer about Image 2.'
        )
    elif condition == 'normal_positive_reference':
        context = (
            'Image 1 is a verified NORMAL reference of the same asset category. '
            'Image 2 is a retrieved verified ANOMALOUS reference from the supervised TRAIN split of the same asset category, '
            'selected only by visual similarity without using target ground truth. Image 3 is the TARGET image. '
            'Use the first two images only as visual references and answer about Image 3.'
        )
    else:
        raise ValueError(f'unknown condition: {condition}')

    return (
        'You are an industrial visual inspector. '
        + context
        + ' Do not use hidden metadata or external knowledge.\n\n'
        + f"Question: {row['question']}\n{option_lines}\n\n"
        + 'Return exactly one option letter and nothing else.'
    )


def build_reference_manifest(rows: list[dict], positive_rows: list[dict[str, str]]) -> list[dict]:
    positive_by_core = {row['core_id']: row for row in positive_rows}
    by_core: dict[str, dict] = {}
    for row in rows:
        core_id = str(row['core_id'])
        positive = positive_by_core.get(core_id)
        if positive is None:
            raise RuntimeError(f'{core_id}: positive reference missing')
        current = {
            'core_id': core_id,
            'asset_category': str(row['asset_category']),
            'target_image': str(row['image']),
            'normal_reference_image': str(row.get('normal_reference_image') or ''),
            'positive_reference_image': str(positive['positive_reference_image']),
            'positive_reference_defect_label': str(positive.get('positive_reference_defect_label') or ''),
            'positive_reference_similarity': str(positive.get('cosine_similarity') or ''),
            'target_gt_used_for_positive_selection': str(positive.get('target_gt_used_for_selection') or '').lower(),
        }
        if not current['normal_reference_image']:
            raise RuntimeError(f'{core_id}: normal reference missing')
        if current['target_gt_used_for_positive_selection'] not in {'false', '0'}:
            raise RuntimeError(f'{core_id}: positive selection is not leakage-safe')
        previous = by_core.get(core_id)
        if previous is not None and previous != current:
            raise RuntimeError(f'{core_id}: inconsistent QA/reference metadata')
        by_core[core_id] = current
    return [by_core[key] for key in sorted(by_core)]


def manifest_identity(rows: list[dict]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def prepare_images(
    rows: list[dict],
    manifest: list[dict],
    sup_zip: Path,
    unsup_zip: Path,
    root: Path,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    target_paths: dict[str, Path] = {}
    normal_paths: dict[str, Path] = {}
    positive_paths: dict[str, Path] = {}
    manifest_by_core = {row['core_id']: row for row in manifest}

    for row in rows:
        core = manifest_by_core[str(row['core_id'])]
        target = str(row['image'])
        normal = core['normal_reference_image']
        positive = core['positive_reference_image']
        if target not in target_paths:
            target_paths[target] = base.safe_extract(sup_zip, target, root / 'target')
        if normal not in normal_paths:
            normal_paths[normal] = base.safe_extract(unsup_zip, normal, root / 'normal')
        if positive not in positive_paths:
            positive_paths[positive] = base.safe_extract(sup_zip, positive, root / 'positive')
    return target_paths, normal_paths, positive_paths


def image_list_for(
    row: dict,
    condition: str,
    manifest_by_core: dict[str, dict],
    target_paths: dict[str, Path],
    normal_paths: dict[str, Path],
    positive_paths: dict[str, Path],
) -> list[Path]:
    core = manifest_by_core[str(row['core_id'])]
    target = target_paths[str(row['image'])]
    normal = normal_paths[core['normal_reference_image']]
    positive = positive_paths[core['positive_reference_image']]
    if condition == 'normal_reference':
        return [normal, target]
    if condition == 'positive_reference':
        return [positive, target]
    if condition == 'normal_positive_reference':
        return [normal, positive, target]
    raise ValueError(condition)


def infer_luxia(images: list[Path], prompt: str, api_key: str) -> tuple[str, float]:
    from insplad_benchmark.luxia_adapter import infer_luxia_pipeline

    started = time.time()
    result = infer_luxia_pipeline(images, prompt, api_key)
    return str(result['output']), round(time.time() - started, 3)


def run_model(
    model_key: str,
    condition: str,
    rows: list[dict],
    manifest_by_core: dict[str, dict],
    target_paths: dict[str, Path],
    normal_paths: dict[str, Path],
    positive_paths: dict[str, Path],
    out_dir: Path,
    checkpoint_every: int,
    max_attempts_per_row: int = 0,
) -> dict:
    spec = base.MODEL_REGISTRY[model_key]
    pred_path = out_dir / condition / 'predictions' / f'{model_key}.jsonl'
    existing_rows = base.read_jsonl(pred_path) if pred_path.is_file() else []
    existing = {str(row['id']): dict(row) for row in existing_rows if row.get('id')}
    predictions_by_id: dict[str, dict] = dict(existing)

    def ordered_predictions() -> list[dict]:
        # Always checkpoint the complete known state, not only the processed prefix.
        # This preserves reusable tail rows if Colab stops mid-resume.
        return [
            predictions_by_id[str(row['id'])]
            for row in rows
            if str(row['id']) in predictions_by_id
        ]

    adapter = None
    if spec['backend'] == 'hf':
        from insplad_benchmark.hf_adapters import load_hf_adapter
        adapter = load_hf_adapter(spec)

    api_key = os.environ.get('LUXIA_API_KEY', '')
    if spec['backend'] == 'luxia' and not api_key:
        raise RuntimeError('LUXIA_API_KEY is required for Luxia')

    called = reused = errors = exhausted = 0
    total_latency = 0.0
    try:
        for index, row in enumerate(rows, start=1):
            row_id = str(row['id'])
            old = existing.get(row_id)
            if base.reusable_prediction(old):
                reused += 1
                continue

            prior_attempts = 0
            if old is not None:
                try:
                    prior_attempts = int(old.get('attempt_count', 1))
                except (TypeError, ValueError):
                    prior_attempts = 1

            if old is not None and bool(old.get('retry_exhausted')):
                exhausted += 1
                continue

            if max_attempts_per_row > 0 and prior_attempts >= max_attempts_per_row:
                frozen = dict(old or {})
                frozen['retry_exhausted'] = True
                frozen['attempt_count'] = prior_attempts
                predictions_by_id[row_id] = frozen
                exhausted += 1
                continue

            prompt = prompt_for_condition(row, condition)
            images = image_list_for(
                row, condition, manifest_by_core,
                target_paths, normal_paths, positive_paths,
            )
            started = time.time()
            output = ''
            error = None
            try:
                if spec['backend'] == 'hf':
                    from insplad_benchmark.hf_adapters import infer_hf
                    output = infer_hf(adapter, images, prompt)
                    latency = round(time.time() - started, 3)
                else:
                    output, latency = infer_luxia(images, prompt, api_key)
            except Exception as exc:
                latency = round(time.time() - started, 3)
                error = f'{type(exc).__name__}: {exc}'
                errors += 1

            valid_letters = {str(key).upper() for key in row['options']}
            choice = base.parse_choice(output, valid_letters)
            core = manifest_by_core[str(row['core_id'])]
            attempt_count = prior_attempts + 1
            retry_exhausted = bool(
                error
                and max_attempts_per_row > 0
                and attempt_count >= max_attempts_per_row
            )
            predictions_by_id[row_id] = {
                'id': row['id'],
                'core_id': row['core_id'],
                'asset_category': row['asset_category'],
                'question_type': row['question_type'],
                'condition': condition,
                'output': output,
                'predicted_choice': choice,
                'latency_s': latency,
                'error': error,
                'normal_reference_image': core['normal_reference_image'],
                'positive_reference_image': core['positive_reference_image'],
                'attempt_count': attempt_count,
                'retry_exhausted': retry_exhausted,
            }
            called += 1
            total_latency += latency
            if checkpoint_every > 0 and called % checkpoint_every == 0:
                base.write_jsonl(pred_path, ordered_predictions())
                print(f'[{condition}/{model_key}] {index}/{len(rows)} checkpoint', flush=True)
        base.write_jsonl(pred_path, ordered_predictions())
    finally:
        if adapter is not None:
            del adapter
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return {
        'condition': condition,
        'model_key': model_key,
        'called_this_run': called,
        'reused_predictions': reused,
        'row_errors': errors,
        'retry_exhausted_rows': exhausted + sum(
            bool(row.get('retry_exhausted'))
            for row in predictions_by_id.values()
            if not base.reusable_prediction(row)
        ),
        'called_latency_s': round(total_latency, 3),
    }


def evaluate_condition(model_key: str, condition: str, rows: list[dict], pred_path: Path) -> tuple[dict, list[dict]]:
    metric, by_asset = base.evaluate_model(model_key, rows, pred_path)
    metric['condition'] = condition
    q3_counts = Counter(
        str(row['asset_category'])
        for row in rows
        if row['question_type'] == 'defect_localization'
    )
    for item in by_asset:
        item['condition'] = condition
        count = q3_counts.get(str(item['asset_category']), 0)
        item['q3_questions'] = count
        if count == 0:
            item['q3_accuracy'] = None
    return metric, by_asset


def build_ablation_comparison(reference_rows: list[dict], baseline_path: Path | None) -> tuple[list[dict], list[dict]]:
    all_rows: list[dict] = []
    baseline_by_model: dict[str, dict] = {}
    if baseline_path and baseline_path.is_file():
        for row in read_csv(baseline_path):
            item = dict(row)
            item['condition'] = 'image_only'
            all_rows.append(item)
            baseline_by_model[str(item['model_key'])] = item
    all_rows.extend(reference_rows)

    deltas: list[dict] = []
    for row in reference_rows:
        baseline = baseline_by_model.get(str(row['model_key']))
        if baseline is None:
            continue
        item = {
            'model_key': row['model_key'],
            'condition': row['condition'],
        }
        for metric in ('overall_accuracy', 'anomaly_f1', 'defect_macro_f1', 'localization_accuracy'):
            try:
                item[f'{metric}_baseline'] = float(baseline[metric])
                item[f'{metric}_reference'] = float(row[metric])
                item[f'{metric}_delta'] = round(float(row[metric]) - float(baseline[metric]), 6)
            except (KeyError, TypeError, ValueError):
                item[f'{metric}_baseline'] = ''
                item[f'{metric}_reference'] = ''
                item[f'{metric}_delta'] = ''
        deltas.append(item)
    return all_rows, deltas


def publish_results(out_dir: Path) -> None:
    token = os.environ.get('GITHUB_TOKEN', '').strip()
    if not token:
        print('[github-publish skipped] GITHUB_TOKEN is missing')
        return
    try:
        path = ROOT / 'scripts' / 'publish_results_to_github.py'
        spec = importlib.util.spec_from_file_location('publish_results_to_github', path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        result = module.publish(
            out_dir,
            repo='arnold6444/lvlm_insplad',
            branch='main',
            prefix='gold_v2_reference',
            token=token,
            filenames=(
                'leaderboard.csv',
                'metrics_by_asset.csv',
                'ablation_comparison.csv',
                'ablation_vs_baseline.csv',
                'reference_manifest.csv',
                'positive_reference_summary.csv',
                'positive_reference_metadata.json',
                'run_status.json',
                'run_metadata.json',
            ),
        )
        print(json.dumps({'github_publish': result}, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f'[github-publish skipped] {type(exc).__name__}: {exc}')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa-dir', type=Path, required=True)
    parser.add_argument('--sup-zip', type=Path, required=True)
    parser.add_argument('--unsup-zip', type=Path, required=True)
    parser.add_argument('--positive-manifest', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--baseline-leaderboard', type=Path)
    parser.add_argument('--conditions', nargs='+', default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument(
        '--models', nargs='+',
        default=['qwen3_5_4b', 'internvl3_5_8b', 'minicpm_v_4_5', 'llava_onevision_7b'],
        choices=list(base.MODEL_REGISTRY),
    )
    parser.add_argument('--checkpoint-every', type=int, default=5)
    parser.add_argument(
        '--max-attempts-per-row',
        type=int,
        default=0,
        help=(
            'Maximum total attempts per QA row. 0 keeps legacy unlimited retry behavior. '
            'Legacy error rows without attempt_count are treated as having one prior attempt.'
        ),
    )
    args = parser.parse_args()
    if args.max_attempts_per_row < 0:
        parser.error('--max-attempts-per-row must be >= 0')

    rows = v2.load_gold_qa_v2(args.qa_dir)
    positive_rows = read_csv(args.positive_manifest)
    manifest = build_reference_manifest(rows, positive_rows)
    manifest_by_core = {row['core_id']: row for row in manifest}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base.write_csv(args.out_dir / 'reference_manifest.csv', manifest)

    for sibling in ('positive_reference_summary.csv', 'positive_reference_metadata.json'):
        source = args.positive_manifest.parent / sibling
        if source.is_file():
            (args.out_dir / sibling).write_bytes(source.read_bytes())

    snapshot = args.out_dir / 'run_qa.jsonl'
    if snapshot.is_file():
        existing = base.read_jsonl(snapshot)
        if base.qa_identity(existing) != base.qa_identity(rows):
            raise RuntimeError('out-dir contains a different frozen QA snapshot')
    else:
        base.write_jsonl(snapshot, rows)

    target_paths, normal_paths, positive_paths = prepare_images(
        rows, manifest, args.sup_zip, args.unsup_zip, args.out_dir / '_images'
    )

    run_status: list[dict] = []
    leaderboard: list[dict] = []
    by_asset_all: list[dict] = []
    for condition in args.conditions:
        for model_key in args.models:
            print(f'\n=== {condition} / {model_key} ===', flush=True)
            try:
                status = run_model(
                    model_key, condition, rows, manifest_by_core,
                    target_paths, normal_paths, positive_paths,
                    args.out_dir, args.checkpoint_every,
                    args.max_attempts_per_row,
                )
                metric, by_asset = evaluate_condition(
                    model_key, condition, rows,
                    args.out_dir / condition / 'predictions' / f'{model_key}.jsonl',
                )
                status['status'] = 'completed'
                leaderboard.append(metric)
                by_asset_all.extend(by_asset)
            except Exception as exc:
                status = {
                    'condition': condition,
                    'model_key': model_key,
                    'status': 'failed',
                    'error': f'{type(exc).__name__}: {exc}',
                }
                print(status['error'], flush=True)
            run_status.append(status)
            base.write_csv(args.out_dir / 'leaderboard.csv', leaderboard)
            base.write_csv(args.out_dir / 'metrics_by_asset.csv', by_asset_all)
            (args.out_dir / 'run_status.json').write_text(
                json.dumps(run_status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
            )

    comparison, deltas = build_ablation_comparison(leaderboard, args.baseline_leaderboard)
    base.write_csv(args.out_dir / 'ablation_comparison.csv', comparison)
    base.write_csv(args.out_dir / 'ablation_vs_baseline.csv', deltas)

    counts = Counter(str(row['question_type']) for row in rows)
    metadata = {
        'benchmark_version': 'gold_v2_final',
        'qa_dir': str(args.qa_dir),
        'qa_identity_sha256': base.qa_identity(rows),
        'reference_manifest_sha256': manifest_identity(manifest),
        'reference_core_count': len(manifest),
        'q1_count': counts['anomaly_detection'],
        'q2_count': counts['defect_classification'],
        'q3_count': counts['defect_localization'],
        'conditions': args.conditions,
        'models_requested': args.models,
        'positive_reference_policy': 'same-asset supervised-train anomalous bank, DINOv2 nearest neighbor; target GT unused',
        'normal_reference_policy': 'existing fixed verified normal reference stored in gold_v2 QA',
        'image_order': {
            'normal_reference': ['normal_reference', 'target'],
            'positive_reference': ['positive_reference', 'target'],
            'normal_positive_reference': ['normal_reference', 'positive_reference', 'target'],
        },
        'q3_scope_note': 'Q3 is evaluated only on the 42 available localization questions.',
        'max_attempts_per_row': args.max_attempts_per_row,
        'retry_policy_note': (
            'Successful rows are always reused. When max_attempts_per_row > 0, '
            'failed rows are retried only until that total-attempt cap and are then '
            'kept as invalid outputs with retry_exhausted=true.'
        ),
    }
    (args.out_dir / 'run_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )

    publish_results(args.out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

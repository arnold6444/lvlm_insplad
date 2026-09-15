#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
NORMAL_LABELS = {'good', 'normal', 'ok', 'healthy'}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalized_member(value: str) -> str:
    return str(value).replace('\\', '/').lstrip('/')


def load_gold_rows(qa_dir: Path) -> list[dict]:
    import importlib.util

    script = ROOT / 'scripts' / 'run_gold_v2_baselines.py'
    spec = importlib.util.spec_from_file_location('run_gold_v2_baselines', script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.load_gold_qa_v2(qa_dir)


def unique_core_rows(rows: list[dict]) -> list[dict]:
    by_core: dict[str, dict] = {}
    for row in rows:
        core_id = str(row['core_id'])
        current = {
            'core_id': core_id,
            'asset_category': str(row['asset_category']),
            'target_image': str(row['image']),
            'anomaly_gt': str(row.get('anomaly_gt', '')),
            'defect_gt': str(row.get('defect_gt', '')),
        }
        previous = by_core.get(core_id)
        if previous is not None and previous != current:
            raise RuntimeError(f'{core_id}: inconsistent QA metadata')
        by_core[core_id] = current
    return [by_core[key] for key in sorted(by_core)]


def parse_supervised_member(name: str, valid_assets: set[str]) -> dict | None:
    path = PurePosixPath(normalized_member(name))
    if path.suffix.lower() not in IMAGE_EXTS:
        return None
    parts = path.parts
    try:
        idx = parts.index('defect_supervised')
    except ValueError:
        return None
    if len(parts) <= idx + 4:
        return None
    asset, split, label = parts[idx + 1], parts[idx + 2], parts[idx + 3]
    if asset not in valid_assets or split.lower() != 'train':
        return None
    if label.lower() in NORMAL_LABELS:
        return None
    return {
        'asset_category': asset,
        'defect_label': label,
        'member': normalized_member(name),
    }


def build_positive_bank(zip_path: Path, assets: set[str]) -> list[dict]:
    bank: list[dict] = []
    with ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            parsed = parse_supervised_member(info.filename, assets)
            if parsed:
                bank.append(parsed)
    if not bank:
        raise RuntimeError('no supervised train anomalous images found in ZIP')
    return sorted(bank, key=lambda row: (row['asset_category'], row['member']))


def zip_index(zip_path: Path) -> dict[str, str]:
    with ZipFile(zip_path) as archive:
        out: dict[str, str] = {}
        for name in archive.namelist():
            key = normalized_member(name)
            if key in out and out[key] != name:
                raise RuntimeError(f'ambiguous ZIP member: {key}')
            out[key] = name
    return out


def embed_members(
    zip_path: Path,
    members: list[str],
    *,
    model_id: str,
    batch_size: int,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    index = zip_index(zip_path)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    unique = sorted(set(normalized_member(member) for member in members))
    vectors: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}

    with ZipFile(zip_path) as archive:
        for start in range(0, len(unique), batch_size):
            batch = unique[start:start + batch_size]
            images = []
            for member in batch:
                actual = index.get(member)
                if actual is None:
                    raise FileNotFoundError(f'ZIP member not found: {member}')
                payload = archive.read(actual)
                hashes[member] = hashlib.sha256(payload).hexdigest()
                with Image.open(BytesIO(payload)) as image:
                    images.append(image.convert('RGB').copy())
            inputs = processor(images=images, return_tensors='pt')
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = model(**inputs)
                hidden = torch.nn.functional.normalize(output.last_hidden_state[:, 0], dim=1)
            array = hidden.detach().cpu().float().numpy()
            for member, vector in zip(batch, array):
                vectors[member] = vector
            print(f'[DINOv2] {min(start + batch_size, len(unique))}/{len(unique)}', flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vectors, hashes


def select_positive_references(
    cores: list[dict],
    bank: list[dict],
    vectors: dict[str, np.ndarray],
    hashes: dict[str, str],
) -> list[dict]:
    bank_by_asset: dict[str, list[dict]] = defaultdict(list)
    for item in bank:
        bank_by_asset[item['asset_category']].append(item)

    manifest: list[dict] = []
    for core in cores:
        asset = core['asset_category']
        candidates = bank_by_asset.get(asset, [])
        if not candidates:
            raise RuntimeError(f'{asset}: no positive train references')

        target = normalized_member(core['target_image'])
        target_vec = vectors[target]
        target_hash = hashes[target]
        scored = []
        exact_duplicates = 0
        for candidate in candidates:
            member = candidate['member']
            if hashes[member] == target_hash:
                exact_duplicates += 1
                continue
            similarity = float(np.dot(target_vec, vectors[member]))
            scored.append((similarity, member, candidate))
        if not scored:
            raise RuntimeError(f"{core['core_id']}: no non-duplicate positive reference candidates")
        scored.sort(key=lambda item: (-item[0], item[1]))
        similarity, member, selected = scored[0]
        manifest.append(
            {
                'core_id': core['core_id'],
                'asset_category': asset,
                'target_image': target,
                'positive_reference_image': member,
                'positive_reference_defect_label': selected['defect_label'],
                'cosine_similarity': round(similarity, 8),
                'positive_bank_size': len(candidates),
                'exact_duplicate_candidates_excluded': exact_duplicates,
                'retrieval_policy': 'same-asset supervised-train anomalous bank; DINOv2 cosine nearest neighbor',
                'target_gt_used_for_selection': False,
                'reference_split': 'train',
                'target_split': 'val',
            }
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa-dir', type=Path, required=True)
    parser.add_argument('--supervised-zip', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--model-id', default='facebook/dinov2-small')
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    rows = load_gold_rows(args.qa_dir)
    cores = unique_core_rows(rows)
    assets = {row['asset_category'] for row in cores}
    bank = build_positive_bank(args.supervised_zip, assets)

    if args.device == 'auto':
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'
    else:
        device = args.device

    all_members = [row['target_image'] for row in cores] + [row['member'] for row in bank]
    vectors, hashes = embed_members(
        args.supervised_zip,
        all_members,
        model_id=args.model_id,
        batch_size=args.batch_size,
        device=device,
    )
    manifest = select_positive_references(cores, bank, vectors, hashes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / 'positive_reference_manifest.csv', manifest)

    summary: list[dict] = []
    for asset in sorted(assets):
        rows_asset = [row for row in manifest if row['asset_category'] == asset]
        bank_asset = [row for row in bank if row['asset_category'] == asset]
        summary.append(
            {
                'asset_category': asset,
                'target_count': len(rows_asset),
                'positive_bank_count': len(bank_asset),
                'positive_bank_defects': '|'.join(sorted({row['defect_label'] for row in bank_asset})),
                'unique_positive_references_selected': len({row['positive_reference_image'] for row in rows_asset}),
                'mean_cosine_similarity': round(float(np.mean([float(row['cosine_similarity']) for row in rows_asset])), 8),
                'max_exact_duplicates_excluded_per_target': max(int(row['exact_duplicate_candidates_excluded']) for row in rows_asset),
            }
        )
    write_csv(args.out_dir / 'positive_reference_summary.csv', summary)

    metadata = {
        'benchmark': 'gold_v2_final',
        'target_count': len(cores),
        'positive_bank_count': len(bank),
        'positive_bank_source': 'supervised_fault_classification train split only',
        'retrieval_model': args.model_id,
        'retrieval_policy': 'same-asset nearest anomalous train image by DINOv2 cosine similarity',
        'target_gt_used_for_selection': False,
        'exact_content_duplicates_excluded': True,
        'reference_target_split_disjoint': True,
    }
    (args.out_dir / 'positive_reference_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

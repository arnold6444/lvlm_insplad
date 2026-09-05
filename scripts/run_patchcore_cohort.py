#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from insplad_benchmark.evidence_ablation import write_csv_atomic  # noqa: E402
from insplad_benchmark.vision_expert import (  # noqa: E402
    save_roi_artifacts,
    validate_roi_manifest,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def batch_field(batch, name: str):
    if isinstance(batch, dict):
        return batch[name]
    return getattr(batch, name)


def _paths(value) -> list[str]:
    if isinstance(value, (str, Path)):
        return [str(value)]
    return [str(item) for item in list(value)]


def _maps(value, n: int) -> list[np.ndarray]:
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    array = np.asarray(value)
    if n == 1 and array.ndim == 2:
        return [array]
    if array.ndim >= 3 and array.shape[0] == n:
        return [np.asarray(array[index]) for index in range(n)]
    values = list(value)
    if len(values) != n:
        raise ValueError(f"prediction anomaly_map count mismatch: {len(values)} != {n}")
    return [np.asarray(item) for item in values]


def _scores(value, n: int) -> list[float | None]:
    if value is None:
        return [None] * n
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    array = np.asarray(value).reshape(-1)
    if len(array) != n:
        raise ValueError(f"prediction score count mismatch: {len(array)} != {n}")
    return [float(item) for item in array]


def flatten_predictions(predictions) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for batch in predictions:
        paths = _paths(batch_field(batch, "image_path"))
        maps = _maps(batch_field(batch, "anomaly_map"), len(paths))
        try:
            scores_raw = batch_field(batch, "pred_score")
        except Exception:
            scores_raw = None
        scores = _scores(scores_raw, len(paths))
        for path, anomaly_map, score in zip(paths, maps, scores, strict=True):
            records.append(
                {
                    "image_path": Path(path),
                    "anomaly_map": anomaly_map,
                    "pred_score": score,
                }
            )
    return records


def _artifact_exists(root: Path, row: Mapping[str, str]) -> bool:
    for field in ("roi_path", "anomaly_map_path", "overlay_path"):
        value = str(row.get(field, ""))
        if not value:
            return False
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PatchCore on a prepared InsPLAD cohort workspace.")
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--coreset-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not 0 < args.coreset_ratio <= 1:
        parser.error("coreset-ratio must be in (0, 1]")
    manifest_path = args.workspace_root / "workspace_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    workspace = pd.DataFrame(read_csv(manifest_path))
    if workspace.empty:
        raise RuntimeError("workspace manifest is empty")
    query_workspace = workspace.loc[workspace.role == "query"].copy()
    train_workspace = workspace.loc[workspace.role == "train_normal"].copy()
    if query_workspace.empty or train_workspace.empty:
        raise RuntimeError("workspace must contain query and train_normal rows")
    query_workspace["label_index"] = query_workspace.label_index.astype(int)
    train_workspace["label_index"] = train_workspace.label_index.astype(int)

    query_ids = [str(value) for value in query_workspace.query_id]
    assets = sorted(set(str(value) for value in query_workspace.asset_category))
    if set(assets) != set(str(value) for value in train_workspace.asset_category):
        raise RuntimeError("query/train asset coverage mismatch")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    roi_manifest = args.out_dir / "roi_manifest.csv"
    existing_rows = [] if args.force or not roi_manifest.is_file() else read_csv(roi_manifest)
    existing = {
        (str(row.get("query_id", "")), str(row.get("expert", ""))): row
        for row in existing_rows
        if row.get("expert") == "patchcore"
    }

    from anomalib.data import Tabular
    from anomalib.data.utils import TestSplitMode, ValSplitMode
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    from lightning import seed_everything

    seed_everything(args.seed, workers=True)
    all_rows = [row for row in existing_rows if row.get("expert") != "patchcore"]
    current_patch_rows: dict[str, dict[str, object]] = {
        qid: dict(existing[(qid, "patchcore")])
        for qid in query_ids
        if (qid, "patchcore") in existing and _artifact_exists(args.out_dir, existing[(qid, "patchcore")])
    }

    for asset in assets:
        query_asset = query_workspace.loc[query_workspace.asset_category == asset].copy()
        train_asset = train_workspace.loc[train_workspace.asset_category == asset].copy()
        asset_ids = [str(value) for value in query_asset.query_id]
        if not args.force and all(qid in current_patch_rows for qid in asset_ids):
            print(f"[patchcore] {asset}: reuse {len(asset_ids)} queries")
            continue

        train_asset["split"] = "train"
        query_asset["split"] = "test"
        combined = pd.concat([train_asset, query_asset], ignore_index=True)
        samples = pd.DataFrame(
            {
                "image_path": [
                    str((args.workspace_root / str(value)).resolve())
                    for value in combined.local_path
                ],
                "label_index": [int(value) for value in combined.label_index],
                "split": [str(value) for value in combined.split],
            }
        )
        datamodule = Tabular(
            name=f"insplad-patchcore-{asset}",
            samples=samples,
            train_batch_size=16,
            eval_batch_size=8,
            num_workers=2,
            test_split_mode=TestSplitMode.FROM_DIR,
            val_split_mode=ValSplitMode.NONE,
            seed=args.seed,
        )
        model = Patchcore(
            backbone="wide_resnet50_2",
            layers=("layer2", "layer3"),
            coreset_sampling_ratio=args.coreset_ratio,
            num_neighbors=9,
            post_processor=False,
            evaluator=False,
            visualizer=False,
        )
        engine = Engine(
            accelerator="gpu",
            devices=1,
            max_epochs=1,
            deterministic=True,
            barebones=True,
            limit_val_batches=0,
            default_root_dir=Path("/content/anomalib_runs/patchcore") / asset,
        )
        engine.fit(model=model, datamodule=datamodule)
        predictions = engine.predict(model=model, datamodule=datamodule)
        records = flatten_predictions(predictions)

        by_path = {
            str((args.workspace_root / str(row.local_path)).resolve()): row
            for _, row in query_asset.iterrows()
        }
        seen: set[str] = set()
        for record in records:
            resolved = str(Path(record["image_path"]).resolve())
            query = by_path.get(resolved)
            if query is None:
                continue
            query_id = str(query.query_id)
            row = save_roi_artifacts(
                image_path=Path(resolved),
                anomaly_map=record["anomaly_map"],
                out_dir=args.out_dir / "vision_experts" / "patchcore" / asset,
                query_id=query_id,
                expert="patchcore",
                asset_category=asset,
                query_image=str(query.source_path),
                metadata={
                    "model_id": "anomalib.Patchcore/wide_resnet50_2",
                    "model_revision": "anomalib==2.6.0",
                    "coreset_sampling_ratio": args.coreset_ratio,
                    "model_pred_score": (
                        "" if record["pred_score"] is None else record["pred_score"]
                    ),
                },
                artifact_root=args.out_dir,
            )
            current_patch_rows[query_id] = row
            seen.add(query_id)
        missing = sorted(set(asset_ids) - seen)
        if missing:
            raise RuntimeError(f"PatchCore prediction coverage missing for {asset}: {missing[:10]}")

        ordered_patch = [current_patch_rows[qid] for qid in query_ids if qid in current_patch_rows]
        write_csv_atomic(roi_manifest, [*all_rows, *ordered_patch])
        print(f"[patchcore] {asset}: complete {len(asset_ids)} queries")

        model = None
        engine = None
        datamodule = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    ordered_patch = [current_patch_rows[qid] for qid in query_ids]
    write_csv_atomic(roi_manifest, [*all_rows, *ordered_patch])
    validate_roi_manifest(
        ordered_patch,
        query_ids=query_ids,
        experts=("patchcore",),
        manifest_path=roi_manifest,
        artifact_root=args.out_dir,
        require_files=True,
    )
    summary = {
        "query_count": len(query_ids),
        "asset_count": len(assets),
        "coreset_ratio": args.coreset_ratio,
        "roi_manifest": str(roi_manifest),
    }
    (args.out_dir / "patchcore_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
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
    parser = argparse.ArgumentParser(description="Run Dinomaly on a prepared full InsPLAD cohort.")
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--encoder", default="vit_small_patch14_reg4_dinov2")
    parser.add_argument("--decoder-depth", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-candidates", nargs="+", type=int, default=[8, 4, 2])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    roi_manifest = args.out_dir / "roi_manifest.csv"
    existing_rows = [] if args.force or not roi_manifest.is_file() else read_csv(roi_manifest)
    existing = {
        (str(row.get("query_id", "")), str(row.get("expert", ""))): row
        for row in existing_rows
        if row.get("expert") == "dinomaly"
    }
    current_rows = {
        qid: dict(existing[(qid, "dinomaly")])
        for qid in query_ids
        if (qid, "dinomaly") in existing
        and _artifact_exists(args.out_dir, existing[(qid, "dinomaly")])
    }
    if len(current_rows) == len(query_ids) and not args.force:
        validate_roi_manifest(
            list(current_rows.values()),
            query_ids=query_ids,
            experts=("dinomaly",),
            manifest_path=roi_manifest,
            artifact_root=args.out_dir,
            require_files=True,
        )
        print("[dinomaly] all queries already complete; reuse")
        return 0

    from anomalib.data import Tabular
    from anomalib.data.utils import TestSplitMode, ValSplitMode
    from anomalib.engine import Engine
    from anomalib.models import Dinomaly
    from lightning import seed_everything
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    seed_everything(args.seed, workers=True)
    train_workspace["split"] = "train"
    query_workspace["split"] = "test"
    combined = pd.concat([train_workspace, query_workspace], ignore_index=True)
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

    predictions = engine = model = datamodule = None
    records = None
    used_batch_size = None
    last_oom = None
    for batch_size in args.batch_candidates:
        try:
            print(
                f"[Dinomaly] batch={batch_size}, train={len(train_workspace)}, "
                f"query={len(query_workspace)}"
            )
            datamodule = Tabular(
                name="insplad-dinomaly-full",
                samples=samples,
                train_batch_size=batch_size,
                eval_batch_size=max(1, batch_size // 2),
                num_workers=2,
                test_split_mode=TestSplitMode.FROM_DIR,
                val_split_mode=ValSplitMode.NONE,
                seed=args.seed,
            )
            model = Dinomaly(
                encoder_name=args.encoder,
                decoder_depth=args.decoder_depth,
                bottleneck_dropout=0.2,
                use_context_recentering=True,
                precision="float16",
                post_processor=False,
                evaluator=False,
                visualizer=False,
            )
            engine = Engine(
                accelerator="gpu",
                devices=1,
                precision="bf16-mixed",
                max_steps=args.max_steps,
                max_epochs=-1,
                deterministic=True,
                barebones=True,
                limit_val_batches=0,
                default_root_dir="/content/anomalib_runs/dinomaly_full",
            )
            engine.fit(model=model, datamodule=datamodule)
            predictions = engine.predict(model=model, datamodule=datamodule)
            records = flatten_predictions(predictions)
            used_batch_size = batch_size
            break
        except torch.cuda.OutOfMemoryError as exc:
            last_oom = exc
            print(f"[Dinomaly] OOM batch={batch_size}; retry smaller")
            predictions = engine = model = datamodule = None
            gc.collect()
            torch.cuda.empty_cache()

    if records is None:
        raise RuntimeError("Dinomaly OOM for all batch candidates") from last_oom

    by_path = {
        str((args.workspace_root / str(row.local_path)).resolve()): row
        for _, row in query_workspace.iterrows()
    }
    new_rows: list[dict] = []
    for record in records:
        resolved = str(Path(record["image_path"]).resolve())
        query = by_path.get(resolved)
        if query is None:
            continue
        query_id = str(query.query_id)
        row = save_roi_artifacts(
            image_path=Path(resolved),
            anomaly_map=record["anomaly_map"],
            out_dir=args.out_dir / "vision_experts" / "dinomaly" / str(query.asset_category),
            query_id=query_id,
            expert="dinomaly",
            asset_category=str(query.asset_category),
            query_image=str(query.source_path),
            metadata={
                "expert_protocol": "unified multi-asset leakage-safe train-normal Dinomaly",
                "model_id": f"anomalib.Dinomaly/{args.encoder}",
                "model_revision": "anomalib==2.6.0",
                "decoder_depth": args.decoder_depth,
                "max_steps": args.max_steps,
                "train_batch_size": used_batch_size,
                "use_context_recentering": True,
                "model_pred_score": (
                    "" if record["pred_score"] is None else record["pred_score"]
                ),
            },
            artifact_root=args.out_dir,
        )
        current_rows[query_id] = row
        new_rows.append(row)

    missing = sorted(set(query_ids) - set(current_rows))
    if missing:
        raise RuntimeError(f"Dinomaly prediction coverage missing: {missing[:10]}")

    kept = [row for row in existing_rows if row.get("expert") != "dinomaly"]
    ordered = [current_rows[qid] for qid in query_ids]
    write_csv_atomic(roi_manifest, [*kept, *ordered])
    validate_roi_manifest(
        ordered,
        query_ids=query_ids,
        experts=("dinomaly",),
        manifest_path=roi_manifest,
        artifact_root=args.out_dir,
        require_files=True,
    )

    summary = {
        "query_count": len(query_ids),
        "train_count": len(train_workspace),
        "encoder": args.encoder,
        "decoder_depth": args.decoder_depth,
        "max_steps": args.max_steps,
        "batch_size": used_batch_size,
        "new_rows": len(new_rows),
    }
    (args.out_dir / "dinomaly_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    predictions = engine = model = datamodule = None
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

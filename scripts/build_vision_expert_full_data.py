#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_vision_expert_smoke_data as base  # noqa: E402


def validate_full_qa(rows):
    if not rows:
        raise RuntimeError("vision QA is empty")
    ids = [str(row.get("id", "")).strip() for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise RuntimeError("vision QA IDs must be present and unique")
    assets = sorted({str(row.get("asset_category", "")).strip() for row in rows})
    if not assets or "" in assets:
        raise RuntimeError("every vision QA row must have asset_category")
    for row in rows:
        missing = [key for key in ("image", "answer") if not row.get(key)]
        if missing:
            raise RuntimeError(f"vision QA row {row.get('id')} missing fields: {missing}")
        if str(row["answer"]) not in {"normal", "anomalous"}:
            raise RuntimeError(
                f"vision QA row {row.get('id')} has invalid answer: {row['answer']}"
            )
    return ids, assets


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a leakage-safe Vision Expert workspace for the full gold-v2 image set."
    )
    parser.add_argument("--anomaly-manifest", type=Path, required=True)
    parser.add_argument("--unsup-zip", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-train-per-asset", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.max_train_per_asset <= 0:
        parser.error("max-train-per-asset must be positive")
    for label, path in (
        ("anomaly manifest", args.anomaly_manifest),
        ("unsupervised ZIP", args.unsup_zip),
        ("vision QA", args.qa),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    qa_rows = base.read_jsonl(args.qa)
    ids, assets = validate_full_qa(qa_rows)
    expected = base.expected_config(args, qa_rows, ids, assets)
    expected.update(
        {
            "workspace_scope": "gold_v2_full_unique_images",
            "query_count": len(qa_rows),
        }
    )

    reusable, reason = base.validate_workspace(args.out_dir, expected)
    if reusable and not args.force:
        print(f"[vision-workspace] reusing {args.out_dir}: {reason}")
        return 0

    if args.force:
        print("[vision-workspace] --force requested; rebuilding safely", flush=True)
    elif args.out_dir.exists() or args.out_dir.is_symlink():
        print(
            f"[vision-workspace] stale/partial workspace; rebuilding safely: {reason}",
            flush=True,
        )

    base.build_workspace(args, qa_rows, assets, expected)
    reusable, reason = base.validate_workspace(args.out_dir, expected)
    if not reusable:
        raise RuntimeError(f"full workspace validation failed: {reason}")

    summary = {
        "query_count": len(qa_rows),
        "assets": assets,
        "max_train_per_asset": args.max_train_per_asset,
        "seed": args.seed,
        "workspace": str(args.out_dir),
        "validation": reason,
    }
    (args.out_dir / "full_workspace_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

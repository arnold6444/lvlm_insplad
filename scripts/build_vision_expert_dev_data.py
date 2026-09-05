#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_vision_expert_smoke_data as base


def validate_dev_qa(rows: list[dict], *, min_rows: int = 20, max_rows: int = 500) -> tuple[list[str], list[str]]:
    if not min_rows <= len(rows) <= max_rows:
        raise RuntimeError(
            f"dev QA must contain {min_rows}-{max_rows} rows, got {len(rows)}"
        )
    ids = [str(row.get("id", "")).strip() for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise RuntimeError("dev QA IDs must be present and unique")
    assets = sorted({str(row.get("asset_category", "")).strip() for row in rows})
    if not assets or "" in assets:
        raise RuntimeError("every dev QA row must have asset_category")
    for row in rows:
        missing = [key for key in ("image", "answer") if not row.get(key)]
        if missing:
            raise RuntimeError(f"dev QA row {row.get('id')} missing fields: {missing}")
        if str(row["answer"]) not in {"normal", "anomalous"}:
            raise RuntimeError(
                f"dev QA row {row.get('id')} has invalid answer: {row['answer']}"
            )
    return ids, assets


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract an expanded dev cohort and leakage-safe normal pools."
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
        ("dev QA", args.qa),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    qa_rows = base.read_jsonl(args.qa)
    ids, assets = validate_dev_qa(qa_rows)
    expected = base.expected_config(args, qa_rows, ids, assets)
    reusable, reason = base.validate_workspace(args.out_dir, expected)
    if reusable and not args.force:
        print(f"[vision-dev-workspace] reusing {args.out_dir}: {reason}")
        return 0
    if args.force:
        print("[vision-dev-workspace] --force requested; rebuilding safely", flush=True)
    elif args.out_dir.exists() or args.out_dir.is_symlink():
        print(
            f"[vision-dev-workspace] stale/partial workspace; rebuilding safely: {reason}",
            flush=True,
        )

    base.build_workspace(args, qa_rows, assets, expected)
    config_path = args.out_dir / "workspace_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["cohort_role"] = "expanded_agent_dev"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[vision-dev-workspace] ready: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

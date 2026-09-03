#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from insplad_benchmark.dataset import (
    leakage_safe_train_normal_candidates,
    parse_unsupervised_manifest,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_suffix(member: str) -> str:
    suffix = PurePosixPath(member.replace("\\", "/")).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"} else ".png"


def extract_member(archive: ZipFile, member: str, out_path: Path) -> None:
    normalized = member.replace("\\", "/").lstrip("/")
    try:
        data = archive.read(member)
    except KeyError:
        matches = [
            name
            for name in archive.namelist()
            if name.replace("\\", "/").lstrip("/") == normalized
        ]
        if len(matches) != 1:
            raise FileNotFoundError(f"ZIP member not found or ambiguous: {member}")
        data = archive.read(matches[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)


def deterministic_cap(rows: Sequence[Mapping[str, object]], limit: int, *, seed: int, asset: str) -> list[Mapping[str, object]]:
    ordered = sorted(rows, key=lambda row: str(row["relative_path"]))
    if len(ordered) <= limit:
        return ordered
    generator = random.Random(f"{seed}:{asset}:train-normal-cap")
    indices = sorted(generator.sample(range(len(ordered)), limit))
    return [ordered[index] for index in indices]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract one fixed smoke cohort and leakage-safe normal training pools."
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
    qa_rows = read_jsonl(args.qa)
    if not 20 <= len(qa_rows) <= 100:
        raise RuntimeError(f"smoke QA must contain 20-100 rows, got {len(qa_rows)}")
    ids = [str(row.get("id", "")) for row in qa_rows]
    if any(not query_id for query_id in ids) or len(set(ids)) != len(ids):
        raise RuntimeError("smoke QA IDs must be present and unique")
    assets = sorted({str(row.get("asset_category", "")) for row in qa_rows})
    if not assets or "" in assets:
        raise RuntimeError("every smoke QA row must have asset_category")

    if args.out_dir.exists() and args.force:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "workspace_manifest.csv"
    config_path = args.out_dir / "workspace_config.json"
    if manifest_path.is_file() and config_path.is_file() and not args.force:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        expected = {
            "qa_count": len(qa_rows),
            "qa_ids": ids,
            "assets": assets,
            "max_train_per_asset": args.max_train_per_asset,
            "seed": args.seed,
        }
        mismatches = {key: {"expected": value, "actual": existing.get(key)} for key, value in expected.items() if existing.get(key) != value}
        if mismatches:
            raise RuntimeError(f"existing workspace config mismatch; use --force: {mismatches}")
        print(f"Reusing existing workspace: {args.out_dir}")
        return 0

    unsupervised_rows = parse_unsupervised_manifest(args.anomaly_manifest)
    candidates_by_asset = leakage_safe_train_normal_candidates(unsupervised_rows, assets)
    selected_normals = {
        asset: deterministic_cap(
            candidates_by_asset[asset],
            args.max_train_per_asset,
            seed=args.seed,
            asset=asset,
        )
        for asset in assets
    }

    manifest_rows: list[dict[str, object]] = []
    with ZipFile(args.unsup_zip) as archive:
        for row in qa_rows:
            query_id = str(row["id"])
            asset = str(row["asset_category"])
            member = str(row["image"])
            out_path = args.out_dir / "queries" / asset / f"{query_id}{safe_suffix(member)}"
            extract_member(archive, member, out_path)
            manifest_rows.append(
                {
                    "role": "query",
                    "query_id": query_id,
                    "asset_category": asset,
                    "source_path": member,
                    "local_path": out_path.relative_to(args.out_dir).as_posix(),
                    "label_index": 0 if str(row["answer"]) == "normal" else 1,
                    "answer": row["answer"],
                    "split": "test",
                }
            )

        for asset in assets:
            for candidate in selected_normals[asset]:
                member = str(candidate["relative_path"])
                identity = hashlib.sha256(member.encode("utf-8")).hexdigest()[:16]
                out_path = args.out_dir / "train" / asset / "good" / f"{identity}{safe_suffix(member)}"
                extract_member(archive, member, out_path)
                manifest_rows.append(
                    {
                        "role": "train_normal",
                        "query_id": "",
                        "asset_category": asset,
                        "source_path": member,
                        "local_path": out_path.relative_to(args.out_dir).as_posix(),
                        "label_index": 0,
                        "answer": "normal",
                        "split": "train",
                        "content_fingerprint": candidate["content_fingerprint"],
                    }
                )

    write_csv(manifest_path, manifest_rows)
    counts = Counter((str(row["role"]), str(row["asset_category"])) for row in manifest_rows)
    config = {
        "schema_version": 1,
        "qa_count": len(qa_rows),
        "qa_ids": ids,
        "assets": assets,
        "max_train_per_asset": args.max_train_per_asset,
        "seed": args.seed,
        "leakage_policy": "same-asset train normal only; exclude every content fingerprint present anywhere in unsupervised test",
        "counts": {f"{role}:{asset}": count for (role, asset), count in sorted(counts.items())},
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import traceback
import uuid
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

SCHEMA_VERSION = 2
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_member(value: str) -> str:
    return value.replace("\\", "/").lstrip("/")


def safe_suffix(member: str) -> str:
    suffix = PurePosixPath(normalized_member(member)).suffix.lower()
    return suffix if suffix in IMAGE_SUFFIXES else ".png"


def deterministic_cap(
    rows: Sequence[Mapping[str, object]],
    limit: int,
    *,
    seed: int,
    asset: str,
) -> list[Mapping[str, object]]:
    ordered = sorted(rows, key=lambda row: str(row["relative_path"]))
    if len(ordered) <= limit:
        return ordered
    indices = sorted(
        random.Random(f"{seed}:{asset}:train-normal-cap").sample(
            range(len(ordered)), limit
        )
    )
    return [ordered[index] for index in indices]


def validate_qa(rows: Sequence[Mapping[str, object]]) -> tuple[list[str], list[str]]:
    if not 20 <= len(rows) <= 100:
        raise RuntimeError(f"smoke QA must contain 20-100 rows, got {len(rows)}")
    ids = [str(row.get("id", "")).strip() for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise RuntimeError("smoke QA IDs must be present and unique")
    assets = sorted({str(row.get("asset_category", "")).strip() for row in rows})
    if not assets or "" in assets:
        raise RuntimeError("every smoke QA row must have asset_category")
    for row in rows:
        missing = [key for key in ("image", "answer") if not row.get(key)]
        if missing:
            raise RuntimeError(f"smoke QA row {row.get('id')} missing fields: {missing}")
        if str(row["answer"]) not in {"normal", "anomalous"}:
            raise RuntimeError(
                f"smoke QA row {row.get('id')} has invalid answer: {row['answer']}"
            )
    return ids, assets


def expected_config(
    args: argparse.Namespace,
    qa_rows: Sequence[Mapping[str, object]],
    ids: Sequence[str],
    assets: Sequence[str],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "qa_count": len(qa_rows),
        "qa_ids": list(ids),
        "qa_sha256": sha256_file(args.qa),
        "assets": list(assets),
        "max_train_per_asset": args.max_train_per_asset,
        "seed": args.seed,
        "anomaly_manifest_sha256": sha256_file(args.anomaly_manifest),
        "source_zip_name": args.unsup_zip.name,
        "source_zip_size": args.unsup_zip.stat().st_size,
    }


def safe_local_file(root: Path, value: str) -> Path:
    relative = Path(value)
    if not value or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe local_path: {value!r}")
    return root / relative


def validate_workspace(
    root: Path,
    expected: Mapping[str, object],
) -> tuple[bool, str]:
    config_path = root / "workspace_config.json"
    manifest_path = root / "workspace_manifest.csv"
    if not root.is_dir() or not config_path.is_file() or not manifest_path.is_file():
        return False, "workspace directory, manifest, or config is missing"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rows = read_csv(manifest_path)
    except (OSError, json.JSONDecodeError, csv.Error) as exc:
        return False, f"workspace metadata is unreadable: {exc}"
    if not isinstance(config, dict) or not rows:
        return False, "workspace metadata is empty or invalid"
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        return False, f"workspace identity changed: {json.dumps(mismatches, ensure_ascii=False)}"

    queries: list[str] = []
    train_assets: set[str] = set()
    local_paths: set[str] = set()
    counts: Counter[tuple[str, str]] = Counter()
    for index, row in enumerate(rows):
        role = str(row.get("role", ""))
        asset = str(row.get("asset_category", ""))
        local_value = str(row.get("local_path", ""))
        if role not in {"query", "train_normal"} or not asset:
            return False, f"workspace row {index} has invalid role/asset"
        if local_value in local_paths:
            return False, f"duplicate workspace local_path: {local_value}"
        local_paths.add(local_value)
        try:
            artifact = safe_local_file(root, local_value)
        except ValueError as exc:
            return False, str(exc)
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            return False, f"workspace artifact is missing or empty: {artifact}"
        counts[(role, asset)] += 1
        if role == "query":
            query_id = str(row.get("query_id", ""))
            if not query_id:
                return False, f"workspace query row {index} has no query_id"
            queries.append(query_id)
        else:
            train_assets.add(asset)

    if len(queries) != len(expected["qa_ids"]) or set(queries) != set(expected["qa_ids"]):
        return False, "workspace query coverage differs from smoke QA"
    if train_assets != set(expected["assets"]):
        return False, "workspace train-normal asset coverage differs from smoke QA"
    actual_counts = {
        f"{role}:{asset}": count for (role, asset), count in sorted(counts.items())
    }
    if config.get("counts") != actual_counts:
        return False, "workspace counts differ from workspace manifest"
    return True, f"validated {len(queries)} queries and {len(rows) - len(queries)} train normals"


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def install_staging(staging: Path, target: Path) -> None:
    backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
    if target.exists() or target.is_symlink():
        target.rename(backup)
    try:
        staging.rename(target)
    except Exception:
        if backup.exists() or backup.is_symlink():
            backup.rename(target)
        raise
    else:
        remove_path(backup)


def build_workspace(
    args: argparse.Namespace,
    qa_rows: Sequence[Mapping[str, object]],
    assets: Sequence[str],
    expected: Mapping[str, object],
) -> None:
    unsup_rows = parse_unsupervised_manifest(args.anomaly_manifest)
    candidates = leakage_safe_train_normal_candidates(unsup_rows, assets)
    selected = {
        asset: deterministic_cap(
            candidates[asset],
            args.max_train_per_asset,
            seed=args.seed,
            asset=asset,
        )
        for asset in assets
    }
    print(
        "[vision-workspace] leakage-safe train pools:",
        {asset: len(rows) for asset, rows in selected.items()},
        flush=True,
    )

    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = args.out_dir.with_name(
        f".{args.out_dir.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    )
    remove_path(staging)
    staging.mkdir(parents=True)
    manifest_rows: list[dict[str, object]] = []
    try:
        with ZipFile(args.unsup_zip) as archive:
            member_index: dict[str, str] = {}
            ambiguous: set[str] = set()
            for actual in archive.namelist():
                key = normalized_member(actual)
                if key in member_index and member_index[key] != actual:
                    ambiguous.add(key)
                else:
                    member_index[key] = actual

            required = [str(row["image"]) for row in qa_rows]
            required += [
                str(row["relative_path"])
                for asset in assets
                for row in selected[asset]
            ]
            normalized_required = {normalized_member(value) for value in required}
            missing = sorted(normalized_required - set(member_index))
            conflicts = sorted(normalized_required & ambiguous)
            if missing or conflicts:
                raise FileNotFoundError(
                    "source ZIP does not match manifest/QA: "
                    + json.dumps(
                        {
                            "zip": str(args.unsup_zip),
                            "missing_count": len(missing),
                            "first_missing": missing[:10],
                            "ambiguous_count": len(conflicts),
                            "first_ambiguous": conflicts[:10],
                        },
                        ensure_ascii=False,
                    )
                )

            def extract(member: str, out_path: Path) -> None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(archive.read(member_index[normalized_member(member)]))

            for position, row in enumerate(qa_rows, start=1):
                query_id = str(row["id"])
                asset = str(row["asset_category"])
                member = str(row["image"])
                out_path = staging / "queries" / asset / f"{query_id}{safe_suffix(member)}"
                extract(member, out_path)
                manifest_rows.append(
                    {
                        "role": "query",
                        "query_id": query_id,
                        "asset_category": asset,
                        "source_path": member,
                        "local_path": out_path.relative_to(staging).as_posix(),
                        "label_index": 0 if str(row["answer"]) == "normal" else 1,
                        "answer": row["answer"],
                        "split": "test",
                    }
                )
                if position % 10 == 0 or position == len(qa_rows):
                    print(
                        f"[vision-workspace] queries {position}/{len(qa_rows)}",
                        flush=True,
                    )

            done = 0
            total = sum(len(rows) for rows in selected.values())
            for asset in assets:
                for row in selected[asset]:
                    member = str(row["relative_path"])
                    identity = hashlib.sha256(member.encode()).hexdigest()[:16]
                    out_path = staging / "train" / asset / "good" / f"{identity}{safe_suffix(member)}"
                    extract(member, out_path)
                    manifest_rows.append(
                        {
                            "role": "train_normal",
                            "query_id": "",
                            "asset_category": asset,
                            "source_path": member,
                            "local_path": out_path.relative_to(staging).as_posix(),
                            "label_index": 0,
                            "answer": "normal",
                            "split": "train",
                            "content_fingerprint": row["content_fingerprint"],
                        }
                    )
                    done += 1
                    if done % 100 == 0 or done == total:
                        print(f"[vision-workspace] train normals {done}/{total}", flush=True)

        counts = Counter(
            (str(row["role"]), str(row["asset_category"])) for row in manifest_rows
        )
        config = {
            **expected,
            "leakage_policy": (
                "same-asset train normal only; exclude every content fingerprint "
                "present anywhere in unsupervised test"
            ),
            "counts": {
                f"{role}:{asset}": count
                for (role, asset), count in sorted(counts.items())
            },
        }
        write_csv(staging / "workspace_manifest.csv", manifest_rows)
        (staging / "workspace_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        valid, reason = validate_workspace(staging, expected)
        if not valid:
            raise RuntimeError(f"new workspace validation failed: {reason}")
        install_staging(staging, args.out_dir)
        print(f"[vision-workspace] ready: {args.out_dir} ({reason})", flush=True)
    finally:
        remove_path(staging)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract a fixed smoke cohort and leakage-safe normal pools."
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
        ("smoke QA", args.qa),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    qa_rows = read_jsonl(args.qa)
    ids, assets = validate_qa(qa_rows)
    expected = expected_config(args, qa_rows, ids, assets)
    reusable, reason = validate_workspace(args.out_dir, expected)
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
    build_workspace(args, qa_rows, assets, expected)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(
            f"\n[vision-workspace] FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        raise SystemExit(1)

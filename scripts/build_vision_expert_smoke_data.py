#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import tempfile
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

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
WORKSPACE_SCHEMA_VERSION = 2


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qa_identity_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    identity_rows = [
        {
            "id": str(row.get("id", "")),
            "image": str(row.get("image", "")),
            "asset_category": str(row.get("asset_category", "")),
            "answer": str(row.get("answer", "")),
        }
        for row in rows
    ]
    payload = json.dumps(
        identity_rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def safe_suffix(member: str) -> str:
    suffix = PurePosixPath(member.replace("\\", "/")).suffix.lower()
    return suffix if suffix in IMAGE_SUFFIXES else ".png"


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
    if not data:
        raise RuntimeError(f"ZIP member is empty: {member}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)


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
    generator = random.Random(f"{seed}:{asset}:train-normal-cap")
    indices = sorted(generator.sample(range(len(ordered)), limit))
    return [ordered[index] for index in indices]


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def expected_workspace_config(
    *,
    qa_rows: Sequence[Mapping[str, object]],
    anomaly_manifest: Path,
    max_train_per_asset: int,
    seed: int,
) -> dict[str, object]:
    ids = [str(row["id"]) for row in qa_rows]
    assets = sorted({str(row["asset_category"]) for row in qa_rows})
    return {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "qa_count": len(qa_rows),
        "qa_ids": ids,
        "qa_identity_sha256": qa_identity_sha256(qa_rows),
        "anomaly_manifest_sha256": sha256_file(anomaly_manifest),
        "assets": assets,
        "max_train_per_asset": max_train_per_asset,
        "seed": seed,
    }


def validate_workspace(
    root: Path,
    expected: Mapping[str, object],
) -> tuple[bool, str]:
    manifest_path = root / "workspace_manifest.csv"
    config_path = root / "workspace_config.json"
    if not root.is_dir():
        return False, "workspace directory is missing"
    if not config_path.is_file():
        return False, "workspace_config.json is missing"
    if not manifest_path.is_file():
        return False, "workspace_manifest.csv is missing"

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid workspace_config.json: {exc}"
    if not isinstance(config, dict):
        return False, "workspace_config.json must contain one JSON object"

    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        return False, f"workspace contract mismatch: {mismatches}"

    try:
        rows = read_csv(manifest_path)
    except (OSError, csv.Error) as exc:
        return False, f"invalid workspace_manifest.csv: {exc}"
    if not rows:
        return False, "workspace_manifest.csv is empty"

    query_rows = [row for row in rows if row.get("role") == "query"]
    expected_ids = [str(value) for value in expected["qa_ids"]]
    actual_ids = [str(row.get("query_id", "")) for row in query_rows]
    if actual_ids != expected_ids:
        return False, "query IDs/order do not match the selected smoke QA"

    expected_assets = {str(value) for value in expected["assets"]}
    train_counts = Counter(
        str(row.get("asset_category", ""))
        for row in rows
        if row.get("role") == "train_normal"
    )
    missing_train_assets = sorted(asset for asset in expected_assets if train_counts[asset] <= 0)
    if missing_train_assets:
        return False, f"normal training pool is missing assets: {missing_train_assets}"

    seen_local_paths: set[str] = set()
    resolved_root = root.resolve()
    for index, row in enumerate(rows):
        value = str(row.get("local_path", "")).strip()
        if not value:
            return False, f"manifest row {index} is missing local_path"
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            return False, f"manifest row {index} has unsafe local_path: {value}"
        if value in seen_local_paths:
            return False, f"duplicate local_path in workspace manifest: {value}"
        seen_local_paths.add(value)
        local_path = (root / relative).resolve()
        try:
            local_path.relative_to(resolved_root)
        except ValueError:
            return False, f"manifest row {index} escapes workspace: {value}"
        if not local_path.is_file() or local_path.stat().st_size <= 0:
            return False, f"workspace image is missing or empty: {value}"

    return True, "complete"


def validate_qa_rows(qa_rows: Sequence[Mapping[str, object]]) -> tuple[list[str], list[str]]:
    if not 20 <= len(qa_rows) <= 100:
        raise RuntimeError(f"smoke QA must contain 20-100 rows, got {len(qa_rows)}")

    ids: list[str] = []
    assets: set[str] = set()
    for index, row in enumerate(qa_rows):
        query_id = str(row.get("id", "")).strip()
        image = str(row.get("image", "")).strip()
        asset = str(row.get("asset_category", "")).strip()
        answer = str(row.get("answer", "")).strip()
        missing = [
            name
            for name, value in (
                ("id", query_id),
                ("image", image),
                ("asset_category", asset),
                ("answer", answer),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(f"smoke QA row {index} is missing fields: {missing}")
        if answer not in {"normal", "anomalous"}:
            raise RuntimeError(
                f"smoke QA row {query_id} has unsupported answer: {answer!r}"
            )
        ids.append(query_id)
        assets.add(asset)

    if len(set(ids)) != len(ids):
        raise RuntimeError("smoke QA IDs must be unique")
    return ids, sorted(assets)


def build_workspace(
    *,
    build_dir: Path,
    qa_rows: Sequence[Mapping[str, object]],
    anomaly_manifest: Path,
    unsup_zip: Path,
    max_train_per_asset: int,
    seed: int,
    expected: Mapping[str, object],
) -> None:
    assets = [str(value) for value in expected["assets"]]
    unsupervised_rows = parse_unsupervised_manifest(anomaly_manifest)
    candidates_by_asset = leakage_safe_train_normal_candidates(unsupervised_rows, assets)
    selected_normals = {
        asset: deterministic_cap(
            candidates_by_asset[asset],
            max_train_per_asset,
            seed=seed,
            asset=asset,
        )
        for asset in assets
    }

    manifest_rows: list[dict[str, object]] = []
    with ZipFile(unsup_zip) as archive:
        for row in qa_rows:
            query_id = str(row["id"])
            asset = str(row["asset_category"])
            member = str(row["image"])
            out_path = build_dir / "queries" / asset / f"{query_id}{safe_suffix(member)}"
            extract_member(archive, member, out_path)
            manifest_rows.append(
                {
                    "role": "query",
                    "query_id": query_id,
                    "asset_category": asset,
                    "source_path": member,
                    "local_path": out_path.relative_to(build_dir).as_posix(),
                    "label_index": 0 if str(row["answer"]) == "normal" else 1,
                    "answer": row["answer"],
                    "split": "test",
                }
            )

        for asset in assets:
            for candidate in selected_normals[asset]:
                member = str(candidate["relative_path"])
                identity = hashlib.sha256(member.encode("utf-8")).hexdigest()[:16]
                out_path = (
                    build_dir
                    / "train"
                    / asset
                    / "good"
                    / f"{identity}{safe_suffix(member)}"
                )
                extract_member(archive, member, out_path)
                manifest_rows.append(
                    {
                        "role": "train_normal",
                        "query_id": "",
                        "asset_category": asset,
                        "source_path": member,
                        "local_path": out_path.relative_to(build_dir).as_posix(),
                        "label_index": 0,
                        "answer": "normal",
                        "split": "train",
                        "content_fingerprint": candidate["content_fingerprint"],
                    }
                )

    manifest_path = build_dir / "workspace_manifest.csv"
    config_path = build_dir / "workspace_config.json"
    write_csv(manifest_path, manifest_rows)
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
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    complete, reason = validate_workspace(build_dir, expected)
    if not complete:
        raise RuntimeError(f"new workspace failed validation: {reason}")


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
    for label, path in (
        ("anomaly manifest", args.anomaly_manifest),
        ("unsupervised ZIP", args.unsup_zip),
        ("smoke QA", args.qa),
    ):
        if not path.is_file():
            parser.error(f"{label} not found: {path}")

    qa_rows = read_jsonl(args.qa)
    validate_qa_rows(qa_rows)
    expected = expected_workspace_config(
        qa_rows=qa_rows,
        anomaly_manifest=args.anomaly_manifest,
        max_train_per_asset=args.max_train_per_asset,
        seed=args.seed,
    )

    if args.out_dir.exists() and not args.force:
        complete, reason = validate_workspace(args.out_dir, expected)
        if complete:
            print(f"Reusing complete workspace: {args.out_dir}")
            return 0
        print(f"Rebuilding stale/incomplete workspace ({reason}): {args.out_dir}")
        remove_path(args.out_dir)
    elif args.out_dir.exists():
        print(f"Force rebuilding workspace: {args.out_dir}")
        remove_path(args.out_dir)

    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{args.out_dir.name}.building-",
            dir=args.out_dir.parent,
        )
    )
    try:
        build_workspace(
            build_dir=build_dir,
            qa_rows=qa_rows,
            anomaly_manifest=args.anomaly_manifest,
            unsup_zip=args.unsup_zip,
            max_train_per_asset=args.max_train_per_asset,
            seed=args.seed,
            expected=expected,
        )
        if args.out_dir.exists():
            remove_path(args.out_dir)
        build_dir.replace(args.out_dir)
    except Exception:
        remove_path(build_dir)
        raise

    config = json.loads(
        (args.out_dir / "workspace_config.json").read_text(encoding="utf-8")
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print(f"Workspace ready: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

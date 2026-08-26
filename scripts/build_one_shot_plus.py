#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from insplad_benchmark.dataset import (  # noqa: E402
    leakage_safe_train_normal_candidates,
    parse_unsupervised_manifest,
)
from insplad_benchmark.reference_retrieval import (  # noqa: E402
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DinoV2Encoder,
    ImageEncoder,
    build_one_shot_plus_rows,
    embed_rows_with_cache,
    retrieve_top1_references,
    validate_retrieval_provenance,
)
from insplad_benchmark.sampling import balanced_binary_sample  # noqa: E402

ALIGNED_CORE_ASSETS = (
    "glass-insulator",
    "lightning-rod-suspension",
    "polymer-insulator-upper-shackle",
    "vari-grip",
    "yoke-suspension",
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def align_base_rows(
    base_rows: Sequence[Mapping[str, object]],
    aligned_core: Path | None,
) -> list[dict]:
    if aligned_core is None:
        return [dict(row) for row in base_rows]
    with Path(aligned_core).open("r", newline="", encoding="utf-8-sig") as handle:
        core_rows = list(csv.DictReader(handle))
    ordered_ids = [str(row.get("anomaly_id") or "").strip() for row in core_rows]
    if any(not row_id for row_id in ordered_ids):
        raise ValueError("aligned core CSV contains a row without anomaly_id")
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("aligned core CSV contains duplicate anomaly_id values")
    by_id = {str(row.get("id") or ""): row for row in base_rows}
    if len(by_id) != len(base_rows):
        raise ValueError("base fixed one-shot QA contains duplicate IDs")
    missing = [row_id for row_id in ordered_ids if row_id not in by_id]
    if missing:
        raise ValueError(f"aligned core IDs missing from base fixed QA: {missing[:5]}")
    return [dict(by_id[row_id]) for row_id in ordered_ids]


def _normalized_member(value: str) -> str:
    return value.replace("\\", "/").lstrip("/")


class ZipImageLoader:
    """Load requested images lazily without extracting the complete source ZIP."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._zip: ZipFile | None = None
        self._member_by_normalized_path: dict[str, str] = {}

    def __enter__(self) -> "ZipImageLoader":
        self._zip = ZipFile(self.path)
        for member in self._zip.namelist():
            normalized = _normalized_member(member)
            existing = self._member_by_normalized_path.get(normalized)
            if existing is not None and existing != member:
                raise ValueError(f"ambiguous normalized ZIP member path: {normalized}")
            self._member_by_normalized_path[normalized] = member
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._zip is not None:
            self._zip.close()
        self._zip = None
        self._member_by_normalized_path.clear()

    def __call__(self, row: Mapping[str, object]):
        if self._zip is None:
            raise RuntimeError("ZIP image loader is not open")
        path = str(row.get("relative_path") or row.get("image") or "")
        member = self._member_by_normalized_path.get(_normalized_member(path))
        if member is None:
            raise KeyError(f"image not found in ZIP: {path}")
        try:
            from PIL import Image, ImageOps
        except ImportError as exc:
            raise RuntimeError("DINOv2 retrieval requires Pillow") from exc
        with Image.open(io.BytesIO(self._zip.read(member))) as image:
            return ImageOps.exif_transpose(image).convert("RGB").copy()


def _manifest_index(rows: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    index: dict[str, Mapping[str, object]] = {}
    for row in rows:
        path = str(row["relative_path"])
        if path in index:
            raise ValueError(f"duplicate manifest image path: {path}")
        index[path] = row
    return index


def _embedding_rows(
    base_rows: Sequence[Mapping[str, object]],
    unsupervised_rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    manifest = _manifest_index(unsupervised_rows)
    assets = sorted({str(row.get("asset_category") or "") for row in base_rows})
    if not assets or "" in assets:
        raise ValueError("base one-shot QA rows must contain asset_category")
    candidates = leakage_safe_train_normal_candidates(unsupervised_rows, assets)
    selected: dict[str, Mapping[str, object]] = {}
    for base in base_rows:
        path = str(base.get("image") or "")
        meta = manifest.get(path)
        if meta is None:
            raise ValueError(f"{base.get('id')}: query image missing from manifest: {path}")
        selected[path] = meta
    for rows in candidates.values():
        for row in rows:
            selected[str(row["relative_path"])] = row
    return [selected[path] for path in sorted(selected)]


def validate_distinct_artifact_paths(**paths: Path | None) -> None:
    """Prevent any source, cache, or output from aliasing another argument."""

    seen: dict[str, tuple[str, Path]] = {}
    for label, value in paths.items():
        if value is None:
            continue
        resolved = Path(value).resolve()
        key = os.path.normcase(str(resolved))
        previous = seen.get(key)
        if previous is not None:
            previous_label, previous_path = previous
            raise ValueError(
                f"path collision: {label} and {previous_label} both resolve to "
                f"{previous_path}"
            )
        seen[key] = (label, resolved)


def resolve_model_revision(model_id: str, model_revision: str | None) -> str:
    if model_revision is None and model_id == DEFAULT_MODEL_ID:
        model_revision = DEFAULT_MODEL_REVISION
    if model_revision is None:
        raise ValueError("custom --model-id requires an immutable --model-revision")
    if re.fullmatch(r"[0-9a-fA-F]{40}", model_revision) is None:
        raise ValueError("--model-revision must be a 40-character immutable commit SHA")
    return model_revision.lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifacts(
    *,
    anomaly_manifest: Path,
    unsup_zip: Path,
    base_one_shot_qa: Path,
    aligned_core: Path | None = None,
    out_qa: Path,
    out_manifest: Path,
    cache: Path,
    encoder: ImageEncoder,
    batch_size: int = 16,
    retrieval_batch_size: int = 256,
    smoke_out: Path | None = None,
    smoke_size: int = 25,
    smoke_seed: int = 42,
    expected_query_count: int | None = 1378,
    expected_assets: Sequence[str] | None = ALIGNED_CORE_ASSETS,
) -> dict[str, object]:
    validate_distinct_artifact_paths(
        anomaly_manifest=anomaly_manifest,
        unsup_zip=unsup_zip,
        base_one_shot_qa=base_one_shot_qa,
        aligned_core=aligned_core,
        out_qa=out_qa,
        out_manifest=out_manifest,
        cache=cache,
        smoke_out=smoke_out,
    )
    unsupervised_rows = parse_unsupervised_manifest(Path(anomaly_manifest))
    all_base_rows = read_jsonl(Path(base_one_shot_qa))
    base_rows = align_base_rows(all_base_rows, aligned_core)
    if expected_query_count is not None and len(base_rows) != expected_query_count:
        raise ValueError(
            f"aligned fixed one-shot QA must contain {expected_query_count} rows, got {len(base_rows)}"
        )
    if not base_rows:
        raise ValueError("base fixed one-shot QA is empty")
    actual_assets = {str(row.get("asset_category") or "") for row in base_rows}
    if expected_assets is not None and actual_assets != set(expected_assets):
        raise ValueError(
            "aligned fixed one-shot QA asset set differs: "
            f"expected {sorted(set(expected_assets))}, got {sorted(actual_assets)}"
        )

    rows_to_embed = _embedding_rows(base_rows, unsupervised_rows)
    with ZipImageLoader(Path(unsup_zip)) as image_loader:
        embeddings = embed_rows_with_cache(
            rows_to_embed,
            encoder=encoder,
            image_loader=image_loader,
            cache_path=Path(cache),
            batch_size=batch_size,
        )
    cache_sha256 = sha256_file(Path(cache))
    matches = retrieve_top1_references(
        base_rows,
        unsupervised_rows,
        embeddings,
        batch_size=retrieval_batch_size,
        provenance=encoder.provenance,
        cache_sha256=cache_sha256,
    )
    plus_rows = build_one_shot_plus_rows(base_rows, matches)
    validate_retrieval_provenance(
        base_rows,
        plus_rows,
        matches,
        unsupervised_rows,
        expected_model_id=encoder.provenance.model_id,
        expected_model_revision=encoder.provenance.model_revision,
        expected_cache_sha256=cache_sha256,
    )

    write_jsonl(Path(out_qa), plus_rows)
    write_csv(Path(out_manifest), matches)
    smoke_rows: list[dict] = []
    if smoke_out is not None:
        smoke_rows = balanced_binary_sample(plus_rows, smoke_size, seed=smoke_seed)
        write_jsonl(Path(smoke_out), smoke_rows)

    return {
        "query_count": len(plus_rows),
        "source_base_count": len(all_base_rows),
        "asset_count": len({str(row["asset_category"]) for row in plus_rows}),
        "embedding_image_count": len(rows_to_embed),
        "smoke_count": len(smoke_rows),
        "model_id": encoder.provenance.model_id,
        "model_revision": encoder.provenance.model_revision,
        "cache_sha256": cache_sha256,
        "cache": str(cache),
        "out_qa": str(out_qa),
        "out_manifest": str(out_manifest),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe DINOv2 retrieved 1-shot+ QA for the Aligned Core."
    )
    parser.add_argument("--anomaly-manifest", type=Path, required=True)
    parser.add_argument("--unsup-zip", type=Path, required=True)
    parser.add_argument("--base-one-shot-qa", type=Path, required=True)
    parser.add_argument(
        "--aligned-core",
        type=Path,
        help="optional aligned_core.csv; selects rows by anomaly_id before retrieval",
    )
    parser.add_argument("--out-qa", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--smoke-out", type=Path)
    parser.add_argument("--smoke-size", type=int, default=25)
    parser.add_argument("--smoke-seed", type=int, default=42)
    parser.add_argument("--expected-query-count", type=int, default=1378)
    parser.add_argument("--expected-assets", nargs="+", default=list(ALIGNED_CORE_ASSETS))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--model-revision",
        help=(
            "immutable Hugging Face commit; defaults to the pinned DINOv2-base "
            f"revision {DEFAULT_MODEL_REVISION}"
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--retrieval-batch-size", type=int, default=256)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.retrieval_batch_size <= 0:
        parser.error("batch sizes must be positive")
    if args.smoke_out is not None and not 20 <= args.smoke_size <= 50:
        parser.error("--smoke-size must be between 20 and 50")
    if args.expected_query_count <= 0:
        parser.error("--expected-query-count must be positive")
    try:
        model_revision = resolve_model_revision(args.model_id, args.model_revision)
    except ValueError as exc:
        parser.error(str(exc))

    encoder = DinoV2Encoder(
        args.model_id,
        model_revision=model_revision,
        device=args.device,
    )
    try:
        summary = build_artifacts(
            anomaly_manifest=args.anomaly_manifest,
            unsup_zip=args.unsup_zip,
            base_one_shot_qa=args.base_one_shot_qa,
            aligned_core=args.aligned_core,
            out_qa=args.out_qa,
            out_manifest=args.out_manifest,
            cache=args.cache,
            encoder=encoder,
            batch_size=args.batch_size,
            retrieval_batch_size=args.retrieval_batch_size,
            smoke_out=args.smoke_out,
            smoke_size=args.smoke_size,
            smoke_seed=args.smoke_seed,
            expected_query_count=args.expected_query_count,
            expected_assets=args.expected_assets,
        )
    finally:
        encoder.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

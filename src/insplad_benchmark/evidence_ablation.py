from __future__ import annotations

import csv
import json
import math
import os
import uuid
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .vision_expert import normalize_anomaly_map


EVIDENCE_CONDITIONS = (
    "score",
    "bbox",
    "heatmap",
    "reference_score",
    "reference_bbox",
    "reference_heatmap",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    if not fields:
        _atomic_write_text(path, "")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    temporary.replace(path)


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def expert_score(row: Mapping[str, object]) -> float:
    for key in ("model_pred_score", "anomaly_score_raw", "anomaly_score_top1pct"):
        value = _finite_float(row.get(key))
        if value is not None:
            return value
    raise ValueError(f"no finite expert score in row for {row.get('expert')} / {row.get('query_id')}")


def percentile_rank(value: float, peers: Sequence[float]) -> float:
    finite = sorted(float(item) for item in peers if math.isfinite(float(item)))
    if not finite:
        raise ValueError("peers must contain at least one finite value")
    # Mid-rank handles ties deterministically and stays in [0, 100].
    less = sum(item < value for item in finite)
    equal = sum(item == value for item in finite)
    return 100.0 * (less + 0.5 * equal) / len(finite)


def _resolve_workspace_queries(workspace_root: Path) -> dict[str, Path]:
    rows = read_csv(Path(workspace_root) / "workspace_manifest.csv")
    result: dict[str, Path] = {}
    for row in rows:
        if row.get("role") != "query":
            continue
        query_id = str(row.get("query_id", ""))
        path = Path(workspace_root) / str(row.get("local_path", ""))
        if not query_id or not path.is_file():
            raise FileNotFoundError(f"workspace query missing: {query_id} -> {path}")
        result[query_id] = path
    return result


def _resolve_artifact(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else Path(root) / path


def make_bbox_overlay(image_path: Path, row: Mapping[str, object], out_path: Path) -> None:
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    left = int(float(row["roi_left"]))
    top = int(float(row["roi_top"]))
    right = int(float(row["roi_right"]))
    bottom = int(float(row["roi_bottom"]))
    draw = ImageDraw.Draw(image)
    width = max(3, round(min(image.size) / 180))
    draw.rectangle((left, top, right - 1, bottom - 1), outline=(255, 255, 0), width=width)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def make_heatmap_overlay(image_path: Path, anomaly_map_path: Path, out_path: Path) -> None:
    raw = np.load(anomaly_map_path, allow_pickle=False)
    normalized = normalize_anomaly_map(raw)
    heat = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    heat[..., 0] = np.rint(255 * normalized).astype(np.uint8)
    heat[..., 1] = np.rint(160 * np.square(normalized)).astype(np.uint8)
    heat_image = Image.fromarray(heat, mode="RGB")
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    heat_image = heat_image.resize(image.size, Image.Resampling.BILINEAR)
    overlay = Image.blend(image, heat_image, alpha=0.42)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)


def build_evidence_manifest(
    *,
    qa_path: Path,
    workspace_root: Path,
    expert_sources: Sequence[tuple[Path, Path]],
    out_dir: Path,
    expected_experts: Iterable[str],
) -> list[dict[str, object]]:
    qa_rows = read_jsonl(qa_path)
    qa_by_id = {str(row["id"]): row for row in qa_rows}
    query_paths = _resolve_workspace_queries(workspace_root)
    expected_experts = tuple(expected_experts)

    expert_rows: list[tuple[dict[str, str], Path]] = []
    for manifest_path, artifact_root in expert_sources:
        for row in read_csv(manifest_path):
            expert_rows.append((row, Path(artifact_root)))

    index: dict[tuple[str, str], tuple[dict[str, str], Path]] = {}
    for row, artifact_root in expert_rows:
        key = (str(row.get("query_id", "")), str(row.get("expert", "")))
        if key[0] not in qa_by_id or key[1] not in expected_experts:
            continue
        if key in index:
            raise ValueError(f"duplicate expert row: {key}")
        index[key] = (row, artifact_root)

    expected = {(query_id, expert) for query_id in qa_by_id for expert in expected_experts}
    missing = sorted(expected - set(index))
    if missing:
        raise ValueError(f"missing expert rows, first={missing[:10]}")

    scores_by_expert: dict[str, list[float]] = {expert: [] for expert in expected_experts}
    for (_, expert), (row, _) in index.items():
        scores_by_expert[expert].append(expert_score(row))

    out_dir = Path(out_dir)
    rows_out: list[dict[str, object]] = []
    for query_id in qa_by_id:
        gt = qa_by_id[query_id]
        query_path = query_paths[query_id]
        for expert in expected_experts:
            row, artifact_root = index[(query_id, expert)]
            raw_score = expert_score(row)
            percentile = percentile_rank(raw_score, scores_by_expert[expert])
            map_path = _resolve_artifact(artifact_root, row["anomaly_map_path"])
            if not map_path.is_file():
                raise FileNotFoundError(map_path)
            bbox_path = out_dir / "visual" / expert / f"{query_id}__bbox.png"
            heatmap_path = out_dir / "visual" / expert / f"{query_id}__heatmap_overlay.png"
            if not bbox_path.is_file():
                make_bbox_overlay(query_path, row, bbox_path)
            if not heatmap_path.is_file():
                make_heatmap_overlay(query_path, map_path, heatmap_path)
            rows_out.append(
                {
                    "query_id": query_id,
                    "expert": expert,
                    "asset_category": gt["asset_category"],
                    "answer": gt["answer"],
                    "query_image": gt["image"],
                    "reference_image": gt.get("reference_image", ""),
                    "expert_score_raw": raw_score,
                    "score_percentile_smoke": round(percentile, 4),
                    "score_calibration": "unlabeled_fixed_smoke_percentile_screening_only",
                    "bbox_overlay_path": bbox_path.relative_to(out_dir).as_posix(),
                    "heatmap_overlay_path": heatmap_path.relative_to(out_dir).as_posix(),
                    "roi_left": row["roi_left"],
                    "roi_top": row["roi_top"],
                    "roi_right": row["roi_right"],
                    "roi_bottom": row["roi_bottom"],
                    "source_expert_model_id": row.get("model_id", ""),
                    "source_expert_model_revision": row.get("model_revision", ""),
                }
            )

    write_csv_atomic(out_dir / "evidence_manifest.csv", rows_out)
    return rows_out


def select_best_fixed_by_model(
    rows: Sequence[Mapping[str, object]],
    *,
    allowed_conditions: Iterable[str] | None = None,
) -> dict[str, dict[str, object]]:
    allowed = set(allowed_conditions or [])
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        condition = str(row.get("condition", ""))
        if allowed and condition not in allowed:
            continue
        grouped.setdefault(str(row["model_key"]), []).append(row)

    winners: dict[str, dict[str, object]] = {}
    for model_key, candidates in grouped.items():
        if not candidates:
            continue
        ranked = sorted(
            candidates,
            key=lambda row: (
                float(row.get("f1", 0.0)),
                float(row.get("balanced_accuracy", 0.0)),
                -float(row.get("invalid_output_rate", 1.0)),
            ),
            reverse=True,
        )
        winners[model_key] = dict(ranked[0])
    return winners

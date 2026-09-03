from __future__ import annotations

import csv
import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class ROIBox:
    """Pixel-space crop rectangle using PIL's left/top/right/bottom convention."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _as_2d_float32(value: object) -> np.ndarray:
    """Convert common model anomaly-map layouts to one finite 2-D float array."""

    array = np.asarray(value, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim == 3:
        # Anomaly libraries commonly return CxHxW or HxWxC. Merge channels/maps
        # conservatively with max so a high response is not averaged away.
        if array.shape[0] <= min(array.shape[1:]):
            array = np.max(array, axis=0)
        elif array.shape[-1] <= min(array.shape[:2]):
            array = np.max(array, axis=-1)
    if array.ndim != 2:
        raise ValueError(f"anomaly map must reduce to 2-D, got shape={array.shape}")
    if array.size == 0:
        raise ValueError("anomaly map is empty")
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    fill = float(np.nanmedian(array[finite]))
    return np.where(finite, array, fill).astype(np.float32, copy=False)


def normalize_anomaly_map(value: object) -> np.ndarray:
    """Robustly normalize an anomaly map to [0, 1] without dataset GT."""

    array = _as_2d_float32(value)
    finite = array[np.isfinite(array)]
    low = float(np.percentile(finite, 1.0))
    high = float(np.percentile(finite, 99.5))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low + 1e-12:
        return np.zeros_like(array, dtype=np.float32)
    normalized = (array - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def anomaly_map_score(value: object, *, top_fraction: float = 0.01) -> float:
    """Return the mean of the hottest pixels as an auditable, threshold-free score."""

    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    normalized = normalize_anomaly_map(value)
    flat = normalized.reshape(-1)
    count = max(1, int(math.ceil(flat.size * top_fraction)))
    if count == flat.size:
        return float(flat.mean())
    hottest = np.partition(flat, flat.size - count)[-count:]
    return float(hottest.mean())


def _component_containing_peak(mask: np.ndarray, peak_y: int, peak_x: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    if not mask[peak_y, peak_x]:
        return np.array([peak_y]), np.array([peak_x])

    visited = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque([(peak_y, peak_x)])
    visited[peak_y, peak_x] = True
    ys: list[int] = []
    xs: list[int] = []

    while queue:
        y, x = queue.popleft()
        ys.append(y)
        xs.append(x)
        for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (
                0 <= next_y < height
                and 0 <= next_x < width
                and mask[next_y, next_x]
                and not visited[next_y, next_x]
            ):
                visited[next_y, next_x] = True
                queue.append((next_y, next_x))
    return np.asarray(ys), np.asarray(xs)


def _fit_square_box(
    center_x: float,
    center_y: float,
    side: float,
    image_width: int,
    image_height: int,
) -> ROIBox:
    side_int = max(2, min(int(math.ceil(side)), image_width, image_height))
    left = int(round(center_x - side_int / 2))
    top = int(round(center_y - side_int / 2))
    left = min(max(left, 0), image_width - side_int)
    top = min(max(top, 0), image_height - side_int)
    return ROIBox(left, top, left + side_int, top + side_int)


def select_roi_box(
    anomaly_map: object,
    image_size: tuple[int, int],
    *,
    quantile: float = 0.90,
    peak_fraction: float = 0.65,
    margin_fraction: float = 0.20,
    min_side_fraction: float = 0.22,
    max_side_fraction: float = 0.72,
) -> ROIBox:
    """Select one deterministic candidate ROI around the hottest connected region.

    This intentionally does not use anomaly ground truth. For flat maps it falls
    back to a centered crop, which keeps the experiment runnable while making the
    low-information map visible in the saved audit artifacts.
    """

    image_width, image_height = image_size
    if image_width <= 1 or image_height <= 1:
        raise ValueError(f"invalid image_size={image_size}")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    if not 0 < peak_fraction <= 1:
        raise ValueError("peak_fraction must be in (0, 1]")
    if not 0 <= margin_fraction <= 1:
        raise ValueError("margin_fraction must be in [0, 1]")
    if not 0 < min_side_fraction <= max_side_fraction <= 1:
        raise ValueError("side fractions must satisfy 0 < min <= max <= 1")

    normalized = normalize_anomaly_map(anomaly_map)
    map_height, map_width = normalized.shape
    peak_index = int(np.argmax(normalized))
    peak_y, peak_x = np.unravel_index(peak_index, normalized.shape)
    peak_value = float(normalized[peak_y, peak_x])

    min_side = min(image_width, image_height) * min_side_fraction
    max_side = min(image_width, image_height) * max_side_fraction

    if peak_value <= 1e-8:
        return _fit_square_box(
            image_width / 2,
            image_height / 2,
            min_side,
            image_width,
            image_height,
        )

    positive = normalized[normalized > 0]
    quantile_value = float(np.quantile(positive, quantile)) if positive.size else peak_value
    threshold = max(quantile_value, peak_value * peak_fraction)
    mask = normalized >= threshold
    ys, xs = _component_containing_peak(mask, int(peak_y), int(peak_x))

    map_left = int(xs.min())
    map_right = int(xs.max()) + 1
    map_top = int(ys.min())
    map_bottom = int(ys.max()) + 1

    left = map_left / map_width * image_width
    right = map_right / map_width * image_width
    top = map_top / map_height * image_height
    bottom = map_bottom / map_height * image_height

    component_width = max(1.0, right - left)
    component_height = max(1.0, bottom - top)
    side = max(component_width, component_height)
    side *= 1.0 + 2.0 * margin_fraction
    side = min(max(side, min_side), max_side)

    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    return _fit_square_box(center_x, center_y, side, image_width, image_height)


def _relative_artifact_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def save_roi_artifacts(
    *,
    image_path: Path,
    anomaly_map: object,
    out_dir: Path,
    query_id: str,
    expert: str,
    asset_category: str,
    query_image: str,
    metadata: Mapping[str, object] | None = None,
    artifact_root: Path | None = None,
) -> dict[str, object]:
    """Save raw map, normalized heatmap, overlay, crop, and one manifest row."""

    if not query_id:
        raise ValueError("query_id is required")
    if not expert:
        raise ValueError("expert is required")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = Path(artifact_root) if artifact_root is not None else out_dir

    raw = _as_2d_float32(anomaly_map)
    normalized = normalize_anomaly_map(raw)
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    box = select_roi_box(normalized, image.size)

    stem = f"{expert}__{query_id}"
    map_path = out_dir / f"{stem}.npy"
    heatmap_path = out_dir / f"{stem}__heatmap.png"
    overlay_path = out_dir / f"{stem}__overlay.png"
    roi_path = out_dir / f"{stem}__roi.png"

    np.save(map_path, raw, allow_pickle=False)

    heat = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    heat[..., 0] = np.rint(255 * normalized).astype(np.uint8)
    heat[..., 1] = np.rint(180 * np.square(normalized)).astype(np.uint8)
    heat_image = Image.fromarray(heat, mode="RGB")
    heat_image.save(heatmap_path)

    resized_heat = heat_image.resize(image.size, Image.Resampling.BILINEAR)
    overlay = Image.blend(image, resized_heat, alpha=0.42)
    draw = ImageDraw.Draw(overlay)
    line_width = max(2, round(min(image.size) / 220))
    draw.rectangle((box.left, box.top, box.right - 1, box.bottom - 1), outline=(255, 255, 0), width=line_width)
    overlay.save(overlay_path)

    image.crop((box.left, box.top, box.right, box.bottom)).save(roi_path)

    row: dict[str, object] = {
        "query_id": query_id,
        "expert": expert,
        "asset_category": asset_category,
        "query_image": query_image,
        "source_image_path": str(Path(image_path).resolve()),
        "anomaly_score_top1pct": anomaly_map_score(normalized),
        "roi_left": box.left,
        "roi_top": box.top,
        "roi_right": box.right,
        "roi_bottom": box.bottom,
        "roi_width": box.width,
        "roi_height": box.height,
        "anomaly_map_path": _relative_artifact_path(artifact_root, map_path),
        "heatmap_path": _relative_artifact_path(artifact_root, heatmap_path),
        "overlay_path": _relative_artifact_path(artifact_root, overlay_path),
        "roi_path": _relative_artifact_path(artifact_root, roi_path),
    }
    if metadata:
        for key, value in metadata.items():
            if key not in row:
                row[key] = value
    return row


def write_manifest(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def resolve_artifact_path(manifest_path: Path, value: str, artifact_root: Path | None = None) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    base = Path(artifact_root) if artifact_root is not None else Path(manifest_path).parent
    return base / candidate


def validate_roi_manifest(
    rows: Sequence[Mapping[str, object]],
    *,
    query_ids: Iterable[str],
    experts: Iterable[str],
    manifest_path: Path | None = None,
    artifact_root: Path | None = None,
    require_files: bool = True,
) -> None:
    expected_ids = set(query_ids)
    expected_experts = set(experts)
    expected = {(query_id, expert) for query_id in expected_ids for expert in expected_experts}
    actual: set[tuple[str, str]] = set()

    for index, row in enumerate(rows):
        key = (str(row.get("query_id", "")), str(row.get("expert", "")))
        if not all(key):
            raise ValueError(f"manifest row {index} is missing query_id/expert")
        if key in actual:
            raise ValueError(f"duplicate manifest key: {key}")
        actual.add(key)
        if require_files:
            if manifest_path is None:
                raise ValueError("manifest_path is required when require_files=True")
            for field in ("roi_path", "anomaly_map_path", "overlay_path"):
                value = str(row.get(field, ""))
                if not value:
                    raise ValueError(f"manifest row {key} missing {field}")
                path = resolve_artifact_path(manifest_path, value, artifact_root)
                if not path.is_file():
                    raise FileNotFoundError(f"missing {field} for {key}: {path}")

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            "ROI manifest coverage mismatch: "
            + json.dumps({"missing": missing[:20], "extra": extra[:20]}, ensure_ascii=False)
        )


def select_best_expert(
    leaderboard_rows: Sequence[Mapping[str, object]],
    *,
    required_models: Iterable[str] | None = None,
    condition: str = "roi",
) -> dict[str, object]:
    """Choose expert by model-mean F1, then BAcc, without peeking at full data."""

    required = set(required_models or [])
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in leaderboard_rows:
        if str(row.get("condition", "")) != condition:
            continue
        expert = str(row.get("expert", ""))
        if expert:
            grouped.setdefault(expert, []).append(row)

    summaries: list[dict[str, object]] = []
    for expert, rows in grouped.items():
        models = {str(row.get("model_key", "")) for row in rows}
        if required and models != required:
            continue
        f1_values = [float(row["f1"]) for row in rows]
        bacc_values = [float(row["balanced_accuracy"]) for row in rows]
        invalid_values = [float(row.get("invalid_output_rate", 0.0)) for row in rows]
        summaries.append(
            {
                "expert": expert,
                "model_n": len(models),
                "models": sorted(models),
                "mean_f1": sum(f1_values) / len(f1_values),
                "mean_balanced_accuracy": sum(bacc_values) / len(bacc_values),
                "mean_invalid_output_rate": sum(invalid_values) / len(invalid_values),
            }
        )
    if not summaries:
        raise ValueError("no complete expert rows are available for selection")
    summaries.sort(
        key=lambda row: (
            -float(row["mean_f1"]),
            -float(row["mean_balanced_accuracy"]),
            float(row["mean_invalid_output_rate"]),
            str(row["expert"]),
        )
    )
    winner = dict(summaries[0])
    winner["ranking"] = summaries
    winner["selection_condition"] = condition
    return winner

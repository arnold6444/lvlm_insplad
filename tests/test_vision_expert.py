from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from insplad_benchmark.vision_expert import (
    anomaly_map_score,
    normalize_anomaly_map,
    read_manifest,
    save_roi_artifacts,
    select_best_expert,
    select_roi_box,
    validate_roi_manifest,
    write_manifest,
)


def test_hotspot_roi_tracks_map_peak() -> None:
    anomaly_map = np.zeros((10, 10), dtype=np.float32)
    anomaly_map[1:3, 7:9] = 10.0
    box = select_roi_box(anomaly_map, (1000, 500))
    assert box.left > 500
    assert box.top < 250
    assert 0 < box.width <= 360
    assert box.width == box.height


def test_flat_map_uses_deterministic_center_crop() -> None:
    anomaly_map = np.full((8, 8), 4.0, dtype=np.float32)
    normalized = normalize_anomaly_map(anomaly_map)
    assert np.all(normalized == 0)
    assert anomaly_map_score(anomaly_map) == 0.0
    box = select_roi_box(anomaly_map, (200, 100))
    assert box == select_roi_box(anomaly_map, (200, 100))
    assert box.left == 89
    assert box.top == 39
    assert box.width == box.height == 22


def test_save_and_validate_roi_artifacts(tmp_path: Path) -> None:
    image_path = tmp_path / "query.png"
    Image.new("RGB", (120, 80), color=(40, 50, 60)).save(image_path)
    anomaly_map = np.zeros((16, 16), dtype=np.float32)
    anomaly_map[5:9, 8:12] = 2.0
    artifact_root = tmp_path / "results"
    row = save_roi_artifacts(
        image_path=image_path,
        anomaly_map=anomaly_map,
        out_dir=artifact_root / "vision_experts" / "patchcore" / "asset-a",
        query_id="q1",
        expert="patchcore",
        asset_category="asset-a",
        query_image="asset-a/test/good/q1.png",
        metadata={"model_revision": "test"},
        artifact_root=artifact_root,
    )
    manifest_path = artifact_root / "vision_experts" / "roi_manifest.csv"
    write_manifest(manifest_path, [row])
    loaded = read_manifest(manifest_path)
    validate_roi_manifest(
        loaded,
        query_ids=["q1"],
        experts=["patchcore"],
        manifest_path=manifest_path,
        artifact_root=artifact_root,
        require_files=True,
    )
    for field in ("roi_path", "anomaly_map_path", "overlay_path", "heatmap_path"):
        assert (artifact_root / loaded[0][field]).is_file()


def test_select_best_expert_requires_complete_model_set_and_tie_breaks() -> None:
    rows = [
        {"condition": "roi", "expert": "patchcore", "model_key": "qwen", "f1": 0.70, "balanced_accuracy": 0.71, "invalid_output_rate": 0.0},
        {"condition": "roi", "expert": "patchcore", "model_key": "internvl", "f1": 0.72, "balanced_accuracy": 0.73, "invalid_output_rate": 0.0},
        {"condition": "roi", "expert": "dinomaly", "model_key": "qwen", "f1": 0.74, "balanced_accuracy": 0.72, "invalid_output_rate": 0.0},
        {"condition": "roi", "expert": "dinomaly", "model_key": "internvl", "f1": 0.70, "balanced_accuracy": 0.78, "invalid_output_rate": 0.0},
        {"condition": "roi", "expert": "incomplete", "model_key": "qwen", "f1": 0.99, "balanced_accuracy": 0.99, "invalid_output_rate": 0.0},
    ]
    winner = select_best_expert(rows, required_models=["qwen", "internvl"])
    # Both complete experts average F1=0.72; Dinomaly wins on balanced accuracy.
    assert winner["expert"] == "dinomaly"
    assert [row["expert"] for row in winner["ranking"]] == ["dinomaly", "patchcore"]

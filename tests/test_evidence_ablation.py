from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from insplad_benchmark.evidence_ablation import (
    build_evidence_manifest,
    percentile_rank,
    select_best_fixed_by_model,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def test_percentile_rank_is_deterministic() -> None:
    assert percentile_rank(2.0, [1.0, 2.0, 3.0]) == 50.0
    assert percentile_rank(1.0, [1.0, 1.0, 3.0]) == 100.0 / 3.0


def test_build_evidence_manifest_creates_bbox_and_heatmap(tmp_path: Path) -> None:
    qa = [
        {"id": "q1", "image": "a/q1.png", "reference_image": "a/ref.png", "answer": "normal", "asset_category": "a"},
        {"id": "q2", "image": "a/q2.png", "reference_image": "a/ref.png", "answer": "anomalous", "asset_category": "a"},
    ]
    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text("".join(json.dumps(row) + "\n" for row in qa), encoding="utf-8")
    workspace = tmp_path / "workspace"
    rows = []
    for index, query_id in enumerate(("q1", "q2"), 1):
        rel = f"queries/a/{query_id}.png"
        path = workspace / rel; path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), (20 * index, 30, 40)).save(path)
        rows.append({"role": "query", "query_id": query_id, "asset_category": "a", "source_path": f"a/{query_id}.png", "local_path": rel})
    _write_csv(workspace / "workspace_manifest.csv", rows)

    base_root = tmp_path / "base"; recon_root = tmp_path / "recon"
    base_manifest = base_root / "roi.csv"; recon_manifest = recon_root / "roi.csv"
    experts = ["patchcore", "reconpatch_lite"]
    source_rows = {"patchcore": [], "reconpatch_lite": []}
    for expert, root in (("patchcore", base_root), ("reconpatch_lite", recon_root)):
        for index, query_id in enumerate(("q1", "q2"), 1):
            map_path = root / f"{expert}-{query_id}.npy"; map_path.parent.mkdir(parents=True, exist_ok=True)
            array = np.zeros((8, 8), dtype=np.float32); array[index:index+2, index:index+2] = index
            np.save(map_path, array)
            source_rows[expert].append({
                "query_id": query_id, "expert": expert, "roi_left": 5, "roi_top": 6,
                "roi_right": 25, "roi_bottom": 26,
                "anomaly_map_path": map_path.relative_to(root).as_posix(),
                "model_pred_score": float(index), "model_id": expert, "model_revision": "x",
            })
    _write_csv(base_manifest, source_rows["patchcore"]); _write_csv(recon_manifest, source_rows["reconpatch_lite"])
    out = tmp_path / "out"
    built = build_evidence_manifest(
        qa_path=qa_path, workspace_root=workspace,
        expert_sources=[(base_manifest, base_root), (recon_manifest, recon_root)],
        out_dir=out, expected_experts=experts,
    )
    assert len(built) == 4
    assert (out / built[0]["bbox_overlay_path"]).is_file()
    assert (out / built[0]["heatmap_overlay_path"]).is_file()
    assert {float(row["score_percentile_smoke"]) for row in built} == {25.0, 75.0}


def test_select_best_fixed_is_per_model() -> None:
    rows = [
        {"model_key": "a", "condition": "score", "expert": "p", "f1": 0.6, "balanced_accuracy": 0.7, "invalid_output_rate": 0.0},
        {"model_key": "a", "condition": "heatmap", "expert": "d", "f1": 0.8, "balanced_accuracy": 0.7, "invalid_output_rate": 0.0},
        {"model_key": "b", "condition": "score", "expert": "p", "f1": 0.9, "balanced_accuracy": 0.6, "invalid_output_rate": 0.0},
    ]
    winners = select_best_fixed_by_model(rows)
    assert winners["a"]["expert"] == "d"
    assert winners["b"]["expert"] == "p"

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types
from pathlib import Path
from zipfile import ZipFile

from PIL import Image


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_agent_runner_end_to_end_with_fake_adapter(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts/run_agent_iad_lite_smoke.py"

    fake_hf = types.ModuleType("insplad_benchmark.hf_adapters")
    fake_hf.infer_hf = lambda *args, **kwargs: ""
    fake_hf.load_hf_adapter = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, "insplad_benchmark.hf_adapters", fake_hf)

    spec = importlib.util.spec_from_file_location("agent_runner", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    workspace = tmp_path / "workspace"
    queries_dir = workspace / "queries"
    queries_dir.mkdir(parents=True)
    q1 = queries_dir / "q1.png"
    q2 = queries_dir / "q2.png"
    Image.new("RGB", (16, 16), "white").save(q1)
    Image.new("RGB", (16, 16), "gray").save(q2)
    _write_csv(
        workspace / "workspace_manifest.csv",
        [
            {"role": "query", "query_id": "q1", "local_path": "queries/q1.png"},
            {"role": "query", "query_id": "q2", "local_path": "queries/q2.png"},
        ],
    )

    qa = tmp_path / "qa.jsonl"
    qa.write_text(
        json.dumps(
            {
                "id": "q1",
                "asset_category": "a",
                "answer": "anomalous",
                "image": "q1.png",
                "reference_image": "refs/r1.png",
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "q2",
                "asset_category": "a",
                "answer": "normal",
                "image": "q2.png",
                "reference_image": "refs/r2.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    evidence = tmp_path / "evidence.csv"
    _write_csv(
        evidence,
        [
            {"query_id": "q1", "expert": "patchcore", "score_percentile_smoke": "95"},
            {"query_id": "q2", "expert": "patchcore", "score_percentile_smoke": "50"},
        ],
    )

    artifact_root = tmp_path / "artifacts"
    roi_dir = artifact_root / "vision_experts/patchcore/a"
    roi_dir.mkdir(parents=True)
    roi1 = roi_dir / "q1_roi.png"
    roi2 = roi_dir / "q2_roi.png"
    Image.new("RGB", (8, 8), "red").save(roi1)
    Image.new("RGB", (8, 8), "blue").save(roi2)
    roi_manifest = tmp_path / "roi.csv"
    _write_csv(
        roi_manifest,
        [
            {
                "query_id": "q1",
                "expert": "patchcore",
                "roi_path": "vision_experts/patchcore/a/q1_roi.png",
            },
            {
                "query_id": "q2",
                "expert": "patchcore",
                "roi_path": "vision_experts/patchcore/a/q2_roi.png",
            },
        ],
    )

    unsup = tmp_path / "unsup.zip"
    with ZipFile(unsup, "w") as archive:
        for name, color in [("refs/r1.png", "white"), ("refs/r2.png", "gray")]:
            image_path = tmp_path / name.replace("/", "_")
            Image.new("RGB", (16, 16), color).save(image_path)
            archive.write(image_path, name)

    monkeypatch.setattr(module, "load_hf_adapter", lambda spec: object())

    def fake_infer(adapter, images, prompt):
        if len(images) == 3:
            return '{"anomaly":"anomalous","confidence":0.95,"request_tool":"none"}'
        return '{"anomaly":"normal","confidence":0.95,"request_tool":"none"}'

    monkeypatch.setattr(module, "infer_hf", fake_infer)

    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--qa",
            str(qa),
            "--workspace-root",
            str(workspace),
            "--unsup-zip",
            str(unsup),
            "--evidence-manifest",
            str(evidence),
            "--patchcore-roi-manifest",
            str(roi_manifest),
            "--patchcore-artifact-root",
            str(artifact_root),
            "--out-dir",
            str(out),
            "--image-cache",
            str(tmp_path / "image_cache"),
            "--checkpoint-every",
            "1",
            "--retry-attempts",
            "1",
        ],
    )
    assert module.main() == 0

    rows = list(csv.DictReader((out / "agent_leaderboard.csv").open(encoding="utf-8")))
    by_variant = {row["variant"]: row for row in rows}
    assert float(by_variant["agent_ref_first"]["f1"]) == 1.0
    assert float(by_variant["agent_query_first"]["f1"]) == 1.0
    assert float(by_variant["agent_ref_first"]["zoom_call_rate"]) == 0.5
    assert float(by_variant["agent_query_first"]["reference_call_rate"]) == 0.5
    assert (out / "agent_predictions.jsonl").is_file()

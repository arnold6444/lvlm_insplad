from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from insplad_benchmark.sampling import balanced_binary_sample


def _load_script(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_expanded_dev_sample_contains_original_smoke_prefix() -> None:
    rows = []
    assets = ["a", "b", "c", "d", "e"]
    for index in range(300):
        rows.append(
            {
                "id": f"q{index:03d}",
                "asset_category": assets[index % len(assets)],
                "answer": "normal" if index < 150 else "anomalous",
            }
        )
    smoke = balanced_binary_sample(rows, 50, seed=42)
    dev = balanced_binary_sample(rows, 150, seed=42)
    assert len(dev) == 150
    assert {row["id"] for row in smoke} <= {row["id"] for row in dev}
    assert sum(row["answer"] == "normal" for row in dev) == 75
    assert sum(row["answer"] == "anomalous" for row in dev) == 75


def test_dev_workspace_validator_accepts_150_rows() -> None:
    module = _load_script("build_vision_expert_dev_data.py")
    rows = [
        {
            "id": f"q{index}",
            "asset_category": f"asset-{index % 5}",
            "image": f"asset-{index % 5}/test/query-{index}.jpg",
            "answer": "normal" if index % 2 == 0 else "anomalous",
        }
        for index in range(150)
    ]
    ids, assets = module.validate_dev_qa(rows)
    assert len(ids) == 150
    assert len(assets) == 5


def test_ref_first_runner_skips_query_only_calls(tmp_path, monkeypatch) -> None:
    module = _load_script("run_agent_iad_lite_ref_first.py")

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

    artifacts = tmp_path / "artifacts"
    roi_dir = artifacts / "vision_experts/patchcore/a"
    roi_dir.mkdir(parents=True)
    for qid in ("q1", "q2"):
        Image.new("RGB", (8, 8), "red").save(roi_dir / f"{qid}.png")
    roi_manifest = tmp_path / "roi.csv"
    _write_csv(
        roi_manifest,
        [
            {
                "query_id": qid,
                "expert": "patchcore",
                "roi_path": f"vision_experts/patchcore/a/{qid}.png",
            }
            for qid in ("q1", "q2")
        ],
    )

    unsup = tmp_path / "unsup.zip"
    with ZipFile(unsup, "w") as archive:
        for name, color in (("refs/r1.png", "white"), ("refs/r2.png", "gray")):
            image_path = tmp_path / name.replace("/", "_")
            Image.new("RGB", (16, 16), color).save(image_path)
            archive.write(image_path, name)

    monkeypatch.setattr(module.base, "load_hf_adapter", lambda spec: object())
    call_sizes: list[int] = []

    def fake_infer(adapter, images, prompt):
        call_sizes.append(len(images))
        if len(images) == 3:
            return '{"anomaly":"anomalous","confidence":0.95,"request_tool":"none"}'
        return '{"anomaly":"normal","confidence":0.95,"request_tool":"none"}'

    monkeypatch.setattr(module.base, "infer_hf", fake_infer)
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPTS / "run_agent_iad_lite_ref_first.py"),
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
            str(artifacts),
            "--out-dir",
            str(out),
            "--image-cache",
            str(tmp_path / "cache"),
            "--retry-attempts",
            "1",
        ],
    )
    assert module.main() == 0
    assert 1 not in call_sizes

    rows = list(csv.DictReader((out / "agent_leaderboard.csv").open(encoding="utf-8")))
    by_variant = {row["variant"]: row for row in rows}
    assert set(by_variant) == {"fixed_reference_agent_prompt", "agent_ref_first"}
    assert float(by_variant["agent_ref_first"]["f1"]) == 1.0
    assert float(by_variant["agent_ref_first"]["zoom_call_rate"]) == 0.5

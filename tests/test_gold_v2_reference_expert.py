from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gold_v2_reference_expert", ROOT / "scripts" / "run_gold_v2_reference_expert.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def _row(i: int, qtype: str = "anomaly_detection") -> dict:
    return {
        "id": f"core-{i}-{qtype}",
        "core_id": f"core-{i}",
        "asset_category": "vari-grip",
        "image": f"sup/{i}.jpg",
        "normal_reference_image": f"unsup/train/good/{i % 2}.jpg",
        "question_type": qtype,
        "question": "Normal or anomalous?",
        "options": {"A": "Normal.", "B": "Anomalous."},
        "answer": "A",
    }


def test_parse_pairs_validates_and_dedupes():
    assert RUNNER.parse_pairs(["anomalyclip:score", "anomalyclip:score", "patchcore:roi"]) == [
        ("anomalyclip", "score"),
        ("patchcore", "roi"),
    ]
    with pytest.raises(ValueError):
        RUNNER.parse_pairs(["anomalyclip"])
    with pytest.raises(ValueError):
        RUNNER.parse_pairs(["unknown:score"])


def test_prompt_mentions_third_image_only_with_visual_evidence():
    row = _row(0)
    assert "Image 3" not in RUNNER.build_prompt(row, "score", "anomalyclip", 91.0)
    assert "91.0th percentile" in RUNNER.build_prompt(row, "score", "anomalyclip", 91.0)
    for condition in ("roi", "heatmap", "score_roi", "score_heatmap"):
        text = RUNNER.build_prompt(row, condition, "patchcore", 50.0)
        assert "Image 1 is a verified NORMAL reference" in text
        assert "Image 3" in text
    assert "yellow box" not in RUNNER.build_prompt(row, "heatmap", "patchcore", 50.0)


def test_images_put_normal_reference_first(tmp_path):
    manifest = tmp_path / "m.csv"
    erow = {"roi_path": str(tmp_path / "roi.png"), "overlay_path": str(tmp_path / "heat.png")}
    normal, target = tmp_path / "n.jpg", tmp_path / "t.jpg"
    assert RUNNER.build_images("score", normal, target, erow, manifest, tmp_path) == [normal, target]
    roi = RUNNER.build_images("roi", normal, target, erow, manifest, tmp_path)
    heat = RUNNER.build_images("score_heatmap", normal, target, erow, manifest, tmp_path)
    assert roi[:2] == [normal, target] and roi[2].name == "roi.png"
    assert heat[:2] == [normal, target] and heat[2].name == "heat.png"


def test_normal_reference_by_core_requires_every_core():
    rows = [_row(0), _row(1)]
    assert RUNNER.normal_reference_by_core(rows)["core-1"] == "unsup/train/good/1.jpg"
    broken = [_row(0), {**_row(1), "normal_reference_image": ""}]
    with pytest.raises(RuntimeError):
        RUNNER.normal_reference_by_core(broken)


def test_run_pair_reuses_predictions_and_loads_model_once(tmp_path, monkeypatch):
    import insplad_benchmark.hf_adapters as hf_adapters

    seen_images = []
    monkeypatch.setattr(
        hf_adapters, "infer_hf", lambda adapter, images, prompt: seen_images.append(images) or "B"
    )
    loads = []
    model_key = next(k for k, s in RUNNER.base.MODEL_REGISTRY.items() if s["backend"] == "hf")
    holder = RUNNER.base.LazyHFAdapter(
        RUNNER.base.MODEL_REGISTRY[model_key], loader=lambda spec: loads.append(1) or object()
    )
    rows = [_row(0), _row(1)]
    expert_index = {
        (row["core_id"], "anomalyclip"): {
            "score_percentile_full": "75.0",
            "roi_path": str(tmp_path / "roi.png"),
            "overlay_path": str(tmp_path / "heat.png"),
        }
        for row in rows
    }
    target_paths = {row["image"]: tmp_path / row["image"] for row in rows}
    normal_by_core = RUNNER.normal_reference_by_core(rows)
    normal_paths = {m: tmp_path / m for m in normal_by_core.values()}
    args = (
        model_key, "anomalyclip", "roi", rows, expert_index, target_paths, normal_paths,
        normal_by_core, tmp_path / "m.csv", tmp_path, tmp_path / "out", 20, holder,
    )
    pred_path, status = RUNNER.run_pair(*args)
    assert status["called_this_run"] == 2 and len(loads) == 1
    assert seen_images[0][0] == normal_paths["unsup/train/good/0.jpg"]
    saved = [json.loads(x) for x in pred_path.read_text(encoding="utf-8").splitlines()]
    assert {r["predicted_choice"] for r in saved} == {"B"}
    assert {r["regime"] for r in saved} == {RUNNER.REGIME}

    _, again = RUNNER.run_pair(*args)
    assert again["called_this_run"] == 0 and again["reused_predictions"] == 2
    assert len(loads) == 1


def test_comparison_reports_deltas_against_each_control():
    leaderboard = [{"model_key": "m", "expert": "anomalyclip", "condition": "score", "anomaly_f1": "0.8"}]
    comparison = RUNNER.build_comparison(
        leaderboard,
        [{"model_key": "m", "anomaly_f1": "0.6"}],
        [
            {"model_key": "m", "condition": "normal_reference", "anomaly_f1": "0.75"},
            {"model_key": "m", "condition": "positive_reference", "anomaly_f1": "0.1"},
        ],
        [{"model_key": "m", "expert": "anomalyclip", "condition": "score", "anomaly_f1": "0.5"}],
    )
    row = comparison[0]
    assert row["anomaly_f1_delta_vs_image_only"] == pytest.approx(0.2)
    assert row["anomaly_f1_delta_vs_normal_reference"] == pytest.approx(0.05)
    assert row["anomaly_f1_delta_vs_expert_only"] == pytest.approx(0.3)


def test_main_end_to_end_with_fake_model(tmp_path, monkeypatch):
    import csv
    import sys
    import zipfile

    import insplad_benchmark.hf_adapters as hf_adapters

    sys.path.insert(0, str(ROOT / "tests"))
    from test_gold_baselines import _master_rows

    loads = []
    monkeypatch.setattr(hf_adapters, "load_hf_adapter", lambda spec: loads.append(spec) or object())
    monkeypatch.setattr(hf_adapters, "infer_hf", lambda adapter, images, prompt: "A")

    master = _master_rows()[:492]  # 246 cores x (Q1, Q2)
    for row in master:
        row["normal_reference_image"] = "unsup/train/good/ref.jpg"
    qa_dir = tmp_path / "qa"
    qa_dir.mkdir()
    (qa_dir / "master_qa_v2.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in master), encoding="utf-8"
    )
    sup_zip, unsup_zip = tmp_path / "sup.zip", tmp_path / "unsup.zip"
    with zipfile.ZipFile(sup_zip, "w") as zf:
        for member in sorted({r["image"] for r in master}):
            zf.writestr(member, b"img")
    with zipfile.ZipFile(unsup_zip, "w") as zf:
        zf.writestr("unsup/train/good/ref.jpg", b"ref")

    art = tmp_path / "art"
    art.mkdir()
    (art / "roi.png").write_bytes(b"x")
    (art / "heat.png").write_bytes(b"x")
    manifest = tmp_path / "roi_manifest_all.csv"
    cores = sorted({r["core_id"] for r in master})
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_id", "expert", "asset_category", "model_pred_score", "roi_path", "overlay_path"],
        )
        writer.writeheader()
        for i, core in enumerate(cores):
            writer.writerow({
                "query_id": core, "expert": "anomalyclip", "asset_category": "vari-grip",
                "model_pred_score": i, "roi_path": "roi.png", "overlay_path": "heat.png",
            })

    model_key = next(k for k, s in RUNNER.base.MODEL_REGISTRY.items() if s["backend"] == "hf")
    out = tmp_path / "out"
    argv = [
        "run", "--qa-dir", str(qa_dir), "--sup-zip", str(sup_zip), "--unsup-zip", str(unsup_zip),
        "--roi-manifest", str(manifest), "--artifact-root", str(art), "--out-dir", str(out),
        "--pairs", "anomalyclip:score", "anomalyclip:roi", "--models", model_key,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert RUNNER.main() == 0
    assert len(loads) == 1
    with (out / "leaderboard.csv").open(encoding="utf-8") as handle:
        board = list(csv.DictReader(handle))
    assert {(r["expert"], r["condition"]) for r in board} == {("anomalyclip", "score"), ("anomalyclip", "roi")}
    assert all(float(r["valid_output_rate"]) == 1.0 for r in board)
    assert (out / "comparison.csv").is_file()
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["pairs"] == ["anomalyclip:score", "anomalyclip:roi"]

    # Re-run: everything reused, model never loaded again.
    assert RUNNER.main() == 0
    assert len(loads) == 1

from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CONS = _load("gold_v2_consistency_metrics_t", "run_gold_v2_consistency_metrics.py")
LOC = _load("gold_v2_loc_mmad_eval_t", "run_gold_v2_loc_mmad_eval.py")
DEFECT = {"A": "Normal / no defect.", "B": "Rust / corrosion.",
          "C": "Missing cap / missing insulator component.", "D": "Bird nest / foreign nesting material."}
LETTER = {"normal": "A", "rust": "B"}
MODELS = ["qwen3_5_4b", "internvl3_5_8b"]


def _gold(n: int = 6) -> list[dict]:
    rows = []
    for i in range(n):
        anomalous = i % 2 == 1
        core = f"core-{i}"
        defect = "rust" if anomalous else "normal"
        common = {"core_id": core, "asset_category": "vari-grip", "image": f"sup/{i}.png",
                  "normal_reference_image": f"unsup/good/{i % 2}.png",
                  "anomaly_gt": "anomalous" if anomalous else "normal", "defect_gt": defect}
        rows.append({**common, "id": f"{core}-q1", "question_type": "anomaly_detection",
                     "question": "Is the target power-line asset normal or anomalous?",
                     "options": {"A": "Normal.", "B": "Anomalous."}, "answer": "B" if anomalous else "A"})
        rows.append({**common, "id": f"{core}-q2", "question_type": "defect_classification",
                     "question": "Which condition best describes the target power-line asset?",
                     "options": dict(DEFECT), "answer": LETTER[defect]})
    return rows


def test_consistency_row_definitions():
    rows = _gold(4)          # core-0 normal, core-1 rust, core-2 normal, core-3 rust
    pairs = CONS.image_pairs(rows)
    choice = {
        "core-0-q1": "A", "core-0-q2": "B",   # normal + rust -> contradiction, false defect
        "core-1-q1": "B", "core-1-q2": "B",   # correct both -> joint
        "core-2-q1": "A", "core-2-q2": None,  # invalid Q2 -> not a valid pair, not joint
        "core-3-q1": "A", "core-3-q2": "A",   # consistent but wrong
    }
    r = CONS.consistency_row(pairs, choice)
    assert r["images"] == 4 and r["valid_pairs"] == 3 and r["contradictions"] == 1
    assert r["consistency_error"] == round(1 / 3, 6)
    assert r["joint_accuracy"] == 0.25
    assert r["false_defect_rate"] == 0.5
    assert r["q2_invalid_rate"] == 0.25


def test_consistency_main_discovers_all_condition_folders(tmp_path):
    rows = _gold(4)
    ids = [r["id"] for r in rows]
    correct = {r["id"]: r["answer"] for r in rows}

    def jsonl(path, recs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(x) for x in recs) + "\n", encoding="utf-8")

    img = tmp_path / "baseline.csv"
    with img.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["model_key", "id", "predicted_choice"])
        w.writeheader()
        for m in MODELS:
            for i in ids:
                w.writerow({"model_key": m, "id": i, "predicted_choice": correct[i]})
    for m in MODELS:
        jsonl(tmp_path / "ref" / "normal_reference" / "predictions" / f"{m}.jsonl",
              [{"id": i, "predicted_choice": correct[i]} for i in ids])
        jsonl(tmp_path / "vx" / m / "patchcore__score.jsonl", [{"id": i, "predicted_choice": "A"} for i in ids])
        jsonl(tmp_path / "cb" / m / "patchcore__score.jsonl", [{"id": i, "predicted_choice": correct[i]} for i in ids[:3]])
    out = tmp_path / "out"
    assert CONS.main(["--qa-dir", str(tmp_path), "--image-only-csv", str(img), "--reference-dir", str(tmp_path / "ref"),
                      "--vx-pred-dir", str(tmp_path / "vx"), "--combined-dir", str(tmp_path / "cb"),
                      "--out-dir", str(out), "--models", *MODELS], load_qa=lambda _: rows) == 0
    with (out / "consistency_leaderboard.csv").open(encoding="utf-8") as h:
        lb = {(r["model_key"], r["condition"]): r for r in csv.DictReader(h)}
    assert set(c for _, c in lb) == {"image_only", "normal_reference", "vx:patchcore:score"}
    assert float(lb[("qwen3_5_4b", "image_only")]["joint_accuracy"]) == 1.0
    assert float(lb[("qwen3_5_4b", "image_only")]["consistency_error"]) == 0.0
    # "A" everywhere: Q1 normal + Q2 normal -> consistent, but rust images wrong
    assert float(lb[("qwen3_5_4b", "vx:patchcore:score")]["joint_accuracy"]) == 0.5
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert {s["condition"] for s in meta["skipped_incomplete"]} == {"ref+patchcore:score"}   # incomplete file skipped


def _loc_rows() -> list[dict]:
    rows = []
    for i in (1, 3, 5):
        rows.append({"id": f"core-{i}-q3m", "core_id": f"core-{i}", "asset_category": "vari-grip", "image": f"sup/{i}.png",
                     "normal_reference_image": f"unsup/good/{i % 2}.png", "anomaly_gt": "anomalous", "defect_gt": "rust",
                     "question_type": "defect_localization",
                     "question": "There is a defect in the target power-line asset. Where is the defect located in the target image?",
                     "options": {"A": "Center region.", "B": "Upper-left region.", "C": "Lower-right region.", "D": "Middle-left region."},
                     "answer": "A", "answer_text": "Center region.", "protocol": "gold_v2_loc_mmad",
                     "localization_gt": {"basis": "manual_box_annotation" if i > 1 else "insplad_seg_instance_annotation",
                                         "coarse_region": "center", "normalized_bboxes_xywh": [[0.4, 0.4, 0.2, 0.2]]}})
    return rows


def test_loc_prompts_match_original_builders():
    row = _loc_rows()[0]
    import run_gold_baselines as base
    import run_gold_v2_reference_ablation as ref
    import run_gold_v2_reference_expert as rx
    import run_gold_v2_vision_expert_full as vx
    assert LOC.build_prompt(row, "image_only", "patchcore", None) == base.prompt_for(row)
    assert LOC.build_prompt(row, "normal_ref", "patchcore", None) == ref.prompt_for_condition(row, "normal_reference")
    assert LOC.build_prompt(row, "expert_score", "patchcore", 91.0) == vx.prompt(row, "score", "patchcore", 91.0)
    assert LOC.build_prompt(row, "combined", "patchcore", 91.0) == rx.build_prompt(row, "score", "patchcore", 91.0)
    t, n = Path("t.png"), Path("n.png")
    assert LOC.images_for("image_only", t, n) == [t] and LOC.images_for("expert_score", t, n) == [t]
    assert LOC.images_for("normal_ref", t, n) == [n, t] and LOC.images_for("combined", t, n) == [n, t]


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), (100, 100, 100)).save(buf, format="PNG")
    return buf.getvalue()


def test_loc_eval_end_to_end_with_fake_model(tmp_path):
    gold, loc = _gold(6), _loc_rows()
    sup, unsup = tmp_path / "sup.zip", tmp_path / "unsup.zip"
    with zipfile.ZipFile(sup, "w") as z:
        for name in sorted({r["image"] for r in gold}):
            z.writestr(name, _png())
    with zipfile.ZipFile(unsup, "w") as z:
        for name in sorted({r["normal_reference_image"] for r in gold}):
            z.writestr(name, _png())
    manifest = tmp_path / "roi_manifest_all.csv"
    with manifest.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["query_id", "expert", "asset_category", "anomaly_score_top1pct"])
        w.writeheader()
        for i in range(6):
            w.writerow({"query_id": f"core-{i}", "expert": "patchcore", "asset_category": "vari-grip",
                        "anomaly_score_top1pct": str(i)})
    loc_path = tmp_path / "localization_mmad_qa_3.jsonl"
    loc_path.write_text("\n".join(json.dumps(r) for r in loc) + "\n", encoding="utf-8")
    calls = []

    def infer(adapter, images, prompt):
        calls.append((len(images), prompt))
        return "A" if len(images) == 2 else "B"     # reference conditions right, single-image conditions wrong

    out = tmp_path / "out"
    argv = ["--loc-qa", str(loc_path), "--qa-dir", str(tmp_path), "--sup-zip", str(sup), "--unsup-zip", str(unsup),
            "--roi-manifest", str(manifest), "--out-dir", str(out), "--models", *MODELS]
    assert LOC.main(argv, load_qa=lambda _: gold, loader=lambda s: {}, infer=infer) == 0
    with (out / "leaderboard.csv").open(encoding="utf-8") as h:
        lb = {(r["model_key"], r["condition"]): r for r in csv.DictReader(h)}
    assert float(lb[("qwen3_5_4b", "image_only")]["loc_accuracy"]) == 0.0
    assert float(lb[("qwen3_5_4b", "combined")]["loc_accuracy"]) == 1.0
    assert int(lb[("qwen3_5_4b", "normal_ref")]["gain_vs_image_only"]) == 3
    pred = [json.loads(x) for x in (out / "predictions" / "qwen3_5_4b" / "expert_score.jsonl").read_text().splitlines()]
    assert [p["score_percentile_full"] for p in pred] == [pytest_approx(2 / 6 * 100), pytest_approx(4 / 6 * 100), 100.0]
    assert any("percentile" in p for _, p in calls)
    with (out / "metrics_by_group.csv").open(encoding="utf-8") as h:
        assert {r["group"] for r in csv.DictReader(h)} == {"asset_category", "defect_gt", "label_basis"}
    n = len(calls)
    assert LOC.main(argv, load_qa=lambda _: gold, loader=lambda s: {}, infer=infer) == 0
    assert len(calls) == n                                   # rerun reuses everything


def pytest_approx(v):
    import pytest
    return pytest.approx(v)


def test_eval_notebook_runs_consistency_then_location():
    nb = json.loads((ROOT / "notebooks" / "lvlm_gold_v2_loc_mmad_eval_colab.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for i, src in enumerate(cells):
        compile(src, f"loc_eval_nb:{i}", "exec")
    code = "\n".join(cells)
    assert code.index("run_gold_v2_consistency_metrics.py") < code.index("run_gold_v2_loc_mmad_eval.py")
    assert '"--prefix", "gold_v2_consistency_metrics"' in code and '"--prefix", "gold_v2_loc_mmad_eval"' in code
    assert 'quality_status") != "FROZEN"' in code and "CHECKPOINT_EVERY = 20" in code

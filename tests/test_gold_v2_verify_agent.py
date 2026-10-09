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
SPEC = importlib.util.spec_from_file_location("gold_v2_verify_agent_t", ROOT / "scripts" / "run_gold_v2_verify_agent.py")
AG = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AG)

DEFECT = {"A": "Normal / no defect.", "B": "Rust / corrosion.",
          "C": "Missing cap / missing insulator component.", "D": "Bird nest / foreign nesting material."}


def _rows(n=6, with_q3=False):
    rows = []
    for i in range(n):
        anomalous = i % 2 == 1
        common = {"core_id": f"c{i}", "asset_category": "vari-grip", "image": f"sup/{i}.png",
                  "anomaly_gt": "anomalous" if anomalous else "normal", "defect_gt": "rust" if anomalous else "normal"}
        rows.append({**common, "id": f"c{i}-q1", "question_type": "anomaly_detection", "question": "Q1?",
                     "options": {"A": "Normal.", "B": "Anomalous."}, "answer": "B" if anomalous else "A"})
        rows.append({**common, "id": f"c{i}-q2", "question_type": "defect_classification",
                     "question": "Which condition best describes the target power-line asset?",
                     "options": dict(DEFECT), "answer": "B" if anomalous else "A"})
        if with_q3:
            rows.append({**common, "id": f"c{i}-q3", "question_type": "defect_localization", "question": "Q3?",
                         "options": {"A": "Center region.", "B": AG.cons.NONE_TEXT, "C": "Upper-left region.",
                                     "D": "Lower-right region."}, "answer": "A" if anomalous else "B"})
    return rows


def _png(v):
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), (v, v, v)).save(buf, format="PNG")
    return buf.getvalue()


def _setup(tmp, rows):
    sup, unsup = tmp / "sup.zip", tmp / "unsup.zip"
    with zipfile.ZipFile(sup, "w") as z:
        for name in sorted({r["image"] for r in rows}):
            z.writestr(name, _png(30))
    with zipfile.ZipFile(unsup, "w") as z:
        z.writestr("unsup/any.png", _png(200))
        z.writestr("unsup/other.png", _png(210))
    refs = tmp / "retrieved_refs.csv"
    with refs.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["image", "retrieved_any", "retrieved_diffdate"])
        w.writeheader()
        for img in sorted({r["image"] for r in rows}):
            w.writerow({"image": img, "retrieved_any": "unsup/any.png", "retrieved_diffdate": "unsup/other.png"})
    # stage 1: c0 normal img says normal+rust (contradiction); c1 rust says normal+normal (consistent but wrong,
    # image-only says anomalous -> disagreement); c2..c5 correct and consistent
    stage = {r["id"]: r["answer"] for r in rows}
    stage["c0-q2"] = "B"
    stage["c1-q1"], stage["c1-q2"] = "A", "A"
    if any(r["question_type"] == "defect_localization" for r in rows):
        stage["c1-q3"] = "B"
    image_only = dict(stage)
    image_only["c1-q1"] = "B"
    for m in ("qwen3_5_4b",):
        d = tmp / "stage1" / m
        d.mkdir(parents=True, exist_ok=True)
        (d / "normal_ref_retrieved_diffdate.jsonl").write_text(
            "\n".join(json.dumps({"id": i, "predicted_choice": c}) for i, c in stage.items()) + "\n", encoding="utf-8")
        (d / "image_only.jsonl").write_text(
            "\n".join(json.dumps({"id": i, "predicted_choice": c}) for i, c in image_only.items()) + "\n", encoding="utf-8")
    return sup, unsup, refs


def test_prompt_variants():
    q2 = _rows(1)[1]
    reask, zoom, define = (AG.build_prompt(q2, v) for v in ("reask", "zoom", "define"))
    assert "second look" in reask and "enlarged central crops" not in reask and "Visual definitions" not in reask
    assert "enlarged central crops" in zoom and "Visual definitions" not in zoom
    assert "Visual definitions" in define and "Rust / corrosion. orange" in define
    assert set(AG.DEFINITIONS) == set(DEFECT.values())


def test_agent_routes_verifies_and_rewrites(tmp_path):
    rows = _rows(6, with_q3=True)
    sup, unsup, refs = _setup(tmp_path, rows)
    truth = {r["core_id"]: r["answer"] for r in rows if r["question_type"] == "defect_classification"}
    calls = []

    def infer(adapter, images, prompt):
        name = Path(images[1]).name
        core = next(c for c in truth if name in (f"{c[1:]}.png", f"{c}.png"))
        calls.append((core, len(images), "Visual definitions" in prompt, [Path(p).name for p in images]))
        if "Visual definitions" in prompt:
            return truth[core]                 # definitions fix the answer
        return "A"                             # otherwise always "normal"

    out = tmp_path / "out"
    argv = ["--qa-dir", str(tmp_path), "--stage1-dir", str(tmp_path / "stage1"), "--refs-csv", str(refs),
            "--sup-zip", str(sup), "--unsup-zip", str(unsup), "--out-dir", str(out), "--models", "qwen3_5_4b",
            "--n-boot", "50"]
    assert AG.main(argv, load_qa=lambda _: rows, loader=lambda s: {}, infer=infer) == 0
    route = json.loads((out / "route_stats.json").read_text(encoding="utf-8"))[0]
    assert route["routed"] == 2 and route["by_reason"] == {"contradiction": 1, "disagreement": 1}
    assert all(n == 2 for _, n, _, _ in calls)
    assert any(names == ["c0.png", "c0.png"] for _, _, _, names in calls)     # zoom crops saved per core
    assert any(names == ["other.png", "0.png"] for _, _, _, names in calls)   # diffdate reference
    with (out / "leaderboard.csv").open(encoding="utf-8") as h:
        lb = {r["variant"]: r for r in csv.DictReader(h)}
    assert float(lb["stage1"]["joint_accuracy"]) < 1.0
    assert float(lb["define"]["joint_accuracy"]) == 1.0 and float(lb["define"]["consistency_error"]) == 0.0
    assert float(lb["define"]["q3_accuracy"]) < 1.0      # c1's Q3 'none' cannot be fixed by a defect verdict
    assert float(lb["reask"]["q1_anomaly_f1"]) <= float(lb["stage1"]["q1_anomaly_f1"])
    assert float(lb["zoom_define_all"]["avg_lvlm_calls_per_q"]) == round(1 + 6 / 18, 4)
    assert float(lb["define"]["avg_lvlm_calls_per_q"]) == round(1 + 2 / 18, 4)
    # rule baseline: c0 (Q1 normal) -> Q2 normal fixes it; c1 stays wrong
    assert float(lb["rule_q1_first"]["consistency_error"]) == 0.0
    with (out / "flips.csv").open(encoding="utf-8") as h:
        flips = {r["variant"]: r for r in csv.DictReader(h)}
    assert flips["define"]["q1_changed"] == "1" and flips["define"]["q1_changed_to_correct"] == "1"
    with (out / "significance.csv").open(encoding="utf-8") as h:
        assert {r["condition"] for r in csv.DictReader(h)} == set(AG.VARIANTS)
    n = len(calls)
    assert AG.main(argv, load_qa=lambda _: rows, loader=lambda s: {}, infer=infer) == 0
    assert len(calls) == n                                                   # rerun reuses everything


def test_agent_notebook():
    nb = json.loads((ROOT / "notebooks" / "lvlm_gold_v2_verify_agent_dev_colab.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for i, src in enumerate(cells):
        compile(src, f"agent_nb:{i}", "exec")
    code = "\n".join(cells)
    assert "run_gold_v2_verify_agent.py" in code and '"--stage1-condition"' in code
    assert '"--prefix", "gold_v2_verify_agent_dev"' in code and "torch.cuda.is_available()" in code

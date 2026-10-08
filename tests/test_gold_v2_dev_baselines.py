from __future__ import annotations

import csv
import gc
import importlib.util
import io
import json
import sys
import weakref
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("gold_v2_dev_baselines_t", ROOT / "scripts" / "run_gold_v2_dev_baselines.py")
RUN = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUN)

DEFECT = {"A": "Normal / no defect.", "B": "Rust / corrosion.",
          "C": "Missing cap / missing insulator component.", "D": "Bird nest / foreign nesting material."}
MODELS = ["internvl3_5_8b", "qwen3_5_4b"]


def _rows(n=6):
    rows = []
    for i in range(n):
        anomalous = i % 2 == 1
        common = {"core_id": f"dev-{i}", "asset_category": "vari-grip" if i < 4 else "yoke-suspension",
                  "image": f"defect_supervised/x/train/{i}.png", "normal_reference_image": "unsup/fixed.png",
                  "retrieved_reference_image": f"unsup/near{i % 2}.png", "anomaly_gt": "anomalous" if anomalous else "normal",
                  "defect_gt": "rust" if anomalous else "normal", "protocol": "gold_v2_dev"}
        rows.append({**common, "id": f"dev-{i}-q1", "question_type": "anomaly_detection",
                     "question": "Is the target power-line asset normal or anomalous?",
                     "options": {"A": "Normal.", "B": "Anomalous."}, "answer": "B" if anomalous else "A"})
        rows.append({**common, "id": f"dev-{i}-q2", "question_type": "defect_classification",
                     "question": "Which condition best describes the target power-line asset?",
                     "options": dict(DEFECT), "answer": "B" if anomalous else "A"})
    return rows


def _png(v):
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), (v, v, v)).save(buf, format="PNG")
    return buf.getvalue()


def test_prompts_and_images_match_gold_builders():
    import run_gold_baselines as base
    import run_gold_v2_reference_ablation as ref
    row = _rows(2)[0]
    assert RUN.build_prompt(row, "image_only") == base.prompt_for(row)
    assert RUN.build_prompt(row, "normal_ref_retrieved") == ref.prompt_for_condition(row, "normal_reference")
    t, f, r = Path("t"), Path("f"), Path("r")
    assert RUN.images_for("image_only", t, f, r) == [t]
    assert RUN.images_for("normal_ref_fixed", t, f, r) == [f, t]
    assert RUN.images_for("normal_ref_retrieved", t, f, r) == [r, t]


def test_dev_baselines_end_to_end(tmp_path):
    rows = _rows(6)
    qa = tmp_path / "dev_master_qa_12.jsonl"
    qa.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    sup, unsup = tmp_path / "sup.zip", tmp_path / "unsup.zip"
    with zipfile.ZipFile(sup, "w") as z:
        for name in sorted({r["image"] for r in rows}):
            z.writestr(name, _png(10))
    with zipfile.ZipFile(unsup, "w") as z:
        for name in ("unsup/fixed.png", "unsup/near0.png", "unsup/near1.png"):
            z.writestr(name, _png(200))
    answers = {r["question"] + r["image"]: r["answer"] for r in rows}
    refs_seen, alive, calls = [], [], []
    tracked: list[weakref.ref] = []

    class Model:
        pass

    def loader(spec):
        gc.collect()
        alive.append(sum(x() is not None for x in tracked))
        m = Model()
        tracked.append(weakref.ref(m))
        return {"m": m}

    def infer(adapter, images, prompt):
        calls.append(1)
        if len(images) == 2:
            refs_seen.append(Path(images[0]).name)
            row = next(r for r in rows if r["question"] in prompt and Path(r["image"]).name == Path(images[1]).name)
            return row["answer"]                       # reference conditions are always right
        return "A"                                     # image-only always says normal / no defect

    out = tmp_path / "out"
    argv = ["--dev-qa", str(qa), "--sup-zip", str(sup), "--unsup-zip", str(unsup), "--out-dir", str(out),
            "--models", *MODELS, "--n-boot", "100"]
    assert RUN.main(argv, loader=loader, infer=infer) == 0
    assert alive == [0, 0]                             # previous model released before the next load
    assert {"fixed.png", "near0.png", "near1.png"} <= set(refs_seen)
    with (out / "leaderboard.csv").open(encoding="utf-8") as h:
        lb = {(r["model_key"], r["condition"]): r for r in csv.DictReader(h)}
    assert float(lb[("qwen3_5_4b", "normal_ref_retrieved")]["joint_accuracy"]) == 1.0
    assert float(lb[("qwen3_5_4b", "image_only")]["q1_anomaly_f1"]) == 0.0
    with (out / "significance.csv").open(encoding="utf-8") as h:
        sig = list(csv.DictReader(h))
    assert {(r["baseline"], r["condition"]) for r in sig} == set(RUN.COMPARISONS)
    assert "q3_accuracy" not in {r["metric"] for r in sig}
    with (out / "metrics_by_asset.csv").open(encoding="utf-8") as h:
        assert {r["asset_category"] for r in csv.DictReader(h)} == {"vari-grip", "yoke-suspension"}
    n = len(calls)
    assert RUN.main(argv, loader=loader, infer=infer) == 0
    assert len(calls) == n                             # rerun reuses everything


def test_dev_baselines_notebook():
    nb = json.loads((ROOT / "notebooks" / "lvlm_gold_v2_dev_baselines_colab.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for i, src in enumerate(cells):
        compile(src, f"dev_baselines_nb:{i}", "exec")
    code = "\n".join(cells)
    assert "run_gold_v2_dev_baselines.py" in code and "CHECKPOINT_EVERY = 20" in code
    assert 'MODELS = ["internvl3_5_8b","qwen3_5_4b"]' in code
    assert '"--prefix", "gold_v2_dev_baselines"' in code and "torch.cuda.is_available()" in code

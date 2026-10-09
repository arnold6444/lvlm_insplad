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


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


MAIN = _load("gold_v2_main_conditions_t", "run_gold_v2_main_conditions.py")
CONS = _load("gold_v2_consistency_metrics_t2", "run_gold_v2_consistency_metrics.py")
SCORE = _load("gold_v2_score_baselines_t", "run_gold_v2_score_baselines.py")
DEFECT = {"A": "Normal / no defect.", "B": "Rust / corrosion.",
          "C": "Missing cap / missing insulator component.", "D": "Bird nest / foreign nesting material."}


def _gold(n=6):
    rows = []
    for i in range(n):
        anomalous = i % 2 == 1
        common = {"core_id": f"core-{i}", "asset_category": "vari-grip", "image": f"sup/Fotos 0{i}-12-2020_DJI_000{i}.png",
                  "normal_reference_image": "unsup/fixed.png", "anomaly_gt": "anomalous" if anomalous else "normal",
                  "defect_gt": "rust" if anomalous else "normal"}
        rows.append({**common, "id": f"core-{i}-q1", "question_type": "anomaly_detection",
                     "question": "Is the target power-line asset normal or anomalous?",
                     "options": {"A": "Normal.", "B": "Anomalous."}, "answer": "B" if anomalous else "A"})
        rows.append({**common, "id": f"core-{i}-q2", "question_type": "defect_classification",
                     "question": "Which condition best describes the target power-line asset?",
                     "options": dict(DEFECT), "answer": "B" if anomalous else "A"})
        rows.append({**common, "id": f"core-{i}-q3", "question_type": "defect_localization",
                     "question": "Where is the defect region located in the target image?",
                     "options": {"A": "Center region.", "B": "No localized defect region is present.",
                                 "C": "Upper-left region.", "D": "Lower-right region."},
                     "answer": "A" if anomalous else "B"})
    return rows


def _png(v):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (v, v, v)).save(buf, format="PNG")
    return buf.getvalue()


def _inputs(tmp: Path, rows):
    sup, unsup = tmp / "sup.zip", tmp / "unsup.zip"
    with zipfile.ZipFile(sup, "w") as z:
        for name in sorted({r["image"] for r in rows}):
            z.writestr(name, _png(10))
    with zipfile.ZipFile(unsup, "w") as z:
        for name in ("unsup/fixed.png", "unsup/near.png", "unsup/other_flight.png"):
            z.writestr(name, _png(200))
    refs = tmp / "retrieved_refs.csv"
    with refs.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["core_id", "image", "anomaly_gt", "retrieved_any", "cosine_any",
                                          "retrieved_diffdate", "cosine_diffdate"])
        w.writeheader()
        for r in rows:
            if r["question_type"] == "anomaly_detection":
                normal = r["anomaly_gt"] == "normal"
                w.writerow({"core_id": r["core_id"], "image": r["image"], "anomaly_gt": r["anomaly_gt"],
                            "retrieved_any": "unsup/near.png", "cosine_any": 0.95 if normal else 0.70,
                            "retrieved_diffdate": "unsup/other_flight.png", "cosine_diffdate": 0.85 if normal else 0.72})
    manifest = tmp / "roi_manifest_all.csv"
    with manifest.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["query_id", "expert", "asset_category", "anomaly_score_top1pct"])
        w.writeheader()
        for i in range(6):
            w.writerow({"query_id": f"core-{i}", "expert": "patchcore", "asset_category": "vari-grip",
                        "anomaly_score_top1pct": str(i % 2 + i / 100)})
    return sup, unsup, refs, manifest


def test_slugs_match_consistency_mapping():
    assert MAIN.SLUG_TO_CONDITION == CONS.RERUN_SLUG_TO_CONDITION


def test_main_conditions_images_prompts_and_release(tmp_path):
    rows = _gold()
    sup, unsup, refs, manifest = _inputs(tmp_path, rows)
    seen, alive, tracked = [], [], []

    class Model:
        pass

    def loader(spec):
        gc.collect()
        alive.append(sum(t() is not None for t in tracked))
        m = Model()
        tracked.append(weakref.ref(m))
        return {"m": m}

    def infer(adapter, images, prompt):
        seen.append(([Path(p).name for p in images], prompt))
        return "A"

    out = tmp_path / "out"
    argv = ["--qa-dir", str(tmp_path), "--sup-zip", str(sup), "--unsup-zip", str(unsup), "--out-dir", str(out),
            "--models", "minicpm_v_4_5", "qwen3_5_4b", "--conditions", *MAIN.CONDITIONS,
            "--roi-manifest", str(manifest), "--refs-csv", str(refs)]
    assert MAIN.main(argv, load_qa=lambda _: rows, loader=loader, infer=infer) == 0
    assert alive == [0, 0]
    refs_used = {tuple(names[:-1]) for names, _ in seen}
    assert {(), ("fixed.png",), ("near.png",), ("other_flight.png",)} == refs_used
    assert any("percentile" in prompt for _, prompt in seen)
    with (out / "leaderboard.csv").open(encoding="utf-8") as h:
        lb = {(r["model_key"], r["condition"]) for r in csv.DictReader(h)}
    assert lb == {(m, c) for m in ("minicpm_v_4_5", "qwen3_5_4b") for c in MAIN.CONDITIONS}
    n = len(seen)
    assert MAIN.main(argv, load_qa=lambda _: rows, loader=loader, infer=infer) == 0
    assert len(seen) == n                                     # rerun reuses everything

    # consistency picks up the rerun and keeps the replaced original as @original
    img = tmp_path / "baseline.csv"
    with img.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["model_key", "id", "predicted_choice"])
        w.writeheader()
        for r in rows:
            w.writerow({"model_key": "minicpm_v_4_5", "id": r["id"], "predicted_choice": r["answer"]})
    cons_out = tmp_path / "cons"
    assert CONS.main(["--qa-dir", str(tmp_path), "--image-only-csv", str(img), "--rerun-dir", str(out),
                      "--out-dir", str(cons_out), "--models", "minicpm_v_4_5", "--n-boot", "50"],
                     load_qa=lambda _: rows) == 0
    with (cons_out / "consistency_leaderboard.csv").open(encoding="utf-8") as h:
        conds = {r["condition"]: r for r in csv.DictReader(h)}
    assert "image_only@original" in conds and float(conds["image_only@original"]["joint_accuracy"]) == 1.0
    assert float(conds["image_only"]["joint_accuracy"]) < 1.0           # the rerun ("A" everywhere) is used
    assert "normal_ref_retrieved_diffdate" in conds
    with (cons_out / "significance_main.csv").open(encoding="utf-8") as h:
        pairs = {(r["baseline"], r["condition"]) for r in csv.DictReader(h)}
    assert ("normal_ref_retrieved", "normal_ref_retrieved_diffdate") in pairs


def test_score_baselines_threshold_from_dev(tmp_path):
    rows = _gold()
    _, _, refs, manifest = _inputs(tmp_path, rows)
    dev = tmp_path / "dev_refs.csv"
    with dev.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["core_id", "anomaly_gt", "cosine_any", "cosine_diffdate"])
        w.writeheader()
        for i, (lab, ca, cd) in enumerate([("normal", 0.96, 0.86), ("anomalous", 0.6, 0.6),
                                           ("normal", 0.9, 0.84), ("anomalous", 0.65, 0.7)]):
            w.writerow({"core_id": f"d{i}", "anomaly_gt": lab, "cosine_any": ca, "cosine_diffdate": cd})
    out = tmp_path / "score"
    assert SCORE.main(["--qa-dir", str(tmp_path), "--roi-manifest", str(manifest), "--gold-refs-csv", str(refs),
                       "--dev-refs-csv", str(dev), "--out-dir", str(out)], load_qa=lambda _: rows) == 0
    with (out / "score_baselines.csv").open(encoding="utf-8") as h:
        res = {r["method"]: r for r in csv.DictReader(h)}
    assert float(res["cosine_any"]["q1_anomaly_f1"]) == 1.0 and float(res["cosine_any"]["auroc"]) == 1.0
    assert float(res["cosine_any"]["threshold"]) <= 0.9
    assert float(res["patchcore_raw_score"]["auroc"]) == 1.0          # odd cores score higher in the fixture
    assert "patchcore_pct50" in res


def test_rerun_notebooks_wire_every_step():
    def code_of(name):
        nb = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
        cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        for i, src in enumerate(cells):
            compile(src, f"{name}:{i}", "exec")
        return "\n".join(cells)

    gold = code_of("lvlm_gold_v2_main_rerun_colab.ipynb")
    for needle in ("build_retrieved_refs.py", "run_gold_v2_score_baselines.py", "run_gold_v2_main_conditions.py",
                   '"--models", "minicpm_v_4_5"', "--rerun-dir", '"--prefix", "gold_v2_main_rerun"',
                   "CHECKPOINT_EVERY = 20", "torch.cuda.is_available()"):
        assert needle in gold, needle
    assert gold.index("build_retrieved_refs.py") < gold.index("run_gold_v2_main_conditions.py") < gold.index("--rerun-dir")
    dev = code_of("lvlm_gold_v2_dev_baselines_colab.ipynb")
    assert '"normal_ref_retrieved_diffdate"' in dev and '"--refs-csv"' in dev

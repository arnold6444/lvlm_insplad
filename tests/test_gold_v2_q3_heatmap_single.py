from __future__ import annotations

import csv
import gc
import importlib.util
import io
import json
import weakref
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gold_v2_q3_heatmap_single",
                                              ROOT / "scripts" / "run_gold_v2_q3_heatmap_single.py")
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)

MODELS = ["qwen3_5_4b", "internvl3_5_8b"]
EXPERTS = ["patchcore", "anomalyclip"]


class _FakeModel:
    """Stands in for GPU weights: the previous model must be freed before the next one loads."""


def _tracking_loader(alive_at_load: list[int]):
    refs: list[weakref.ref] = []

    def loader(spec):
        gc.collect()
        alive_at_load.append(sum(ref() is not None for ref in refs))
        model = _FakeModel()
        refs.append(weakref.ref(model))
        return {"model": model}

    return loader


def _q3(i: int, answer_text: str) -> dict:
    texts = [answer_text, "Center region.", "Upper-left region.", RUNNER.NONE_TEXT]
    texts = list(dict.fromkeys(texts))[:4]
    if len(texts) < 4:
        texts.append("Lower-right region.")
    options = dict(zip("ABCD", texts))
    return {"id": f"core-{i}-q3", "core_id": f"core-{i}", "image": f"sup/{i}.png",
            "question_type": RUNNER.Q3, "question": "Where is the defect region located in the target image?",
            "options": options, "answer": next(k for k, v in options.items() if v == answer_text)}


def test_region_of_point_matches_builder_rule():
    assert RUNNER.region_of_point(0.5, 0.5) == "center"
    assert RUNNER.region_of_point(0.1, 0.1) == "upper-left"
    assert RUNNER.region_of_point(0.9, 0.5) == "right"
    assert RUNNER.region_of_point(0.5, 0.9) == "lower"


def test_prompt_describes_single_marked_image():
    row = _q3(0, "Middle-right region.")
    heat = RUNNER.build_prompt(row, "heatmap_single", "patchcore")
    box = RUNNER.build_prompt(row, "roi_box_single", "patchcore")
    assert "Image 2" not in heat and "Image 2" not in box
    assert "heatmap overlaid" in heat and "yellow" not in heat
    assert "red rectangle" in box
    assert box.rstrip().endswith("Return exactly one option letter and nothing else.")


def test_mcnemar_exact():
    assert RUNNER.mcnemar_exact([False] * 5, [True] * 5) == (5, 0, 0.0625)
    assert RUNNER.mcnemar_exact([True, False], [True, False]) == (0, 0, 1.0)


def test_expert_only_choice_uses_roi_centre():
    row = _q3(0, "Middle-right region.")
    roi = {"roi_left": "70", "roi_top": "40", "roi_right": "90", "roi_bottom": "60"}
    assert row["options"][RUNNER.expert_only_choice(row, roi, 100, 100)] == "Middle-right region."


def _png(size=(60, 60), color=(120, 120, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_main_end_to_end_with_fake_model(tmp_path):
    rows = [_q3(0, "Middle-right region."), _q3(1, "Upper-left region."), _q3(2, RUNNER.NONE_TEXT)]
    other = {"id": "core-0-q1", "core_id": "core-0", "question_type": "anomaly_detection"}
    sup = tmp_path / "sup.zip"
    with zipfile.ZipFile(sup, "w") as z:
        for r in rows:
            z.writestr(r["image"], _png())
    art = tmp_path / "art"
    manifest_rows = []
    for r in rows:
        for e in EXPERTS:
            rel = f"pure_heatmap/{e}/{r['core_id']}.png"
            (art / rel).parent.mkdir(parents=True, exist_ok=True)
            (art / rel).write_bytes(_png(color=(200, 80, 0)))
            manifest_rows.append({"query_id": r["core_id"], "expert": e, "overlay_path": rel,
                                  "roi_left": 40, "roi_top": 20, "roi_right": 55, "roi_bottom": 40})
    manifest = tmp_path / "roi_manifest_all.csv"
    with manifest.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(manifest_rows[0]))
        w.writeheader()
        w.writerows(manifest_rows)
    img_csv = tmp_path / "baseline.csv"
    with img_csv.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["model_key", "id", "predicted_choice"])
        w.writeheader()
        for m in MODELS:
            for r in rows:   # image-only gets everything wrong
                w.writerow({"model_key": m, "id": r["id"],
                            "predicted_choice": next(k for k in r["options"] if k != r["answer"])})
    vx = tmp_path / "vx"
    (vx / MODELS[0]).mkdir(parents=True)
    (vx / MODELS[0] / "patchcore__heatmap.jsonl").write_text(
        "\n".join(json.dumps({"id": r["id"], "predicted_choice": r["answer"]}) for r in rows) + "\n", encoding="utf-8")

    answers = {r["id"]: r["answer"] for r in rows}
    seen, loads = [], []

    def infer(adapter, images, prompt):
        assert len(images) == 1 and Path(images[0]).is_file()
        seen.append(Path(images[0]))
        rid = next(r["id"] for r in rows if r["question"] in prompt and r["core_id"] in str(images[0]))
        return answers[rid] if "roi_box" in str(images[0]) else "A"

    out = tmp_path / "out"
    argv = ["--qa-dir", str(tmp_path), "--sup-zip", str(sup), "--roi-manifest", str(manifest),
            "--artifact-root", str(art), "--image-only-csv", str(img_csv), "--vx-pred-dir", str(vx),
            "--out-dir", str(out), "--models", *MODELS, "--experts", *EXPERTS]
    load = lambda _: rows + [other]  # noqa: E731
    alive_at_load: list[int] = []
    track = _tracking_loader(alive_at_load)
    assert RUNNER.main(argv, load_qa=load, loader=lambda s: loads.append(s) or track(s), infer=infer) == 0
    assert alive_at_load == [0] * len(MODELS)   # a model still in memory pushed MiniCPM onto meta tensors
    with (out / "leaderboard.csv").open(encoding="utf-8") as h:
        lb = {(r["model_key"], r["expert"], r["condition"]): r for r in csv.DictReader(h)}
    box = lb[(MODELS[0], "patchcore", "roi_box_single")]
    assert float(box["q3_acc"]) == 1.0 and int(box["gain_vs_image_only"]) == 3
    assert float(lb[(MODELS[0], "", "image_only")]["q3_acc"]) == 0.0
    assert "mcnemar_p_vs_two_image" in lb[(MODELS[0], "patchcore", "heatmap_single")]
    assert any("roi_box" in str(p) for p in seen) and len(loads) == len(MODELS)
    with (out / "expert_only.csv").open(encoding="utf-8") as h:
        assert {r["expert"] for r in csv.DictReader(h)} == set(EXPERTS)

    n_calls = len(seen)
    assert RUNNER.main(argv, load_qa=load, loader=lambda s: loads.append(s) or {}, infer=infer) == 0
    assert len(seen) == n_calls and len(loads) == len(MODELS)   # rerun reuses everything


def test_q3_notebook_runs_single_image_conditions_and_publishes():
    nb = json.loads((ROOT / "notebooks" / "lvlm_gold_v2_q3_heatmap_single_colab.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
             for c in nb["cells"] if c["cell_type"] == "code"]
    for i, src in enumerate(cells):
        compile(src, f"q3_nb:{i}", "exec")
    code = "\n".join(cells)
    assert "run_gold_v2_q3_heatmap_single.py" in code and "CHECKPOINT_EVERY = 20" in code
    assert "torch.cuda.is_available()" in code
    assert "\"--prefix\", \"gold_v2_q3_heatmap_single\"" in code
    assert "--vx-pred-dir" in code and "MERGED_MANIFEST" in code

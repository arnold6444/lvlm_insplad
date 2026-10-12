from __future__ import annotations

import csv
import importlib.util
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import pytest
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


AG = _load("gold_v2_verify_agent_t", "run_gold_v2_verify_agent.py")
Q1, Q2, Q3 = AG.Q1, AG.Q2, AG.Q3
DEFECT = {"A": "Normal / no defect.", "B": "Rust / corrosion.",
          "C": "Missing cap / missing insulator component.", "D": "Bird nest / foreign nesting material."}
LOC = {"A": "Center region.", "B": AG.cons.NONE_TEXT, "C": "Upper-left region.", "D": "Lower-right region."}
QTEXT = {Q1: "Is the target power-line asset normal or anomalous?",
         Q2: "Which condition best describes the target power-line asset?",
         Q3: "Where is the defect region located in the target image?"}
MODEL = "qwen3_5_4b"


def _rows(n=6, dev=True):
    rows = []
    for i in range(n):
        core, anomalous = f"c{i}", i % 2 == 1
        common = {"core_id": core, "asset_category": "vari-grip" if i < 4 else "yoke-suspension",
                  "image": f"sup/{core}.png", "anomaly_gt": "anomalous" if anomalous else "normal",
                  "defect_gt": "rust" if anomalous else "normal", **({"protocol": "gold_v2_dev"} if dev else {})}
        for qt, options, answer in ((Q1, {"A": "Normal.", "B": "Anomalous."}, "B" if anomalous else "A"),
                                    (Q2, dict(DEFECT), "B" if anomalous else "A"),
                                    (Q3, dict(LOC), "A" if anomalous else "B")):
            rows.append({**common, "id": f"{core}-{AG.SHORT[qt]}", "question_type": qt,
                         "question": f"[{core}] {QTEXT[qt]}", "options": options, "answer": answer})
    return rows


def _stage(rows):
    """Stage 1: c0 normal says rust (Q1-Q2 contradiction); c1 rust says normal/normal/none (wrong, consistent);
    c2 normal names the centre in Q3 (Q1-Q3 contradiction). Image-only differs only on c1-q1 and c3-q2."""
    stage = {r["id"]: r["answer"] for r in rows}
    stage.update({"c0-q2": "B", "c1-q1": "A", "c1-q2": "A", "c1-q3": "B", "c2-q3": "A"})
    image_only = dict(stage)
    image_only.update({"c1-q1": "B", "c3-q2": "A"})
    return stage, image_only


def _png(v):
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), (v, v, v)).save(buf, format="PNG")
    return buf.getvalue()


def _write_preds(path, choices, latency=0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps({"id": i, "predicted_choice": c, "latency_s": latency})
                              for i, c in choices.items()) + "\n", encoding="utf-8")


def _inputs(tmp, rows, stage, image_only):
    tmp.mkdir(parents=True, exist_ok=True)
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
    _write_preds(tmp / "stage1" / MODEL / "normal_ref_retrieved_diffdate.jsonl", stage)
    _write_preds(tmp / "stage1" / MODEL / "image_only.jsonl", image_only)
    qa = tmp / "dev_master_qa.jsonl"
    qa.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return ["--stage1-dir", str(tmp / "stage1"), "--refs-csv", str(refs), "--sup-zip", str(sup),
            "--unsup-zip", str(unsup), "--models", MODEL, "--n-boot", "50"], qa


def _who(prompt):
    core = re.search(r"\[(c\d+)\]", prompt).group(1)
    qt = next(q for q, text in QTEXT.items() if text in prompt)
    return core, qt


def _fake_infer(rows, calls):
    truth = {(r["core_id"], r["question_type"]): r["answer"] for r in rows}

    def infer(adapter, images, prompt):
        core, qt = _who(prompt)
        calls.append({"core": core, "qt": qt, "images": [Path(p).name for p in images], "prompt": prompt})
        if "Visual definitions" in prompt:
            return truth[(core, qt)]                 # definitions fix the answer
        if "enlarged central crops" in prompt:
            return "B"                               # zoom always says option B
        return "A"                                   # re-asking always says option A
    return infer


def _run_dev(tmp, rows, stage, image_only, extra=()):
    common, qa = _inputs(tmp, rows, stage, image_only)
    calls = []
    out = tmp / "out"
    assert AG.main(["--qa-file", str(qa), "--out-dir", str(out), *common, *extra],
                   loader=lambda s: {}, infer=_fake_infer(rows, calls)) == 0
    return calls, out


def _csv(path):
    with path.open(encoding="utf-8") as h:
        return list(csv.DictReader(h))


def _answers(out):
    return {(r["variant"], r["id"]): (r["rechecked"], r["recheck_choice"], r["final_choice"])
            for r in _csv(out / "answers.csv")}


def _rotate(row, letter):
    keys = sorted(row["options"])
    return keys[(keys.index(letter) + 1) % len(keys)] if letter in keys else keys[0]


@pytest.mark.parametrize("own", [Q1, Q2, Q3])
def test_changing_other_questions_answers_does_not_touch_this_question(tmp_path, own):
    """Q1/Q2/Q3 are verified independently: rewriting every other question's stage-1 and image-only answers leaves
    this question's routing, prompts, images, re-check answers and final answers unchanged."""
    rows = _rows()
    stage, image_only = _stage(rows)
    calls_a, out_a = _run_dev(tmp_path / "a", rows, stage, image_only)
    stage_b, image_only_b = dict(stage), dict(image_only)
    for r in rows:
        if r["question_type"] != own:
            stage_b[r["id"]] = _rotate(r, stage[r["id"]])
            image_only_b[r["id"]] = _rotate(r, _rotate(r, image_only[r["id"]]))
    calls_b, out_b = _run_dev(tmp_path / "b", rows, stage_b, image_only_b)

    def mine(calls):
        return sorted((c["core"], c["images"], c["prompt"]) for c in calls if c["qt"] == own)

    assert mine(calls_a) and mine(calls_a) == mine(calls_b)
    own_ids = {r["id"] for r in rows if r["question_type"] == own}
    ans_a, ans_b = _answers(out_a), _answers(out_b)
    assert {k: v for k, v in ans_a.items() if k[1] in own_ids} == {k: v for k, v in ans_b.items() if k[1] in own_ids}
    assert {k: v for k, v in ans_a.items() if k[1] not in own_ids} != {k: v for k, v in ans_b.items() if k[1] not in own_ids}
    routed_a = {r["id"] for r in _csv(out_a / "answers.csv") if r["rechecked"] == "True" and r["variant"].endswith("disagreement")}
    routed_b = {r["id"] for r in _csv(out_b / "answers.csv") if r["rechecked"] == "True" and r["variant"].endswith("disagreement")}
    assert routed_a & own_ids == routed_b & own_ids


def test_prompts_hold_only_their_own_question(tmp_path):
    rows = _rows()
    stage, image_only = _stage(rows)
    calls, _ = _run_dev(tmp_path, rows, stage, image_only)
    others = {Q1: ["Rust / corrosion.", "Center region.", QTEXT[Q2], QTEXT[Q3]],
              Q2: ["Anomalous.", "Center region.", QTEXT[Q1], QTEXT[Q3]],
              Q3: ["Anomalous.", "Rust / corrosion.", QTEXT[Q1], QTEXT[Q2]]}
    for c in calls:
        assert QTEXT[c["qt"]] in c["prompt"]
        assert not [t for t in others[c["qt"]] if t in c["prompt"]], c["prompt"]
    row = next(r for r in rows if r["id"] == "c0-q2")
    assert 'the answer to this question was "B. Rust / corrosion."' in AG.build_prompt(row, "B", "reask")


def test_contradictions_are_reported_not_acted_on(tmp_path):
    rows = _rows()
    stage, image_only = _stage(rows)
    calls, out = _run_dev(tmp_path, rows, stage, image_only,
                          ["--tools", "define", "--routes", "disagreement"])
    # only the two questions whose own answers disagree are re-checked; c0's Q1-Q2 contradiction is not a trigger
    assert sorted((c["core"], c["qt"]) for c in calls) == [("c1", Q1), ("c3", Q2)]
    answers = _csv(out / "answers.csv")
    for r in answers:
        if r["rechecked"] == "False":
            assert r["final_choice"] == r["stage1_choice"]
    lb = {r["variant"]: r for r in _csv(out / "leaderboard.csv")}
    # c1-q1 is fixed to Anomalous while c1-q2 keeps "normal": a new contradiction is kept and counted
    assert lb["stage1"]["contradictions"] == "1" and lb["define@disagreement"]["contradictions"] == "2"
    assert lb["define@disagreement"]["q1q2_contradictions_new"] == "1"
    assert lb["define@disagreement"]["q1q2_contradictions_resolved"] == "0"
    assert float(lb["define@disagreement"]["q1q3_contradiction"]) > 0
    assert float(lb["define@disagreement"]["avg_lvlm_calls_per_q"]) == round(1 + 2 / 18, 4)


def test_full_dev_run_keeps_both_answers_costs_and_reuse(tmp_path):
    rows = _rows()
    stage, image_only = _stage(rows)
    calls, out = _run_dev(tmp_path, rows, stage, image_only)
    assert len(calls) == len(AG.TOOLS) * len(rows)               # route "all" re-checks every question once per tool
    answers = _csv(out / "answers.csv")
    assert len(answers) == len(AG.TOOLS) * len(AG.ROUTES) * len(rows)
    for r in answers:
        assert r["stage1_choice"] == stage[r["id"]]
        if r["changed"] == "True":
            assert r["rechecked"] == "True" and r["final_choice"] == r["recheck_choice"]
    # zoom crops for Q1/Q2 only; Q3 keeps the full reference and target
    zoom = [c for c in calls if "enlarged central crops" in c["prompt"]]
    assert zoom and all(c["qt"] != Q3 and c["images"][0] != "other.png" for c in zoom)
    q3_zoom_tool = [c for c in calls if c["qt"] == Q3 and "Visual definitions" not in c["prompt"]]
    assert all(c["images"] == ["other.png", f"{c['core']}.png"] for c in q3_zoom_tool)
    lb = {r["variant"]: r for r in _csv(out / "leaderboard.csv")}
    assert set(lb) == {"stage1"} | {f"{t}@{r}" for t in AG.TOOLS for r in AG.ROUTES}
    assert float(lb["define@all"]["q1_accuracy"]) == float(lb["define@all"]["q2_accuracy"]) == 1.0
    assert float(lb["define@all"]["q3_accuracy"]) == 1.0
    assert float(lb["reask@all"]["avg_lvlm_calls_per_q"]) == 2.0 and lb["reask@all"]["rechecks"] == "18"
    assert float(lb["stage1"]["avg_lvlm_calls_per_q"]) == 1.0 and float(lb["stage1"]["stage1_latency_s"]) == 9.0
    pq = {(r["variant"], r["question_type"]): r for r in _csv(out / "per_question.csv")}
    assert pq[("define@all", Q1)]["changed"] == "1" and pq[("define@all", Q1)]["changed_to_correct"] == "1"
    assert pq[("define@disagreement", Q2)]["rechecked"] == "1"
    sel = _csv(out / "selection.csv")
    assert sel[0]["rank"] == "1" and sel[0]["variant"].startswith("define")
    assert {r["condition"] for r in _csv(out / "significance.csv")} == {f"{t}@{r}" for t in AG.TOOLS for r in AG.ROUTES}
    assert {r["asset_category"] for r in _csv(out / "metrics_by_asset.csv")} == {"vari-grip", "yoke-suspension"}
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["split"] == "dev" and meta["model_ids"][MODEL] == "Qwen/Qwen3.5-4B"

    # rerun reuses everything; changing one question's own stage-1 answer re-checks only that question
    calls.clear()
    common, qa = _inputs(tmp_path, rows, stage, image_only)
    infer = _fake_infer(rows, calls)
    assert AG.main(["--qa-file", str(qa), "--out-dir", str(out), *common], loader=lambda s: {}, infer=infer) == 0
    assert calls == []
    stage2 = dict(stage, **{"c4-q2": "C"})
    common, qa = _inputs(tmp_path, rows, stage2, image_only)
    assert AG.main(["--qa-file", str(qa), "--out-dir", str(out), *common], loader=lambda s: {}, infer=infer) == 0
    assert {(c["core"], c["qt"]) for c in calls} == {("c4", Q2)} and len(calls) == len(AG.TOOLS)


def test_gold_runs_only_the_frozen_dev_choice(tmp_path):
    dev_rows = _rows()
    stage, image_only = _stage(dev_rows)
    _, dev_out = _run_dev(tmp_path / "dev", dev_rows, stage, image_only, ["--tools", "reask", "define"])
    dev_meta = json.loads((dev_out / "run_metadata.json").read_text(encoding="utf-8"))
    selection = _csv(dev_out / "selection.csv")
    with pytest.raises(ValueError):
        AG.make_frozen_config(dev_meta, "zoom@all")              # not run on dev
    cfg = AG.make_frozen_config(dev_meta, "define@all", [{"variant": r["variant"], "rank": int(r["rank"])} for r in selection])
    assert cfg["selection_rank"] == 1 and cfg["tool"] == "define" and cfg["route"] == "all"
    cfg_path = tmp_path / "frozen_config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    gold_rows = _rows(dev=False)
    common, _ = _inputs(tmp_path / "gold", gold_rows, stage, image_only)
    out = tmp_path / "gold" / "out"
    calls = []
    run = lambda extra: AG.main(["--qa-dir", str(tmp_path), "--out-dir", str(out), *common, *extra],  # noqa: E731
                                load_qa=lambda _: gold_rows, loader=lambda s: {}, infer=_fake_infer(gold_rows, calls))
    with pytest.raises(RuntimeError, match="frozen-config"):
        run([])
    with pytest.raises(RuntimeError, match="only from --frozen-config"):
        run(["--frozen-config", str(cfg_path), "--tools", "zoom"])
    with pytest.raises(RuntimeError, match="stage1_condition"):
        run(["--frozen-config", str(cfg_path), "--stage1-condition", "image_only"])
    assert run(["--frozen-config", str(cfg_path)]) == 0
    assert {r["variant"] for r in _csv(out / "leaderboard.csv")} == {"stage1", "define@all"}
    assert not (out / "selection.csv").exists()
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["split"] == "gold" and meta["frozen_config"]["variant"] == "define@all" and meta["frozen_config_sha256"]
    other = tmp_path / "other_config.json"
    other.write_text(json.dumps(dict(cfg, variant="reask@all", tool="reask")), encoding="utf-8")
    with pytest.raises(RuntimeError, match="already run on gold"):
        run(["--frozen-config", str(other)])
    # a dev run refuses gold questions, and a gold run refuses dev questions
    common_dev, _ = _inputs(tmp_path / "mix", gold_rows, stage, image_only)
    qa = tmp_path / "mix" / "gold_as_dev.jsonl"
    qa.write_text("\n".join(json.dumps(r) for r in gold_rows) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="gold_v2_dev"):
        AG.main(["--qa-file", str(qa), "--out-dir", str(tmp_path / "mix" / "out"), *common_dev],
                loader=lambda s: {}, infer=_fake_infer(gold_rows, []))
    with pytest.raises(RuntimeError, match="gold questions"):
        AG.main(["--qa-dir", str(tmp_path), "--out-dir", str(tmp_path / "mix" / "out2"), *common_dev,
                 "--frozen-config", str(cfg_path)], load_qa=lambda _: dev_rows, loader=lambda s: {},
                infer=_fake_infer(dev_rows, []))


def test_definitions_cover_every_option_text():
    gold_builder = _load("build_explainable_qa_gold_t", "build_explainable_qa_gold.py")
    texts = {"Normal.", "Anomalous."} | set(DEFECT.values()) | set(gold_builder.REGION_TEXT.values())
    assert texts == set(AG.DEFINITIONS)
    q3 = next(r for r in _rows() if r["question_type"] == Q3)
    assert "enlarged central crops" not in AG.build_prompt(q3, "A", "zoom_define")
    assert "Visual definitions" in AG.build_prompt(q3, "A", "zoom_define")
    assert "not a valid option" in AG.build_prompt(q3, None, "reask")


def _notebook_code(name):
    nb = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for i, src in enumerate(cells):
        compile(src, f"{name}:{i}", "exec")
    return "\n".join(cells)


def test_dev_notebook():
    code = _notebook_code("lvlm_gold_v2_verify_agent_dev_colab.ipynb")
    assert "run_gold_v2_verify_agent.py" in code and '"--stage1-condition"' in code
    assert '"--tools"' in code and '"--routes"' in code and "--triggers" not in code and "rule_q1_first" not in code
    assert "make_frozen_config" in code and "frozen_config.json" in code
    assert '"--prefix", "gold_v2_verify_agent_dev"' in code and "torch.cuda.is_available()" in code


def test_gold_notebook():
    code = _notebook_code("lvlm_gold_v2_verify_agent_gold_colab.ipynb")
    assert "run_gold_v2_verify_agent.py" in code and '"--qa-dir"' in code and '"--frozen-config"' in code
    assert "results/latest/gold_v2_verify_agent_dev/frozen_config.json" in code
    assert '"--tools"' not in code and '"--routes"' not in code
    assert '"--prefix", "gold_v2_verify_agent_gold"' in code and "torch.cuda.is_available()" in code

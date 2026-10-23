from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(ROOT/"tests"))
import build_gold_v2_matched_call_notebook as builder
import run_gold_v2_matched_call as runner
import test_gold_v2_q23_candidate as cand
import test_gold_v2_verify_agent as fixtures

MODEL = "internvl3_5_8b"


def setup(tmp, external=True):
    rows, first, args, cand_out = cand.setup(tmp, external)
    out = tmp/"gold_v2_matched_call_test"
    args = [*args[:args.index("--out-dir")+1], str(out), *args[args.index("--out-dir")+2:]]
    return rows, first, args, cand_out, out


def fake(calls, invalid=None):
    """Candidate fixture answers, plus own-answer re-checks: Q2 says A, Q3 says centre on odd cores."""
    inner = cand.fake(calls)
    def infer(adapter, images, prompt):
        if "second look" not in prompt or "Visual definitions" in prompt:
            return inner(adapter, images, prompt)
        core = Path(images[-1]).stem
        texts = dict(re.findall(r"^([A-J])\. (.+)$", prompt, re.M))
        if runner.old.NONE in texts.values():
            step = "q3_reask"
            target = "Center region." if int(core[1:]) % 2 else runner.old.NONE
            answer = next(k for k, v in texts.items() if v == target)
        else:
            step, answer = "q2_reask", "A"
        calls.append((core, step, prompt, [Path(p).name for p in images]))
        return "unparseable" if invalid and invalid(core, step) else answer
    return infer


def run(rows, args, calls, **kwargs):
    return runner.main(args, load_qa=lambda _: rows, loader=lambda spec: {}, infer=fake(calls, **kwargs))


def by_policy(path):
    return {r["policy"]: r for r in fixtures._csv(path)}


def test_control_spends_previous_agent_calls_keeps_q1_and_resumes_without_calls(tmp_path):
    rows, first, args, _, out = setup(tmp_path)
    calls = []
    assert run(rows, args, calls) == 0
    steps = [step for _, step, _, _ in calls]
    assert steps.count("q2_reask") == 6 and steps.count("q3_reask") == 6 + 3  # odd cores add a position call
    budget = {r["policy"]: r for r in fixtures._csv(out/"call_budget.csv")}
    for key in ("q2_calls", "q3_calls"):
        assert budget["matched_reask"][key] == budget["previous_agent"][key]
    assert int(budget["baseline"]["q2_calls"]) == 6 and int(budget["baseline"]["q3_calls"]) == 6
    answers = fixtures._csv(out/"answers.csv")
    assert len(answers) == 4*18
    assert all(r["predicted_choice"] == first[r["id"]] for r in answers if r["question_type"] == runner.agent.Q1)
    assert set(by_policy(out/"leaderboard.csv")) == set(runner.POLICIES)
    sig = fixtures._csv(out/"significance.csv")
    assert any(r["baseline"] == "matched_reask" and r["condition"] == "previous_agent" for r in sig)
    # The second Q3 re-check shows the first re-check's own answer, never another question's answer.
    second = [prompt for core, step, prompt, _ in calls if step == "q3_reask" and core == "c1"][1]
    assert re.search(r'first look, the answer to this question was "[A-J]\. Center region\."', second)
    matched = {r["id"]: r["predicted_text"] for r in answers if r["policy"] == "matched_reask"}
    assert matched["c1-q3-gold-presence"] == "Center region." and matched["c0-q3-gold-presence"] == runner.old.NONE
    calls.clear()
    assert run(rows, args, calls) == 0 and calls == []
    meta = json.loads((out/"run_metadata.json").read_text())
    assert meta["complete"] and meta["called_this_run"] == 0 and meta["q1_calls"] == 0


def test_reused_candidate_run_keeps_published_policies_and_only_adds_control_calls(tmp_path):
    rows, _, args, cand_out, out = setup(tmp_path)
    cand_args = [*args[:args.index("--out-dir")+1], str(cand_out), *args[args.index("--out-dir")+2:]]
    assert cand.run(rows, cand_args, []) == 0
    steps = cand_out/"predictions"/MODEL/"steps.jsonl"
    before = steps.read_bytes()
    calls = []
    reuse = ["--reuse-steps", str(steps), "--reuse-manifest", str(cand_out/"image_manifest.csv")]
    assert run(rows, [*args, *reuse], calls) == 0
    assert {step for _, step, _, _ in calls} == {"q2_reask", "q3_reask"} and len(calls) == 15
    meta = json.loads((out/"run_metadata.json").read_text())
    assert meta["reused_candidate_steps"] == 39 and meta["imported_baseline_calls"] == 0
    old = {(r["policy"], r["id"]): r["predicted_choice"] for r in fixtures._csv(cand_out/"answers.csv")}
    new = {(r["policy"], r["id"]): r["predicted_choice"] for r in fixtures._csv(out/"answers.csv")}
    assert all(new[key] == value for key, value in old.items())
    old_lb, new_lb = by_policy(cand_out/"leaderboard.csv"), by_policy(out/"leaderboard.csv")
    for policy in ("baseline", "previous_agent", "candidate"):
        assert new_lb[policy]["q2_defect_macro_f1"] == old_lb[policy]["q2_defect_macro_f1"]
        assert new_lb[policy]["q3_presence_balanced_accuracy"] == old_lb[policy]["q3_presence_balanced_accuracy"]
    assert steps.read_bytes() == before


def test_reuse_rejects_different_image_bytes(tmp_path):
    rows, _, args, cand_out, out = setup(tmp_path)
    cand_args = [*args[:args.index("--out-dir")+1], str(cand_out), *args[args.index("--out-dir")+2:]]
    assert cand.run(rows, cand_args, []) == 0
    manifest = cand_out/"image_manifest.csv"
    records = fixtures._csv(manifest)
    records[0]["sha256"] = "0"*64
    runner.agent.cons.write_csv(manifest, records)
    reuse = ["--reuse-steps", str(cand_out/"predictions"/MODEL/"steps.jsonl"), "--reuse-manifest", str(manifest)]
    with pytest.raises(ValueError, match="image bytes differ"):
        run(rows, [*args, *reuse], [])
    with pytest.raises(ValueError, match="needs both"):
        run(rows, [*args, reuse[0], reuse[1]], [])


def test_invalid_control_answer_excludes_only_control_then_retries_once(tmp_path):
    rows, _, args, _, out = setup(tmp_path)
    calls = []
    assert run(rows, args, calls, invalid=lambda core, step: core == "c0" and step == "q2_reask") == 1
    assert set(by_policy(out/"leaderboard.csv")) == {"baseline", "previous_agent", "candidate"}
    status = json.loads((out/"run_status.json").read_text())
    assert not status["complete"] and not status["selection_performed"]
    assert [p["eligible_for_comparison"] for p in status["policies"]] == [True, True, True, False]
    calls.clear()
    assert run(rows, args, calls) == 0 and [(core, step) for core, step, _, _ in calls] == [("c0", "q2_reask")]


def test_wrong_folder_or_changed_lock_is_refused(tmp_path):
    rows, _, args, _, out = setup(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        run(rows, [*args, "--out-dir", str(tmp_path/"gold_v2_q23_candidate_v2")], [])
    config = json.loads(runner.CONFIG.read_text())
    config["matched_reask_q3"] = "something else"
    bad = tmp_path/"bad.json"
    bad.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="committed lock"):
        run(rows, [*args, "--config", str(bad)], [])
    assert not out.exists()


def test_notebook_inputs_prefer_largest_saved_candidate_run_and_its_references(tmp_path):
    rows = [r for r in fixtures._rows(246, dev=False) if r["question_type"] != runner.agent.Q3]
    qa_dir = tmp_path/"lvlm_insplad_qa/gold_v2_final"
    qa_dir.mkdir(parents=True)
    runner.candidate.save_records(qa_dir/"master_qa_test.jsonl", rows)
    runner.old.atomic_json(qa_dir/"build_summary.json", {"quality_status": "FROZEN", "final_master_images": 246})
    for name in ("supervised_fault_classification.zip", "unsupervised_anomaly_detection.zip"):
        (tmp_path/name).write_bytes(b"fixture zip file")
    refs = [{"image": r["image"], "retrieved_diffdate": r["asset_category"]+"/train/good/ref.png"} for r in rows[::2]]
    folders = ["lvlm_insplad_results/gold_v2_q23_candidate_v2", "lvlm_insplad_results/gold_v2_q23_candidate_v1"]
    for folder, n in zip(folders, (3, 5)):
        run_dir = tmp_path/folder
        runner.candidate.save_records(run_dir/"predictions"/MODEL/"steps.jsonl", [{"id": str(i)} for i in range(n)])
        (run_dir/"image_manifest.csv").write_text("role,member,sha256\n")
        runner.agent.cons.write_csv(run_dir/"refs_gold/retrieved_refs.csv", refs)
    notebook = json.loads(builder.NOTEBOOK.read_text(encoding="utf-8"))
    commands = []
    context = {"DRIVE_ROOT": tmp_path, "QA_DIR": qa_dir, "OLD_GOLD_DIR": tmp_path/"missing_main_rerun",
        "RUN_ROOT": tmp_path/"lvlm_insplad_results/gold_v2_matched_call_v1", "REPO_DIR": ROOT, "json": json,
        "sys": sys, "Path": Path, "CANDIDATE_RUN_FOLDERS": folders,
        "SUP_ZIP_NAME": "supervised_fault_classification.zip", "UNSUP_ZIP_NAME": "unsupervised_anomaly_detection.zip",
        "SUP_ZIP_SIZE": 16, "UNSUP_ZIP_SIZE": 16, "SUP_ZIP_PATH": "", "UNSUP_ZIP_PATH": "",
        "run": lambda cmd: commands.append(cmd)}
    exec("".join(notebook["cells"][3]["source"]), context)
    assert not commands  # no new reference retrieval when the saved run has its references
    assert context["REUSE_DIR"] == tmp_path/folders[1]
    assert context["REFS_CSV"] == tmp_path/folders[1]/"refs_gold/retrieved_refs.csv"
    assert len(runner.agent.read_jsonl(context["GOLD_QA"])) == 492


def test_notebook_is_built_from_current_code_and_gates_real_outputs(tmp_path):
    notebook = json.loads((ROOT/"notebooks/lvlm_gold_v2_matched_call_colab.ipynb").read_text(encoding="utf-8"))
    assert notebook == builder.build(), "rebuild the notebook: python scripts/build_gold_v2_matched_call_notebook.py"
    for i, c in enumerate(notebook["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"cell{i}", "exec")
    config = {}
    exec("".join(notebook["cells"][1]["source"]), config)
    assert "scripts/run_gold_v2_matched_call.py" in config["CODE_BLOBS"]
    assert "scripts/run_gold_v2_q23_candidate.py" in config["CODE_BLOBS"]
    assert config["RUN_FOLDER"].endswith("gold_v2_matched_call_v1")
    rows, _, args, _, out = setup(tmp_path)
    assert run(rows, args, []) == 0
    import pandas as pd
    cell = "".join(notebook["cells"][5]["source"])
    context = {"pd": pd, "json": json, "RUN_ROOT": out, "display": lambda obj: None}
    exec(cell, context)
    status = json.loads((out/"run_status.json").read_text())
    status["complete"] = False
    runner.old.atomic_json(out/"run_status.json", status)
    with pytest.raises(RuntimeError, match="실패/무효/누락"):
        exec(cell, context)

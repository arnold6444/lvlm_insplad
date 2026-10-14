from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import run_gold_v2_q23_candidate as runner
import test_gold_v2_verify_agent as fixtures

MODEL = "internvl3_5_8b"


def setup(tmp, external=True):
    rows = [r for r in fixtures._rows(dev=False) if r["question_type"] != runner.agent.Q3]
    first, _ = fixtures._stage(rows)
    common, qa = fixtures._inputs(tmp, rows, first, first)
    (tmp/"build_summary.json").write_text(json.dumps({"quality_status": "FROZEN", "final_master_images": 6}))
    refs = []
    for row in rows[::2]:
        member = row["asset_category"]+"/train/good/reference.png"
        refs.append({"image": row["image"], "retrieved_diffdate": member})
    runner.agent.cons.write_csv(tmp/"refs.csv", refs)
    with ZipFile(tmp/"unsup.zip", "w") as archive:
        for member in sorted({r["retrieved_diffdate"] for r in refs}):
            archive.writestr(member, fixtures._png(200))
    source = tmp/"baseline.jsonl"
    by_image = {r["image"]: r["retrieved_diffdate"] for r in refs}
    runner.save_records(source, [{"id": r["id"], "core_id": r["core_id"], "question_type": r["question_type"],
        "predicted_choice": first[r["id"]], "output": first[r["id"]], "model_key": MODEL,
        "condition": "normal_ref_retrieved_diffdate", "prompt_version": 1,
        "reference_image": "/old/normal/"+by_image[r["image"]], "latency_s": .5, "error": ""} for r in rows])
    out = tmp/"gold_v2_q23_candidate_test"
    args = ["--qa-dir", str(tmp), "--refs-csv", str(tmp/"refs.csv"), "--sup-zip", str(tmp/"sup.zip"),
        "--unsup-zip", str(tmp/"unsup.zip"), "--out-dir", str(out), "--n-boot", "10", "--checkpoint-every", "2"]
    if external:
        args += ["--baseline-predictions", str(source)]
    return rows, first, args, out


def fake(calls, fail=None, invalid=None):
    def infer(adapter, images, prompt):
        core = Path(images[-1]).stem
        texts = dict(re.findall(r"^([A-J])\. (.+)$", prompt, re.M))
        if "Visual definitions" in prompt:
            step = "q2_old" if "second look" in prompt else "q2_new"
            answer = "B" if int(core[1:]) % 2 else "A"
        elif "For the localization task" in prompt:
            step = "presence"
            answer = next(k for k, v in texts.items() if v == (
                "A visible localized defect region." if int(core[1:]) % 2 else "No visible localized defect region."))
        elif "Use the full TARGET frame" in prompt:
            step = "position"
            answer = next(k for k, v in texts.items() if v == "Center region.")
        elif runner.old.NONE in texts.values():
            step = "direct_q3"
            answer = next(k for k, v in texts.items() if v == runner.old.NONE)
        else:
            step = "baseline_q1" if set(texts.values()) == {"Normal.", "Anomalous."} else "baseline_q2"
            answer = "B" if int(core[1:]) % 2 else "A"
        calls.append((core, step, prompt, [Path(p).name for p in images]))
        if fail and fail(core, step):
            raise RuntimeError("fixture transient failure")
        return "unparseable" if invalid and invalid(core, step) else answer
    return infer


def run(rows, args, calls, **kwargs):
    return runner.main(args, load_qa=lambda _: rows, loader=lambda spec: {}, infer=fake(calls, **kwargs))


def test_full_candidate_q1_fixed_controls_and_zero_call_resume(tmp_path):
    rows, first, args, out = setup(tmp_path)
    source_before = (tmp_path/"baseline.jsonl").read_bytes()
    calls = []
    assert run(rows, args, calls) == 0
    assert len(calls) == 27 and not any(step.startswith("baseline") for _, step, _, _ in calls)
    answers = fixtures._csv(out/"answers.csv")
    assert len(answers) == 54
    assert all(r["predicted_choice"] == first[r["id"]] for r in answers if r["question_type"] == runner.agent.Q1)
    lb = {r["policy"]: r for r in fixtures._csv(out/"leaderboard.csv")}
    assert float(lb["candidate"]["q3_presence_balanced_accuracy"]) == 1
    assert all(r["q3_location_accuracy"] == "" for r in lb.values())
    assert float(lb["candidate"]["joint_accuracy"]) == round(5/6, 6)
    assert len(fixtures._csv(out/"metrics_by_asset.csv")) == 6
    assert len(fixtures._csv(out/"q2_by_defect.csv")) == 6
    assert fixtures._csv(out/"significance.csv")
    assert len(fixtures._csv(out/"q3_constant_baselines.csv")) == 2
    assert (tmp_path/"baseline.jsonl").read_bytes() == source_before
    assert not (out/"frozen_config.json").exists() and not (out/"selection.csv").exists()
    calls.clear()
    assert run(rows, args, calls) == 0 and calls == []
    meta = json.loads((out/"run_metadata.json").read_text())
    assert meta["called_this_run"] == 0 and meta["reused_calls"] == 39 and meta["complete"]


@pytest.mark.parametrize("invalid", [False, True])
def test_failed_or_invalid_candidate_excluded_and_only_one_retry(tmp_path, invalid):
    rows, _, args, out = setup(tmp_path)
    calls = []
    condition = lambda core, step: core == "c0" and step == "q2_new"
    assert run(rows, args, calls, **({"invalid": condition} if invalid else {"fail": condition})) == 1
    assert {r["policy"] for r in fixtures._csv(out/"leaderboard.csv")} == {"baseline", "previous_agent"}
    status = json.loads((out/"run_status.json").read_text())
    assert not status["complete"] and not status["selection_performed"]
    assert not json.loads((out/"run_metadata.json").read_text())["complete"]
    calls.clear()
    assert run(rows, args, calls) == 0 and [(core, step) for core, step, _, _ in calls] == [("c0", "q2_new")]


def test_missing_baseline_generated_then_q1_fixed(tmp_path):
    rows, _, args, out = setup(tmp_path, external=False)
    calls = []
    assert run(rows, args, calls) == 0
    assert sum(step == "baseline_q1" for _, step, _, _ in calls) == 6
    meta = json.loads((out/"run_metadata.json").read_text())
    assert meta["baseline_calls_this_run"] == 12 and meta["q1_candidate_calls"] == 0
    calls.clear()
    assert run(rows, args, calls) == 0 and not calls


def test_model_load_failure_blocks_completion_and_retries(tmp_path):
    rows, _, args, out = setup(tmp_path, external=False)
    loads = []
    def broken(spec):
        loads.append(spec)
        raise RuntimeError("load failed")
    assert runner.main(args, load_qa=lambda _: rows, loader=broken, infer=fake([])) == 1
    assert len(loads) == 1
    meta = json.loads((out/"run_metadata.json").read_text())
    assert meta["called_this_run"] == 1 and meta["adapter_load_blocked_steps"] == 11
    assert fixtures._csv(out/"leaderboard.csv") == []
    assert not json.loads((out/"run_status.json").read_text())["complete"]
    assert run(rows, args, []) == 0


def test_other_question_changes_and_gt_never_change_q3_prompt_or_routes(tmp_path):
    rows, _, args, out = setup(tmp_path/"a")
    changed, _, other_args, other_out = setup(tmp_path/"b")
    source = tmp_path/"b/baseline.jsonl"
    predictions = runner.agent.read_jsonl(source)
    for p in predictions:
        p["predicted_choice"] = p["output"] = "A" if p["output"] == "B" else "B"
    runner.save_records(source, predictions)
    for r in changed:
        r["answer"] = next(k for k in r["options"] if k != r["answer"])
        r["anomaly_gt"] = "normal" if r["anomaly_gt"] == "anomalous" else "anomalous"
    a, b = [], []
    assert run(rows, args, a) == run(changed, other_args, b) == 0
    assert [c for c in a if c[1] in ("presence", "position", "direct_q3", "q2_new")] == [c for c in b if c[1] in ("presence", "position", "direct_q3", "q2_new")]


def test_changed_inputs_config_and_identity_preserve_previous_output(tmp_path):
    rows, _, args, out = setup(tmp_path)
    assert run(rows, args, []) == 0
    before = (out/"answers.csv").read_bytes()
    source = tmp_path/"baseline.jsonl"
    payload = source.read_bytes()
    source.write_bytes(payload+b"\n")
    with pytest.raises(ValueError, match="inputs/config/code changed"):
        run(rows, args, [])
    source.write_bytes(payload)
    config = json.loads(runner.CONFIG.read_text())
    config["q3"] = "new Q3 review"
    new_config = tmp_path/"bad.json"
    new_config.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="pre-gold lock"):
        run(rows, [*args, "--config", str(new_config)], [])
    path = out/"predictions"/MODEL/"steps.jsonl"
    records = runner.agent.read_jsonl(path)
    records[0]["identity"] = "tampered"
    runner.save_records(path, records)
    with pytest.raises(ValueError, match="identity mismatch"):
        run(rows, args, [])
    assert (out/"answers.csv").read_bytes() == before
    assert not json.loads((out/"run_status.json").read_text())["complete"]


def test_no_gold_overwrite_bad_ref_or_dev_qa(tmp_path):
    rows, _, args, out = setup(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        run(rows, [*args, "--out-dir", str(tmp_path/"gold_v2_reference")], [])
    with pytest.raises(ValueError, match="not dev"):
        run([{**r, "protocol": "gold_v2_dev"} for r in rows], args, [])
    runner.agent.cons.write_csv(tmp_path/"refs.csv", [{"image": r["image"], "retrieved_diffdate": "bad/train/good/ref.png"} for r in rows[::2]])
    with pytest.raises(ValueError, match="asset differs"):
        run(rows, args, [])
    assert not out.exists()


def test_notebook_display_checks_real_fixture_outputs_and_failure_gate(tmp_path):
    rows, _, args, out = setup(tmp_path)
    assert run(rows, args, []) == 0
    import pandas as pd
    path = Path(__file__).resolve().parents[1]/"notebooks/lvlm_gold_v2_q23_candidate_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for i, c in enumerate(notebook["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"cell{i}", "exec")
    cell = "".join(notebook["cells"][5]["source"])
    context = {"pd": pd, "json": json, "RUN_ROOT": out, "display": lambda obj: None}
    exec(cell, context)
    status = json.loads((out/"run_status.json").read_text())
    status["complete"] = False
    runner.old.atomic_json(out/"run_status.json", status)
    with pytest.raises(RuntimeError, match="실패/무효/누락"):
        exec(cell, context)


def test_checkpoint_keeps_tail_after_unexpected_interruption(tmp_path, monkeypatch):
    rows, _, args, out = setup(tmp_path)
    assert run(rows, args, []) == 0
    path = out/"predictions"/MODEL/"steps.jsonl"
    before = path.read_bytes()
    def interrupt(*unused, **kw):
        raise RuntimeError("unexpected routing interruption")
    monkeypatch.setattr(runner.refined, "definition_prompt", interrupt)
    with pytest.raises(RuntimeError, match="routing interruption"):
        run(rows, args, [])
    assert path.read_bytes() == before
    assert not json.loads((out/"run_status.json").read_text())["complete"]


def test_external_reference_and_saved_image_changes_rejected(tmp_path):
    rows, _, args, out = setup(tmp_path)
    source = tmp_path/"baseline.jsonl"
    records = runner.agent.read_jsonl(source)
    payload = source.read_bytes()
    records[0]["reference_image"] = "/wrong/reference.png"
    runner.save_records(source, records)
    with pytest.raises(ValueError, match="reference differs"):
        run(rows, args, [])
    source.write_bytes(payload)
    assert run(rows, args, []) == 0
    old_answers = (out/"answers.csv").read_bytes()
    image = out/"_images/target"/rows[0]["image"]
    image.write_bytes(fixtures._png(45))
    with pytest.raises(ValueError, match="image bytes changed"):
        run(rows, args, [])
    assert (out/"answers.csv").read_bytes() == old_answers


def test_colab_preparation_default_paths_no_manual_work_and_qa_preserved(tmp_path):
    rows = [r for r in fixtures._rows(246, dev=False) if r["question_type"] != runner.agent.Q3]
    qa_dir = tmp_path/"lvlm_insplad_qa/gold_v2_final"
    qa_dir.mkdir(parents=True)
    runner.save_records(qa_dir/"master_qa_test.jsonl", rows)
    original_q3 = [r for r in fixtures._rows(42, dev=False) if r["question_type"] == runner.agent.Q3]
    runner.save_records(qa_dir/"localization_qa_test.jsonl", original_q3)
    runner.old.atomic_json(qa_dir/"build_summary.json", {"quality_status": "FROZEN", "final_master_images": 246})
    before = {p.name: p.read_bytes() for p in qa_dir.iterdir()}
    # find_zip must succeed using the ordinary default dataset paths.
    for name in ("supervised_fault_classification.zip", "unsupervised_anomaly_detection.zip"):
        (tmp_path/name).write_bytes(b"fixture zip file")
    old_dir = tmp_path/"lvlm_insplad_results/gold_v2_main_rerun"
    old_refs = old_dir/"refs_gold/retrieved_refs.csv"
    runner.agent.cons.write_csv(old_refs, [{"image": r["image"],
        "retrieved_diffdate": r["asset_category"]+"/train/good/ref.png"} for r in rows[::2]])
    old_bytes = old_refs.read_bytes()
    out = tmp_path/"lvlm_insplad_results/gold_v2_q23_candidate_v1"
    notebook = json.loads((runner.ROOT/"notebooks/lvlm_gold_v2_q23_candidate_colab.ipynb").read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][3]["source"])
    commands = []
    context = {"DRIVE_ROOT": tmp_path, "QA_DIR": qa_dir, "OLD_GOLD_DIR": old_dir,
        "RUN_ROOT": out, "REPO_DIR": runner.ROOT, "json": json, "sys": sys, "Path": Path,
        "SUP_ZIP_NAME": "supervised_fault_classification.zip", "UNSUP_ZIP_NAME": "unsupervised_anomaly_detection.zip",
        "SUP_ZIP_SIZE": len(b"fixture zip file"), "UNSUP_ZIP_SIZE": len(b"fixture zip file"),
        "SUP_ZIP_PATH": "", "UNSUP_ZIP_PATH": "", "run": lambda cmd: commands.append(cmd)}
    exec(source, context)
    assert not commands and context["REFS_CSV"] == old_refs
    assert len(runner.agent.read_jsonl(out/"retrieval_input_qa.jsonl")) == 492
    assert {p.name: p.read_bytes() for p in qa_dir.iterdir()} == before and old_refs.read_bytes() == old_bytes
    # Missing refs route to the builder in the new folder, never the original folder.
    second = tmp_path/"lvlm_insplad_results/gold_v2_q23_candidate_other"
    def build(cmd):
        commands.append(cmd)
        assert "build_retrieved_refs.py" in str(cmd[1])
        dest = Path(cmd[cmd.index("--out-dir")+1])
        assert dest == second/"refs_gold"
        runner.agent.cons.write_csv(dest/"retrieved_refs.csv", [{"image": r["image"],
            "retrieved_diffdate": r["asset_category"]+"/train/good/ref.png"} for r in rows[::2]])
    context.update(RUN_ROOT=second, OLD_GOLD_DIR=tmp_path/"missing_old_run", run=build)
    exec(source, context)
    assert len(commands) == 1 and old_refs.read_bytes() == old_bytes

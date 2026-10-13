from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_gold_v2_q2_q3_agent_dev as runner
import test_gold_v2_verify_agent as fixtures


def setup(tmp):
    rows = [r for r in fixtures._rows() if r["question_type"] != runner.agent.Q3]
    first, image_only = fixtures._stage(rows)
    first = {r["id"]: first[r["id"]] for r in rows}
    common, qa = fixtures._inputs(tmp, rows, first, image_only)
    out = tmp / "gold_v2_q2_q3_independent_dev_test"
    args = [*common, "--dev-qa", str(qa), "--out-dir", str(out), "--n-images", "6", "--checkpoint-every", "2"]
    return rows, first, args, out


def fake(calls, fail=None, invalid=False):
    def infer(adapter, images, prompt):
        core = Path(images[-1]).stem
        qt = runner.agent.Q2 if f"[{core}]" in prompt else runner.agent.Q3
        calls.append((core, qt, prompt))
        if fail and fail(prompt):
            raise RuntimeError("transient test failure")
        if invalid:
            return "no parseable answer"
        if qt == runner.agent.Q2:
            return "B" if int(core[1:]) % 2 else "A"
        texts = dict(re.findall(r"^([A-J])\. (.+)$", prompt, re.M))
        if "For the localization task" in prompt:
            wanted = "A visible localized defect region." if int(core[1:]) % 2 else "No visible localized defect region."
        elif "Use the full TARGET frame" in prompt:
            wanted = "Center region."
        else:
            wanted = runner.NONE  # deliberately wrong on anomalies; presence tool fixes it
        return next(k for k, v in texts.items() if v == wanted)
    return infer


def run(args, calls, **kw):
    return runner.main(args, loader=lambda spec: {}, infer=fake(calls, **kw))


def test_partial_labels_and_options_do_not_leak_gt():
    rows = [r for r in fixtures._rows() if r["question_type"] != runner.agent.Q3]
    q3 = runner.q3_rows(rows)
    changed = runner.q3_rows([{**r, "anomaly_gt": "normal" if r["anomaly_gt"] == "anomalous" else "anomalous"} for r in rows])
    for a, b in zip(q3, changed):
        assert a["options"] == b["options"] and len(set(a["options"].values())) == 10
        assert runner.direct_prompt(a) == runner.direct_prompt(b)
        assert runner.presence_task(a)[1] == runner.presence_task(b)[1]
        assert runner.position_prompt(a) == runner.position_prompt(b)
        assert a["answer"] == (a["none_choice"] if a["anomaly_gt"] == "normal" else None)
    for choices in ({r["id"]: r["none_choice"] for r in q3},
                    {r["id"]: next(k for k in r["options"] if k != r["none_choice"]) for r in q3}):
        m = runner.presence_metrics(q3, choices)
        assert m["q3_presence_balanced_accuracy"] == .5
        assert m["q3_location_accuracy"] is None
        assert m["q3_anomalous_location_labels"] == 0


def test_full_run_q1_fixed_equal_budget_and_resume(tmp_path):
    rows, first, args, out = setup(tmp_path)
    sentinel = tmp_path / "existing_results" / "frozen_config.json"
    sentinel.parent.mkdir()
    sentinel.write_bytes(b"preserve exactly")
    calls = []
    assert run(args, calls) == 0
    assert len(calls) == 36 and all(qt != runner.agent.Q1 for _, qt, _ in calls)
    answers = fixtures._csv(out / "answers.csv")
    assert len(answers) == 6 * 3 * 4
    assert all(r["predicted_choice"] == first[r["id"]] for r in answers if r["question_type"] == runner.agent.Q1)
    status = json.loads((out / "run_status.json").read_text())
    counts = {s["policy"]: s["q2_q3_calls"] for s in status}
    assert counts["matched_reask"] == counts["q2_define_q3_independent"] == 21
    lb = {r["policy"]: r for r in fixtures._csv(out / "leaderboard.csv")}
    assert float(lb["baseline"]["q3_presence_balanced_accuracy"]) == .5
    assert float(lb["q2_define_q3_independent"]["q3_presence_balanced_accuracy"]) == 1
    assert all(r["q3_location_accuracy"] == "" for r in lb.values())
    assert any(r["baseline"] == "matched_reask" for r in fixtures._csv(out / "significance.csv"))
    import pandas as pd
    notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks/lvlm_gold_v2_q2_q3_agent_dev_colab.ipynb").read_text(encoding="utf-8"))
    report_cell = "".join(notebook["cells"][5]["source"])
    exec(report_cell, {"pd": pd, "json": json, "RUN_ROOT": out, "display": lambda obj: None})
    assert sentinel.read_bytes() == b"preserve exactly"
    assert not (out / "frozen_config.json").exists()
    assert not (out / "selection.csv").exists()
    calls.clear()
    assert run(args, calls) == 0 and calls == []


def test_failure_excluded_and_only_failed_calls_retried(tmp_path):
    _, _, args, out = setup(tmp_path)
    calls = []
    assert run(args, calls, fail=lambda p: "[c0]" in p and "Visual definitions" in p) == 1
    lb = fixtures._csv(out / "leaderboard.csv")
    assert {r["policy"] for r in lb} == {"baseline", "matched_reask"}
    saved = runner.agent.read_jsonl(out / "predictions" / fixtures.MODEL / "steps.jsonl")
    assert sum(bool(r["error"]) for r in saved) == 1
    calls.clear()
    assert run(args, calls) == 0 and len(calls) == 1
    assert json.loads((out / "run_metadata.json").read_text())["row_errors"] == 0


def test_other_question_answers_do_not_change_q3(tmp_path):
    _, _, args_a, out_a = setup(tmp_path / "a")
    _, _, args_b, out_b = setup(tmp_path / "b")
    path = tmp_path / "b" / "stage1" / fixtures.MODEL / "normal_ref_retrieved_diffdate.jsonl"
    preds = runner.agent.read_jsonl(path)
    for p in preds:
        p["predicted_choice"] = "B" if p["predicted_choice"] == "A" else "A"
    runner.agent.base.write_jsonl(path, preds)
    ca, cb = [], []
    assert run(args_a, ca) == run(args_b, cb) == 0
    assert [c for c in ca if c[1] == runner.agent.Q3] == [c for c in cb if c[1] == runner.agent.Q3]
    q3a = [r for r in fixtures._csv(out_a / "answers.csv") if r["question_type"] == runner.agent.Q3]
    q3b = [r for r in fixtures._csv(out_b / "answers.csv") if r["question_type"] == runner.agent.Q3]
    assert q3a == q3b


def test_changed_baseline_or_zip_blocks_overwriting_results(tmp_path):
    _, _, args, out = setup(tmp_path)
    assert run(args, []) == 0
    before = (out / "answers.csv").read_bytes()
    baseline = tmp_path / "stage1" / fixtures.MODEL / "normal_ref_retrieved_diffdate.jsonl"
    payload = baseline.read_bytes()
    records = runner.agent.read_jsonl(baseline)
    records[0]["predicted_choice"] = "A" if records[0]["predicted_choice"] == "B" else "B"
    runner.agent.base.write_jsonl(baseline, records)
    with pytest.raises(ValueError, match="different inputs"):
        run(args, [])
    baseline.write_bytes(payload)
    with zipfile.ZipFile(tmp_path / "sup.zip", "w") as z:
        for i in range(6):
            z.writestr(f"sup/c{i}.png", fixtures._png(80))
    with pytest.raises(ValueError, match="image bytes changed"):
        run(args, [])
    assert (out / "answers.csv").read_bytes() == before


@pytest.mark.parametrize("problem", ["failed", "missing", "invalid", "wrong_folder", "gold"])
def test_preflight_blocks_bad_inputs_without_inference(tmp_path, problem):
    _, _, args, out = setup(tmp_path)
    path = tmp_path / "stage1" / fixtures.MODEL / "normal_ref_retrieved_diffdate.jsonl"
    records = runner.agent.read_jsonl(path)
    if problem == "failed":
        records[0]["error"] = "OOM"
    elif problem == "missing":
        records.pop(0)
    elif problem == "invalid":
        records[0]["predicted_choice"] = None
    elif problem == "wrong_folder":
        args[args.index("--out-dir")+1] = str(tmp_path / "gold_v2_dev_baselines")
    elif problem == "gold":
        qa = Path(args[args.index("--dev-qa")+1])
        rows = runner.agent.read_jsonl(qa)
        rows[0]["protocol"] = "gold_v2_final"
        runner.agent.base.write_jsonl(qa, rows)
    runner.agent.base.write_jsonl(path, records)
    calls = []
    with pytest.raises(ValueError):
        run(args, calls)
    assert calls == [] and not out.exists()


def test_invalid_tool_outputs_preserved_and_do_not_fake_location_score(tmp_path):
    _, first, args, out = setup(tmp_path)
    calls = []
    assert run(args, calls, invalid=True) == 0
    lb = fixtures._csv(out / "leaderboard.csv")
    assert all(float(r["q3_invalid_rate"]) == 1 for r in lb)
    ans = fixtures._csv(out / "answers.csv")
    assert all(r["predicted_choice"] == first[r["id"]] for r in ans if r["question_type"] == runner.agent.Q2)
    status = json.loads((out / "run_status.json").read_text())
    assert all(s["invalid_tool_answers"] > 0 for s in status)
    assert all(r["q3_location_accuracy"] == "" for r in lb)


def test_colab_notebook_cells_compile():
    path = Path(__file__).resolve().parents[1] / "notebooks/lvlm_gold_v2_q2_q3_agent_dev_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert not cell["outputs"] and cell["execution_count"] is None
            compile("".join(cell["source"]), f"cell-{i}", "exec")
    source = "\n".join("".join(c["source"]) for c in notebook["cells"])
    assert "gold_v2_q2_q3_independent_dev_v1" in source
    assert "q3_location_accuracy" in source
    assert "run_gold_v2_q2_q3_agent_dev.py" in source
    assert "files.download" in source

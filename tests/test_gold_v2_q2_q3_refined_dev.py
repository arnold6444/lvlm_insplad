from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_gold_v2_q2_q3_refined_dev as refined
import test_gold_v2_verify_agent as fixtures

old = refined.old


def setup(tmp, n=6):
    rows = [r for r in fixtures._rows(n) if r["question_type"] != old.agent.Q3]
    first, image_only = fixtures._stage(rows)
    first = {r["id"]: first[r["id"]] for r in rows}
    common, qa = fixtures._inputs(tmp, rows, first, image_only)
    out = tmp / "gold_v2_q2_q3_refined_dev_test"
    args = [*common, "--dev-qa", str(qa), "--out-dir", str(out), "--n-images", str(n), "--checkpoint-every", "20"]
    args[args.index("--n-boot")+1] = "10"
    return rows, first, args, out


def mock_infer(calls, fail=None, invalid_review=False):
    def infer(adapter, images, prompt):
        core = Path(images[-1]).stem
        positive = int(core[1:]) % 2 == 1
        texts = dict(re.findall(r"^([A-J])\. (.+)$", prompt, re.M))
        if f"[{core}]" in prompt:
            step = "q2"
            answer = "B" if positive else "A"
        elif "Two independent attempts" in prompt:
            step, answer = "review", "Center region."
            if invalid_review:
                calls.append((core, step, prompt))
                return "unparseable"
        elif "Does the TARGET contain visible defect evidence" in prompt:
            step = "presence"
            # c1 is positive at presence then incorrectly none at position; other odd cores miss at presence.
            answer = "Visible defect evidence." if core == "c1" else "No visible defect evidence."
        elif "Where is the visible defect evidence" in prompt:
            step = "position"
            answer = "Center region." if positive and core != "c1" else old.NONE
        elif "For the localization task" in prompt:
            step, answer = "legacy_presence", "No visible localized defect region."
        else:
            step, answer = "direct", old.NONE
        calls.append((core, step, prompt))
        if fail and fail(core, step):
            raise RuntimeError("transient failure")
        return answer if step == "q2" else next(k for k, v in texts.items() if v == answer)
    return infer


def run(args, calls, **kw):
    return refined.main(args, loader=lambda spec: {}, infer=mock_infer(calls, **kw))


def test_full_dev_and_fixed_design_cohort_are_disjoint():
    rows = [r for r in fixtures._rows(240) if r["question_type"] != old.agent.Q3]
    selected, groups = refined.select_rows(rows)
    assert len(selected) == 480
    assert len(groups["design_dev40"]) == 40 and len(groups["additional_dev"]) == 200
    assert not groups["design_dev40"] & groups["additional_dev"]
    assert groups["all_dev"] == groups["design_dev40"] | groups["additional_dev"]
    original = {r["core_id"] for r in old.diagnostic.select_subset(rows)}
    assert groups["design_dev40"] == original
    assert refined.select_rows(rows, 160, 7)[1]["design_dev40"] == original
    bad = [{**r, "protocol": "gold_v2_final"} for r in rows]
    with pytest.raises(ValueError, match="only gold_v2_dev"):
        refined.select_rows(bad)


def test_fresh_definition_ablation_removes_review_but_keeps_old_visual_definitions():
    q2 = next(r for r in fixtures._rows() if r["question_type"] == old.agent.Q2)
    fresh = refined.definition_prompt(q2)
    assert "first look" not in fresh and "second look" not in fresh
    assert all(old.agent.DEFINITIONS[text] in fresh for text in q2["options"].values())
    revised = refined.definition_prompt(q2, True)
    assert "Bright orange is not required" in revised and "metal cap can remain" in revised
    assert "Background branches" in revised and "no visible corrosion" in revised
    changed = {**q2, "answer": "D", "defect_gt": "normal", "anomaly_gt": "normal"}
    assert refined.definition_prompt(changed, True) == revised


def test_always_position_and_disagreement_review_fix_both_veto_paths(tmp_path):
    _, first, args, out = setup(tmp_path)
    calls = []
    assert run(args, calls) == 0
    assert len([c for c in calls if c[1] == "position"]) == 6
    assert {c[0] for c in calls if c[1] == "review"} == {"c1", "c3", "c5"}
    answers = fixtures._csv(out / "answers.csv")
    assert len(answers) == 6*3*len(refined.POLICIES)
    assert all(r["predicted_choice"] == first[r["id"]] for r in answers if r["question_type"] == old.agent.Q1)
    lb = {r["policy"]: r for r in fixtures._csv(out / "leaderboard.csv")}
    assert float(lb["previous_agent"]["q3_presence_balanced_accuracy"]) == .5
    assert float(lb["q2_cues_q3_review"]["q3_presence_balanced_accuracy"]) == 1
    assert all(r["q3_location_accuracy"] == "" for r in lb.values())
    counts = {r["policy"]: r["q2_q3_calls"] for r in json.loads((out / "run_status.json").read_text())}
    assert counts["q2_cues_q3_review"] == counts["matched_reask"] == 27
    assert not (out / "frozen_config.json").exists() and not (out / "selection.csv").exists()
    calls.clear()
    assert run(args, calls) == 0 and not calls


def test_additional_dev_metrics_and_ci_coverage_and_notebook_display(tmp_path):
    _, _, args, out = setup(tmp_path, 60)
    assert run(args, []) == 0
    metrics = fixtures._csv(out / "cohort_metrics.csv")
    for cohort, support in (("all_dev",60),("design_dev40",40),("additional_dev",20)):
        mine = [r for r in metrics if r["cohort"] == cohort]
        assert len(mine) == len(refined.POLICIES)
        assert all(int(r["images"]) == support for r in mine)
        assert all(int(r["q3_normal_images"])+int(r["q3_anomalous_images"]) == support for r in mine)
        assert all(r["q3_location_accuracy"] == "" for r in mine)
    assert {r["cohort"] for r in fixtures._csv(out / "cohort_significance.csv")} == {"all_dev","design_dev40","additional_dev"}
    assert {r["cohort"] for r in fixtures._csv(out / "cohort_q2_by_defect.csv")} == {"all_dev","design_dev40","additional_dev"}
    import pandas as pd
    nb = json.loads((Path(__file__).resolve().parents[1]/"notebooks/lvlm_gold_v2_q2_q3_refined_dev_colab.ipynb").read_text(encoding="utf-8"))
    exec("".join(nb["cells"][5]["source"]), {"pd":pd,"json":json,"RUN_ROOT":out,"display":lambda obj:None})


def test_failure_exclusion_retry_and_newly_required_dependents(tmp_path):
    _, _, args, out = setup(tmp_path)
    calls = []
    assert run(args, calls, fail=lambda c,s:c == "c1" and s == "position") == 1
    lb = {r["policy"] for r in fixtures._csv(out / "leaderboard.csv")}
    assert "q2_cues_q3_review" not in lb and "q2_cues_q3_position" not in lb
    assert "previous_agent" in lb
    assert all(r["policy"] not in {"q2_cues_q3_review","q2_cues_q3_position"} for r in fixtures._csv(out / "cohort_metrics.csv"))
    calls.clear()
    assert run(args, calls) == 0
    assert [(c,s) for c,s,_ in calls] == [("c1","position"),("c1","review"),("c1","direct")]


def test_invalid_review_falls_back_to_own_position(tmp_path):
    _, _, args, out = setup(tmp_path)
    assert run(args, [], invalid_review=True) == 0
    rows = fixtures._csv(out / "answers.csv")
    loc = {(r["policy"],r["id"]):r["predicted_choice"] for r in rows if r["question_type"] == old.agent.Q3}
    assert all(loc[("q2_cues_q3_review",qid)] == choice for (policy,qid),choice in loc.items() if policy == "q2_cues_q3_position")
    status = json.loads((out / "run_status.json").read_text())
    assert next(s for s in status if s["policy"] == "q2_cues_q3_review")["invalid_tool_answers"] == 3


def test_refined_q3_prompts_do_not_expose_evaluation_labels():
    qa = [r for r in fixtures._rows() if r["question_type"] != old.agent.Q3]
    rows = old.q3_rows(qa)
    for row in rows:
        changed = {**row, "anomaly_gt": "anomalous" if row["anomaly_gt"] == "normal" else "normal",
                   "answer": "UNKNOWN", "defect_gt": "LABEL_ONLY_SENTINEL"}
        assert refined.presence_task(row)[1] == refined.presence_task(changed)[1]
        assert refined.position_prompt(row) == refined.position_prompt(changed)
        yes = next(k for k,v in refined.presence_task(row)[0]["options"].items() if v == "Visible defect evidence.")
        assert refined.review_prompt(row, yes, row["none_choice"]) == refined.review_prompt(changed, yes, row["none_choice"])


def test_q3_never_uses_q1_or_q2_answers_and_old_results_are_preserved(tmp_path):
    _, _, a, oa = setup(tmp_path/"a")
    _, _, b, ob = setup(tmp_path/"b")
    path = tmp_path/"b"/"stage1"/fixtures.MODEL/"normal_ref_retrieved_diffdate.jsonl"
    records = old.agent.read_jsonl(path)
    for r in records:
        r["predicted_choice"] = "B" if r["predicted_choice"] == "A" else "A"
    old.agent.base.write_jsonl(path, records)
    ca,cb = [],[]
    assert run(a,ca) == run(b,cb) == 0
    assert [c for c in ca if c[1] != "q2"] == [c for c in cb if c[1] != "q2"]
    before = (oa/"answers.csv").read_bytes()
    a[a.index("--out-dir")+1] = str(tmp_path/"gold_v2_q2_q3_independent_dev_v1")
    with pytest.raises(ValueError, match="separate"):
        run(a,[])
    assert (oa/"answers.csv").read_bytes() == before


def test_notebook_compiles_default_full_dev_and_separate_output():
    nb = json.loads((Path(__file__).resolve().parents[1]/"notebooks/lvlm_gold_v2_q2_q3_refined_dev_colab.ipynb").read_text(encoding="utf-8"))
    for i,c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]),f"cell-{i}","exec")
            assert c["outputs"] == [] and c["execution_count"] is None
    source = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "N_IMAGES = 240" in source and "gold_v2_q2_q3_refined_dev_v1" in source
    assert "additional_dev" in source and "files.download" in source

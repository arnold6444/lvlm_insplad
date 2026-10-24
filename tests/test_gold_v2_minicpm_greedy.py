from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(ROOT/"tests"))
import build_gold_v2_minicpm_greedy_notebook as builder
import run_gold_v2_minicpm_greedy as runner
import test_gold_v2_verify_agent as fixtures

COND_KEYS = runner.CONDITIONS


def setup(tmp, *, bad_normal=False, leaky=False):
    rows = fixtures._rows(6, dev=False)
    for r in rows:
        r["normal_reference_image"] = f"unsup/normal_{r['asset_category']}.png"
    qa = tmp/"qa"
    qa.mkdir(parents=True)
    runner.base.write_jsonl(qa/"master_qa_test.jsonl", [r for r in rows if r["question_type"] != fixtures.Q3])
    runner.base.write_jsonl(qa/"localization_qa_test.jsonl", [r for r in rows if r["question_type"] == fixtures.Q3])
    (qa/"build_summary.json").write_text(json.dumps({"quality_status": "FROZEN", "final_master_images": 6}))
    with zipfile.ZipFile(tmp/"sup.zip", "w") as z:
        for name in sorted({r["image"] for r in rows}):
            z.writestr(name, fixtures._png(30))
        for asset in sorted({r["asset_category"] for r in rows}):
            z.writestr(f"sup/train/pos_{asset}.png", fixtures._png(90))
    with zipfile.ZipFile(tmp/"unsup.zip", "w") as z:
        for asset in sorted({r["asset_category"] for r in rows}):
            z.writestr(f"unsup/normal_{asset}.png", fixtures._png(200))
    manifest = [{"core_id": r["core_id"], "asset_category": r["asset_category"], "target_image": r["image"],
                 "normal_reference_image": "unsup/wrong.png" if bad_normal else r["normal_reference_image"],
                 "positive_reference_image": f"sup/train/pos_{r['asset_category']}.png",
                 "positive_reference_defect_label": "rust", "positive_reference_similarity": "0.9",
                 "target_gt_used_for_positive_selection": "true" if leaky else "false"}
                for r in rows if r["question_type"] == fixtures.Q1]
    runner.base.write_csv(tmp/"reference_manifest.csv", manifest)
    sampled = [{"model_key": runner.MODEL, "condition": c, "q1_anomaly_f1": .5, "q2_defect_macro_f1": .4,
                "joint_accuracy": .3, "consistency_error": .2, "false_defect_rate": .1, "q3_accuracy": .5,
                "q1_invalid_rate": 0, "q2_invalid_rate": 0, "q3_invalid_rate": 0} for c in COND_KEYS]
    runner.base.write_csv(tmp/"sampled.csv", sampled)
    out = tmp/"gold_v2_minicpm_greedy_test"
    args = ["--qa-dir", str(qa), "--sup-zip", str(tmp/"sup.zip"), "--unsup-zip", str(tmp/"unsup.zip"),
            "--out-dir", str(out), "--reference-manifest", str(tmp/"reference_manifest.csv"),
            "--sampled-leaderboard", str(tmp/"sampled.csv"), "--checkpoint-every", "5", "--n-boot", "20"]
    return rows, args, out


def condition_of(prompt):
    if "using only the target image" in prompt:
        return "image_only"
    if "Image 3 is the TARGET" in prompt:
        return "normal_positive_reference"
    if "ANOMALOUS reference" in prompt:
        return "positive_reference"
    return "normal_reference"


def fake(rows, calls, fail=None):
    truth = {(r["core_id"], r["question_type"]): r["answer"] for r in rows}

    def infer(adapter, images, prompt):
        core, qt = fixtures._who(prompt)
        condition = condition_of(prompt)
        calls.append({"core": core, "qt": qt, "condition": condition, "images": [Path(p).name for p in images],
                      "prompt": prompt})
        if fail and fail(core, qt, condition):
            raise RuntimeError("fixture transient failure")
        return truth[(core, qt)] if condition == "normal_reference" else "A"
    return infer


def run(rows, args, calls, **kw):
    # The real gold loader, passed explicitly so the 534-question size guard allows this 6-image fixture.
    return runner.main(args, loader=lambda spec: {}, infer=fake(rows, calls, **kw), load_qa=runner.v2.load_gold_qa_v2)


def test_four_conditions_use_original_prompts_images_and_resume_without_calls(tmp_path):
    rows, args, out = setup(tmp_path)
    calls = []
    assert run(rows, args, calls) == 0
    assert len(calls) == 4 * len(rows)
    by_cond = {c: [x for x in calls if x["condition"] == c] for c in COND_KEYS}
    assert {c: len(v) for c, v in by_cond.items()} == {c: len(rows) for c in COND_KEYS}
    row = next(r for r in rows if r["id"] == "c1-q2")
    expected = {"image_only": runner.base.prompt_for(dict(row)),
                **{c: runner.ref.prompt_for_condition(dict(row), c) for c in COND_KEYS[1:]}}
    for c in COND_KEYS:
        call = next(x for x in by_cond[c] if x["core"] == "c1" and x["qt"] == fixtures.Q2)
        assert call["prompt"] == expected[c]
    images = {c: next(x for x in by_cond[c] if x["core"] == "c1")["images"] for c in COND_KEYS}
    assert images == {"image_only": ["c1.png"], "normal_reference": ["normal_vari-grip.png", "c1.png"],
                      "positive_reference": ["pos_vari-grip.png", "c1.png"],
                      "normal_positive_reference": ["normal_vari-grip.png", "pos_vari-grip.png", "c1.png"]}
    lb = {r["condition"]: r for r in fixtures._csv(out/"leaderboard.csv")}
    assert set(lb) == set(COND_KEYS)
    assert float(lb["normal_reference"]["q1_anomaly_f1"]) == 1.0 and float(lb["normal_reference"]["q3_accuracy"]) == 1.0
    cmp = fixtures._csv(out/"comparison_vs_sampled.csv")
    assert {r["condition"] for r in cmp} == set(COND_KEYS) and all(r["sampled_published"] for r in cmp)
    sig = fixtures._csv(out/"significance.csv")
    assert {(r["baseline"], r["condition"]) for r in sig} == set(runner.COMPARISONS)
    meta = json.loads((out/"run_metadata.json").read_text())
    assert meta["complete"] and not meta["published"] and "greedy" in meta["decoding"]
    assert "publish_results_to_github" not in sys.modules
    calls.clear()
    assert run(rows, args, calls) == 0 and calls == []
    assert json.loads((out/"run_metadata.json").read_text())["reused_calls"] == 4 * len(rows)


def test_errors_block_completion_and_only_failed_calls_are_retried(tmp_path):
    rows, args, out = setup(tmp_path)
    calls = []
    fail = lambda core, qt, condition: core == "c2" and qt == fixtures.Q3 and condition == "positive_reference"
    assert run(rows, args, calls, fail=fail) == 1
    status = json.loads((out/"run_status.json").read_text())
    assert not status["complete"]
    assert {s["condition"]: s["row_errors"] for s in status["conditions"]}["positive_reference"] == 1
    assert "positive_reference" not in {r["condition"] for r in fixtures._csv(out/"leaderboard.csv")}
    calls.clear()
    assert run(rows, args, calls) == 0
    assert [(x["core"], x["qt"], x["condition"]) for x in calls] == [("c2", fixtures.Q3, "positive_reference")]


@pytest.mark.parametrize("kwargs, message", [({"bad_normal": True}, "differs from frozen QA"),
                                             ({"leaky": True}, "used target labels")])
def test_reference_manifest_must_match_qa_and_be_label_free(tmp_path, kwargs, message):
    rows, args, out = setup(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=message):
        run(rows, args, [])
    assert not out.exists()


def test_wrong_folder_and_changed_inputs_are_refused(tmp_path):
    rows, args, out = setup(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        run(rows, [*args, "--out-dir", str(tmp_path/"gold_v2_reference")], [])
    assert run(rows, args, []) == 0
    with pytest.raises(ValueError, match="inputs/code changed"):
        run(rows, [*args, "--n-boot", "30"], [])


def test_notebook_is_built_from_current_code_never_publishes_and_gates_outputs(tmp_path):
    notebook = json.loads(builder.NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook == builder.build(), "rebuild: python scripts/build_gold_v2_minicpm_greedy_notebook.py"
    source = "".join("".join(c["source"]) for c in notebook["cells"])
    assert "publish_results_to_github" not in source and 'userdata.get("GITHUB_TOKEN")' not in source
    for i, c in enumerate(notebook["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"cell{i}", "exec")
    config = {}
    exec("".join(notebook["cells"][1]["source"]), config)
    for rel in ("scripts/run_gold_v2_minicpm_greedy.py", "scripts/run_gold_v2_reference_ablation.py",
                "results/latest/gold_v2_reference/reference_manifest.csv"):
        assert rel in config["CODE_BLOBS"]
    run_cell = "".join(notebook["cells"][4]["source"])
    assert "RUN_RC = run(" in run_cell and '"--invocation-id", INVOCATION_ID' in run_cell
    rows, args, out = setup(tmp_path)
    assert run(rows, [*args, "--invocation-id", "first"], []) == 0
    import pandas as pd
    cell = "".join(notebook["cells"][5]["source"])
    context = {"pd": pd, "json": json, "RUN_ROOT": out, "display": lambda obj: None, "RUN_RC": 0, "INVOCATION_ID": "first"}
    exec(cell, context)
    # A re-run that fails before writing anything leaves the earlier run's complete files behind: never accept them.
    with pytest.raises(ValueError, match="inputs/code changed"):
        run(rows, [*args, "--n-boot", "30", "--invocation-id", "second"], [])
    assert json.loads((out/"run_status.json").read_text())["complete"]
    with pytest.raises(RuntimeError, match="종료 코드 1"):
        exec(cell, {**context, "RUN_RC": 1, "INVOCATION_ID": "second"})
    with pytest.raises(RuntimeError, match="이번 실행에서 나온 것이 아닙니다"):
        exec(cell, {**context, "INVOCATION_ID": "second"})
    status = json.loads((out/"run_status.json").read_text())
    status["complete"] = False
    (out/"run_status.json").write_text(json.dumps(status))
    with pytest.raises(RuntimeError, match="오류/누락"):
        exec(cell, context)


def test_status_belongs_to_this_invocation_even_when_aggregation_crashes(tmp_path, monkeypatch):
    rows, args, out = setup(tmp_path)
    assert run(rows, [*args, "--invocation-id", "first"], []) == 0
    meta = json.loads((out/"run_metadata.json").read_text())
    assert meta["invocation_id"] == json.loads((out/"run_status.json").read_text())["invocation_id"] == "first"
    monkeypatch.setattr(runner, "aggregate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError, match="crash"):
        run(rows, [*args, "--invocation-id", "second"], [])
    status = json.loads((out/"run_status.json").read_text())
    assert status["status"] == "running" and not status["complete"] and status["invocation_id"] == "second"


def test_required_anomaly_and_structured_metrics_keep_invalid_normal_apart_from_fp(tmp_path):
    rows, args, out = setup(tmp_path)
    infer = fake(rows, [])

    def with_invalid(adapter, images, prompt):
        core, qt = fixtures._who(prompt)
        return "no idea" if (core, qt, condition_of(prompt)) == ("c0", fixtures.Q1, "image_only") else infer(adapter, images, prompt)
    assert runner.main(args, loader=lambda spec: {}, infer=with_invalid, load_qa=runner.v2.load_gold_qa_v2) == 0
    lb = {r["condition"]: r for r in fixtures._csv(out/"leaderboard.csv")}
    # image_only answers "A" (normal) everywhere except the invalid c0 Q1: 3 anomalous missed, 2 normal correct.
    img = lb["image_only"]
    assert (img["q1_tp"], img["q1_fp"], img["q1_tn"], img["q1_fn"], img["q1_invalid_normal_n"]) == ("0", "0", "2", "3", "1")
    assert float(img["q1_accuracy"]) == pytest.approx(2/6) and float(img["q1_balanced_accuracy"]) == pytest.approx(1/3)
    assert float(img["q1_specificity"]) == pytest.approx(2/3) and float(img["q1_recall"]) == 0
    assert float(img["q2_accuracy"]) == pytest.approx(.5) and float(img["q2_defect_balanced_accuracy"]) == pytest.approx(.5)
    ref = lb["normal_reference"]
    assert (ref["q1_tp"], ref["q1_fp"], ref["q1_tn"], ref["q1_fn"]) == ("3", "0", "3", "0")
    assert float(ref["q1_balanced_accuracy"]) == float(ref["q2_defect_balanced_accuracy"]) == 1.0
    by_asset = fixtures._csv(out/"metrics_by_asset.csv")
    assert {(r["condition"], r["asset_category"]) for r in by_asset} == {(c, a) for c in COND_KEYS
                                                                         for a in ("vari-grip", "yoke-suspension")}
    assert all("q1_normal_images" in r and "q1_anomalous_images" in r for r in by_asset)
    macro = {r["condition"]: r for r in fixtures._csv(out/"metrics_asset_macro.csv")}
    assert set(macro) == set(COND_KEYS) and macro["normal_reference"]["assets"] == "2"
    for key in runner.MACRO_KEYS:
        values = [float(r[key]) for r in by_asset if r["condition"] == "image_only"]
        assert float(macro["image_only"][key]) == pytest.approx(sum(values)/len(values), abs=1e-6)


def test_aggregate_rebuilds_the_same_reports_from_saved_steps_without_a_model(tmp_path):
    rows, args, out = setup(tmp_path)
    assert run(rows, args, []) == 0
    steps = {r["id"]: r for r in runner.base.read_jsonl(out/"predictions"/runner.MODEL/"steps.jsonl")}
    snapshot = runner.base.read_jsonl(out/"gold_qa_snapshot.jsonl")
    again = tmp_path/"again"
    statuses = runner.aggregate(snapshot, steps, again, tmp_path/"sampled.csv", n_boot=20)
    assert all(s["status"] == "complete" for s in statuses)
    for name in ("leaderboard.csv", "metrics_by_asset.csv", "metrics_asset_macro.csv", "comparison_vs_sampled.csv",
                 "significance.csv"):
        assert (again/name).read_bytes() == (out/name).read_bytes(), name


def test_default_loader_needs_20gib_and_cached_resume_never_loads(tmp_path, monkeypatch):
    import types

    def fake_torch(gib):
        props = types.SimpleNamespace(total_memory=gib * 1024**3)
        cuda = types.SimpleNamespace(is_available=lambda: gib > 0, get_device_properties=lambda i: props)
        return types.SimpleNamespace(cuda=cuda)
    loaded = []
    monkeypatch.setitem(sys.modules, "insplad_benchmark.hf_adapters",
                        types.SimpleNamespace(load_hf_adapter=lambda spec: loaded.append(spec) or "adapter"))
    for gib in (0, 16):
        monkeypatch.setitem(sys.modules, "torch", fake_torch(gib))
        with pytest.raises(RuntimeError, match=">= 20 GiB"):
            runner.default_loader(runner.base.MODEL_REGISTRY[runner.MODEL])
    monkeypatch.setitem(sys.modules, "torch", fake_torch(24))
    assert runner.default_loader(runner.base.MODEL_REGISTRY[runner.MODEL]) == "adapter" and len(loaded) == 1
    monkeypatch.setitem(sys.modules, "torch", fake_torch(0))
    rows, args, out = setup(tmp_path)
    assert run(rows, args, []) == 0
    calls = []
    assert runner.main(args, infer=fake(rows, calls), load_qa=runner.v2.load_gold_qa_v2) == 0 and calls == []
    assert len(loaded) == 1

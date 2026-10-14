from __future__ import annotations

import hashlib
import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import reaggregate_gold_v2_q23_candidate as aggregate
import verify_gold_revision as revision
import test_gold_v2_q23_candidate as fixtures

runner = fixtures.runner


def test_required_metrics_with_hand_calculated_counts(tmp_path):
    rows, _, args, out = fixtures.setup(tmp_path)
    assert fixtures.run(rows, args, []) == 0
    metrics = {r["policy"]: r for r in fixtures.fixtures._csv(out/"leaderboard.csv")}
    for row in metrics.values():
        assert [int(row[f"q1_{key}"]) for key in ("tp", "fp", "tn", "fn")] == [2, 0, 3, 1]
        assert float(row["q1_balanced_accuracy"]) == pytest.approx(5/6)
        assert float(row["q1_precision"]) == 1
        assert float(row["q1_recall"]) == pytest.approx(2/3)
        assert row["q1_normal_images"] == row["q1_anomalous_images"] == "3"
    assert float(metrics["baseline"]["q2_defect_balanced_accuracy"]) == pytest.approx(2/3)
    assert float(metrics["candidate"]["q2_defect_balanced_accuracy"]) == 1
    macro = {r["policy"]: r for r in fixtures.fixtures._csv(out/"metrics_asset_macro.csv")}
    assert float(macro["baseline"]["q1_balanced_accuracy"]) == .875
    assert float(macro["baseline"]["q2_defect_balanced_accuracy"]) == .75
    required = {"q1_balanced_accuracy", "q1_precision", "q1_recall", "q1_tp", "q1_fp", "q1_tn", "q1_fn", "q2_defect_balanced_accuracy"}
    assert all(required <= set(r) for r in fixtures.fixtures._csv(out/"metrics_by_asset.csv"))
    sig = fixtures.fixtures._csv(out/"significance.csv")
    assert all(r["scope"].startswith("gold6 ") for r in sig if r["metric"].startswith("q3_") or r["metric"] == "q1_q2_q3_presence_joint")


def original_hashes(out):
    return {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*')
            if p.is_file() and "aggregation_v2" not in p.relative_to(out).parts}


def test_normal_only_support_and_legacy_dev_scope_unchanged():
    rows = [r for r in fixtures.fixtures._rows(dev=False) if r["question_type"] != runner.agent.Q3 and r["anomaly_gt"] == "normal"]
    metric = runner.q12_metrics(rows, {r["id"]: r["answer"] for r in rows})
    assert metric["q1_normal_images"] == 3 and metric["q1_anomalous_images"] == 0
    assert not metric["q1_has_both_classes"] and metric["q1_balanced_accuracy"] == 1
    assert metric["q2_defect_balanced_accuracy"] == 1
    q3 = runner.old.q3_rows(rows)
    choice = {r["id"]: r["none_choice"] for r in q3}
    found = {(fixtures.MODEL, p): choice for p in ("baseline", "candidate")}
    joint = {k: {r["core_id"]: True for r in q3} for k in found}
    sig = runner.old.q3_bootstrap(q3, found, joint, [fixtures.MODEL], 3, [("baseline", "candidate")])
    assert sig and all(r["scope"] == "dev presence only, no localization claim" for r in sig)


def test_saved_pr56_reporting_repair_without_model_or_source_overwrite(tmp_path, monkeypatch):
    rows, _, args, out = fixtures.setup(tmp_path)
    assert fixtures.run(rows, args, []) == 0
    # Simulate PR56's report omissions/dev scope; raw calls and identities stay intact.
    leaderboard = fixtures.fixtures._csv(out/"leaderboard.csv")
    for r in leaderboard:
        for key in list(r):
            if key in {"q1_balanced_accuracy", "q1_precision", "q1_recall", "q1_tp", "q1_fp", "q1_tn", "q1_fn", "q2_defect_balanced_accuracy"}:
                del r[key]
    runner.agent.cons.write_csv(out/"leaderboard.csv", leaderboard)
    sig = fixtures.fixtures._csv(out/"significance.csv")
    for r in sig:
        if r["scope"]:
            r["scope"] = "dev presence only, no localization claim"
    runner.agent.cons.write_csv(out/"significance.csv", sig)
    before = original_hashes(out)
    def never_infer(*args, **kwargs):
        raise AssertionError("reaggregation must never load a model")
    monkeypatch.setattr(runner.agent.base, "LazyHFAdapter", never_infer)
    # Neither original source archives nor gold folder are needed for aggregation.
    (tmp_path/"sup.zip").unlink()
    (tmp_path/"unsup.zip").unlink()
    assert aggregate.main(["--run-dir", str(out), "--n-boot", "10"]) == 0
    assert original_hashes(out) == before
    repaired = fixtures.fixtures._csv(out/"aggregation_v2/leaderboard.csv")
    assert float(repaired[0]["q1_balanced_accuracy"]) == pytest.approx(5/6)
    assert "q2_defect_balanced_accuracy" in repaired[0]
    sig = fixtures.fixtures._csv(out/"aggregation_v2/significance.csv")
    assert not any(r["scope"].startswith("dev") for r in sig)
    meta = json.loads((out/"aggregation_v2/aggregation_metadata.json").read_text())
    assert meta["inference_calls"] == 0 and meta["complete"]
    assert meta["source_prediction_identity"] == json.loads((out/"run_metadata.json").read_text())["prediction_identity"]


@pytest.mark.parametrize("kind", ["missing", "error", "invalid"])
def test_incomplete_cached_dependency_cannot_be_repaired_into_success(tmp_path, kind):
    rows, _, args, out = fixtures.setup(tmp_path)
    assert fixtures.run(rows, args, []) == 0
    path = out/"predictions/internvl3_5_8b/steps.jsonl"
    records = runner.agent.read_jsonl(path)
    failed = next(r for r in records if r["id"] == "c0-q2/q2_cues")
    if kind == "missing":
        records.remove(failed)
    else:
        failed.update(error="inference error" if kind == "error" else "", output="bad response", predicted_choice=None)
    runner.save_records(path, records)
    before = original_hashes(out)
    assert aggregate.main(["--run-dir", str(out), "--n-boot", "10"]) == 1
    assert {r["policy"] for r in fixtures.fixtures._csv(out/"aggregation_v2/leaderboard.csv")} == {"baseline", "previous_agent"}
    assert not json.loads((out/"aggregation_v2/run_status.json").read_text())["complete"]
    assert original_hashes(out) == before


@pytest.mark.parametrize("kind", ["identity", "choice", "prompt", "manifest"])
def test_corrupt_cache_rejected_without_new_report(tmp_path, kind):
    rows, _, args, out = fixtures.setup(tmp_path)
    assert fixtures.run(rows, args, []) == 0
    path = out/"predictions/internvl3_5_8b/steps.jsonl"
    records = runner.agent.read_jsonl(path)
    if kind == "manifest":
        manifest = fixtures.fixtures._csv(out/"image_manifest.csv")
        manifest[0]["sha256"] = "changed"
        runner.agent.cons.write_csv(out/"image_manifest.csv", manifest)
    else:
        records[0][{"identity": "identity", "choice": "predicted_choice", "prompt": "prompt"}[kind]] = "changed"
        runner.save_records(path, records)
    with pytest.raises(ValueError):
        aggregate.main(["--run-dir", str(out), "--n-boot", "10"])
    assert not (out/"aggregation_v2").exists()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def test_exact_revision_rejects_moving_main_and_dirty_code(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    git(tmp_path, "config", "user.email", "fixture@example.invalid")
    git(tmp_path, "config", "user.name", "Fixture")
    source = tmp_path/"runner.py"
    source.write_text("old code")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "gold lock")
    pinned = git(tmp_path, "rev-parse", "HEAD")
    assert revision.verify(tmp_path, pinned) == pinned
    with pytest.raises(ValueError, match="full immutable"):
        revision.verify(tmp_path, "main")
    source.write_text("later code")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "advance main")
    with pytest.raises(RuntimeError, match="revision differs"):
        revision.verify(tmp_path, pinned)
    git(tmp_path, "checkout", "--detach", pinned)
    assert revision.verify(tmp_path, pinned) == pinned
    source.write_text("uncommitted prompt change")
    with pytest.raises(RuntimeError, match="modified"):
        revision.verify(tmp_path, pinned)


def test_notebook_checkout_fetches_old_sha_even_after_main_advances(tmp_path):
    repo = tmp_path/"upstream"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "config", "user.name", "Fixture")
    source = repo/"runner.py"
    source.write_text("locked code")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "gold lock")
    locked = git(repo, "rev-parse", "HEAD")
    source.write_text("later main code")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "advance main")
    future = git(repo, "rev-parse", "HEAD")
    for name in ("lvlm_gold_v2_q23_candidate_colab.ipynb", "lvlm_gold_v2_q23_candidate_reaggregate_colab.ipynb"):
        notebook = json.loads((runner.ROOT/"notebooks"/name).read_text(encoding="utf-8"))
        config = {}
        exec("".join(notebook["cells"][1]["source"]), config)
        assert len(config["EXECUTION_COMMIT"]) == 40 and "REPO_REF" not in config
        code = "".join(notebook["cells"][2]["source"])
        checkout = code[code.index("# Fetch exactly"):code.index("del auth, git_env")]
        dest = tmp_path/name.removesuffix(".ipynb")
        commands = []
        def run(cmd, **kwargs):
            cmd = list(map(str, cmd))
            commands.append(cmd)
            if "remote" in cmd and "add" in cmd:
                cmd[-1] = str(repo)
            subprocess.run(cmd, check=True, capture_output=True)
        context = {"REPO_DIR": dest, "REPO": "fixture/repo", "EXECUTION_COMMIT": locked,
                   "subprocess": subprocess, "git_env": {}, "run": run}
        exec(checkout, context)
        assert git(dest, "rev-parse", "HEAD") == locked
        assert any(cmd[-1] == locked and "fetch" in cmd for cmd in commands)
        git(dest, "remote", "set-url", "origin", "https://github.com/fixture/repo.git")
        # Cached exact revision is checked without fetching a later main.
        commands.clear()
        exec(checkout, context)
        assert not commands
        git(dest, "fetch", str(repo), future)
        git(dest, "checkout", "--detach", future)
        with pytest.raises(RuntimeError, match="고정 커밋"):
            exec(checkout, context)


def test_cpu_notebook_aggregation_and_display_from_saved_fixture(tmp_path, monkeypatch):
    rows, _, args, out = fixtures.setup(tmp_path)
    assert fixtures.run(rows, args, []) == 0
    before = original_hashes(out)
    notebook = json.loads((runner.ROOT/"notebooks/lvlm_gold_v2_q23_candidate_reaggregate_colab.ipynb").read_text(encoding="utf-8"))
    import pandas as pd
    for name in ("lvlm_gold_v2_q23_candidate_colab.ipynb", "lvlm_gold_v2_q23_candidate_reaggregate_colab.ipynb"):
        n = json.loads((runner.ROOT/"notebooks"/name).read_text(encoding="utf-8"))
        for i, c in enumerate(n["cells"]):
            if c["cell_type"] == "code":
                compile("".join(c["source"]), f"cell{i}", "exec")
    cpu_source = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert not any(s in cpu_source for s in ("import torch", "build_retrieved_refs", "run_gold_v2_q23_candidate.py", "SUP_ZIP"))
    def never_load(*args, **kwargs):
        raise AssertionError("CPU notebook must not infer")
    monkeypatch.setattr(runner.agent.base, "LazyHFAdapter", never_load)
    commands = []
    def run(cmd, **kwargs):
        commands.append(list(map(str, cmd)))
        return aggregate.main([*map(str, cmd[2:]), "--n-boot", "10"])
    context = {"DRIVE_ROOT": tmp_path, "SAVED_RUN_FOLDERS": [out.name], "REPO_DIR": runner.ROOT,
        "EXECUTION_COMMIT": "792c9eac45eafdedda2013cc31ab969b7df7e041", "verify_gold_revision": lambda *args: None,
        "run": run, "sys": sys, "pd": pd, "json": json, "display": lambda obj: None}
    exec("".join(notebook["cells"][3]["source"]), context)
    exec("".join(notebook["cells"][4]["source"]), context)
    assert len(commands) == 1 and original_hashes(out) == before


def test_cpu_notebook_reporter_error_stops_before_stale_completed_tables(tmp_path):
    rows, _, args, out = fixtures.setup(tmp_path)
    assert fixtures.run(rows, args, []) == 0
    assert aggregate.main(["--run-dir", str(out), "--n-boot", "10"]) == 0
    manifest = fixtures.fixtures._csv(out/"image_manifest.csv")
    manifest[0]["sha256"] = "corrupt after earlier successful reaggregation"
    runner.agent.cons.write_csv(out/"image_manifest.csv", manifest)
    notebook = json.loads((runner.ROOT/"notebooks/lvlm_gold_v2_q23_candidate_reaggregate_colab.ipynb").read_text(encoding="utf-8"))
    boot = ast.parse("".join(notebook["cells"][2]["source"]))
    run_func = next(n for n in boot.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    context = {"DRIVE_ROOT": tmp_path, "SAVED_RUN_FOLDERS": [out.name], "REPO_DIR": runner.ROOT,
        "EXECUTION_COMMIT": "792c9eac45eafdedda2013cc31ab969b7df7e041", "verify_gold_revision": lambda *args: None,
        "sys": sys, "subprocess": subprocess}
    exec(compile(ast.Module(body=[run_func], type_ignores=[]), "actual_cpu_run_helper", "exec"), context)
    with pytest.raises(RuntimeError, match="command failed"):
        exec("".join(notebook["cells"][3]["source"]), context)

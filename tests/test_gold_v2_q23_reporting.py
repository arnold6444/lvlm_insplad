from __future__ import annotations

import hashlib
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

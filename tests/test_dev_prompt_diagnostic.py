from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests")]
import run_dev_prompt_diagnostic as diag
import test_gold_v2_verify_agent as fixtures


def test_prompts_preserve_baseline_and_only_use_the_own_answer():
    rows = fixtures._rows()
    row = rows[0]
    first = "A"
    baseline = diag.dev.build_prompt(row, "normal_ref_retrieved_diffdate")
    assert diag.build_prompt(row, first, "baseline") == baseline
    assert diag.build_prompt(row, first, "current_agent") == diag.agent.build_prompt(row, first, "reask")
    review = diag.build_prompt(row, first, "review_only")
    prior = diag.build_prompt(row, first, "prior_only")
    assert diag.REVIEW in review and "previous answer" not in review
    assert '"A. Normal."' in prior and diag.REVIEW not in prior
    assert review.replace(diag.REVIEW + "\n\n", "") == baseline
    assert prior.replace('The previous answer to this question was "A. Normal.".\n\n', "") == baseline
    assert diag.build_prompt(row, "B", "review_only") == review
    assert diag.build_prompt(row, "B", "prior_only") != prior


def test_subset_is_deterministic_paired_and_rejects_gold():
    rows = fixtures._rows(n=12)
    picked = diag.select_subset(rows, 8, 42)
    assert picked == diag.select_subset(list(reversed(rows)), 8, 42)
    assert len(picked) == 16 and len({r["core_id"] for r in picked}) == 8
    for i in range(0, len(picked), 2):
        assert picked[i]["core_id"] == picked[i + 1]["core_id"]
        assert [r["question_type"] for r in picked[i:i + 2]] == [diag.agent.Q1, diag.agent.Q2]
    with pytest.raises(ValueError, match="gold_v2_dev"):
        diag.select_subset(fixtures._rows(dev=False), 4)
    with pytest.raises(ValueError, match="1..80"):
        diag.select_subset(rows, 81)


def test_run_is_paired_preserves_old_results_retries_errors_and_guards_resume(tmp_path):
    rows = fixtures._rows()
    stage, image_only = fixtures._stage(rows)
    common, qa = fixtures._inputs(tmp_path, rows, stage, image_only)
    sentinel = tmp_path / "gold_v2_verify_agent_dev" / "frozen_config.json"
    sentinel.parent.mkdir()
    sentinel.write_text('{"untouched": true}')
    out = tmp_path / "gold_v2_prompt_diagnostic_test"
    argv = ["--dev-qa", str(qa), "--out-dir", str(out), "--n-images", "4", *common]
    calls = []
    fake = fixtures._fake_infer(rows, calls)
    fail = True

    def once(adapter, images, prompt):
        nonlocal fail
        if fail:
            fail = False
            raise RuntimeError("temporary inference failure")
        return fake(adapter, images, prompt)

    assert diag.main(argv, loader=lambda _: {}, infer=once) == 1
    statuses = json.loads((out / "run_status.json").read_text())
    assert sum(s["row_errors"] for s in statuses) == 1
    lb = list(csv.DictReader((out / "leaderboard.csv").open()))
    assert "baseline" not in {r["condition"] for r in lb}
    assert not (out / "selection.csv").exists() and not (out / "frozen_config.json").exists()
    assert sentinel.read_text() == '{"untouched": true}'
    for call in calls:
        assert call["images"][0] == "other.png" and call["images"][1] == call["core"] + ".png"
    calls.clear()
    assert diag.main(argv, loader=lambda _: {}, infer=fake) == 0
    assert len(calls) == 1
    calls.clear()
    assert diag.main(argv, loader=lambda _: {}, infer=fake) == 0
    assert calls == []
    assert sum(s["reused_predictions"] for s in json.loads((out / "run_status.json").read_text())) == 32
    with pytest.raises(ValueError, match="different inputs"):
        diag.main(argv + ["--seed", "7"], loader=lambda _: {}, infer=fake)
    with pytest.raises(ValueError, match="separate"):
        diag.main(["--dev-qa", str(qa), "--out-dir", str(sentinel.parent), "--n-images", "4", *common],
                  loader=lambda _: {}, infer=fake)


def test_notebook_is_self_contained_and_code_compiles():
    nb = json.loads((ROOT / "notebooks/lvlm_gold_v2_prompt_diagnostic_colab.ipynb").read_text(encoding="utf-8"))
    texts = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for i, text in enumerate(texts):
        compile(text, f"notebook_cell_{i}", "exec")
    namespace = {"REPO_DIR": ROOT}
    source_cell = next(t for t in texts if "RUNNER_SOURCE =" in t)
    # Extract the embedded assignment without writing the runtime file during the test.
    exec(source_cell.split("RUNNER_PATH =")[0], namespace)
    assert namespace["RUNNER_SOURCE"] == (ROOT / "scripts/run_dev_prompt_diagnostic.py").read_text(encoding="utf-8")
    assert not any("publish_results_to_github" in t or "make_frozen_config(" in t for t in texts)

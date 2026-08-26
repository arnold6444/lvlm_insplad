import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_hf_aligned_core_full.py"
SPEC = importlib.util.spec_from_file_location("hf_aligned_core_runner", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


ASSETS = (
    "glass-insulator",
    "lightning-rod-suspension",
    "polymer-insulator-upper-shackle",
    "vari-grip",
    "yoke-suspension",
)


def _one_shot_plus_rows(count: int = 1378) -> list[dict]:
    rows = []
    for index in range(count):
        rows.append(
            {
                "id": f"query-{index:04d}",
                "image": f"{ASSETS[index % len(ASSETS)]}/test/query-{index}.jpg",
                "reference_image": f"{ASSETS[index % len(ASSETS)]}/train/reference-{index}.jpg",
                "asset_category": ASSETS[index % len(ASSETS)],
                "answer": "normal" if index % 2 == 0 else "anomalous",
                "retrieval_method": "dinov2_cosine_top1",
            }
        )
    return rows


def test_model_defaults_preserve_legacy_and_exclude_llava_only_from_plus():
    legacy = RUNNER.resolve_models(RUNNER.LEGACY_EXPERIMENT, None)
    plus = RUNNER.resolve_models(RUNNER.ONE_SHOT_PLUS_EXPERIMENT, None)

    assert legacy == list(RUNNER.MODEL_REGISTRY)
    assert "llava_onevision_7b" in legacy
    assert plus == list(RUNNER.ONE_SHOT_PLUS_MODELS)
    assert plus == ["qwen3_5_4b", "internvl3_5_8b", "minicpm_v_4_5"]


def test_one_shot_plus_rejects_llava_even_when_explicitly_requested():
    with pytest.raises(ValueError, match="supports only Qwen/InternVL/MiniCPM"):
        RUNNER.resolve_models(
            RUNNER.ONE_SHOT_PLUS_EXPERIMENT,
            ["qwen3_5_4b", "llava_onevision_7b"],
        )


def test_one_shot_plus_requires_full_1378_row_qa_before_sampling():
    with pytest.raises(RuntimeError, match="must be 1378"):
        RUNNER.select_one_shot_plus_rows(
            _one_shot_plus_rows(1377),
            full=False,
        )


def test_one_shot_plus_default_smoke_is_balanced_deterministic_and_asset_spanning():
    rows = _one_shot_plus_rows()
    first = RUNNER.select_one_shot_plus_rows(rows, full=False)
    second = RUNNER.select_one_shot_plus_rows(rows, full=False)

    assert len(first) == 25
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert {row["asset_category"] for row in first} == set(ASSETS)
    assert sum(row["answer"] == "normal" for row in first) == 12
    assert sum(row["answer"] == "anomalous" for row in first) == 13


@pytest.mark.parametrize("smoke_size", [0, 19, 51])
def test_one_shot_plus_smoke_size_is_limited_to_20_through_50(smoke_size):
    with pytest.raises(ValueError, match="between 20 and 50"):
        RUNNER.select_one_shot_plus_rows(
            _one_shot_plus_rows(),
            full=False,
            smoke_size=smoke_size,
        )


def test_one_shot_plus_full_keeps_all_rows_and_validates_retrieval_metadata():
    rows = _one_shot_plus_rows()
    selected = RUNNER.select_one_shot_plus_rows(rows, full=True)
    assert len(selected) == 1378
    assert [row["id"] for row in selected] == [row["id"] for row in rows]

    rows[0]["retrieval_method"] = ""
    with pytest.raises(RuntimeError, match="retrieval_method"):
        RUNNER.select_one_shot_plus_rows(rows, full=True)


def test_one_shot_plus_reuses_fixed_prompt_and_reference_then_query_order():
    reference = Path("reference.jpg")
    query = Path("query.jpg")
    row = {"reference_image": "ref", "image": "query"}
    cache = {"ref": reference, "query": query}

    plus_paths, plus_prompt = RUNNER.task_inputs(
        RUNNER.ONE_SHOT_PLUS_EXPERIMENT,
        row,
        cache,
    )
    fixed_paths, fixed_prompt = RUNNER.task_inputs("one_shot", row, cache)

    assert plus_paths == [reference, query]
    assert plus_paths == fixed_paths
    assert plus_prompt == RUNNER.ONE_SHOT_PROMPT
    assert plus_prompt == fixed_prompt


def test_one_shot_plus_resume_requires_matching_reference_and_retrieval_method():
    gt = {
        "reference_image": "asset/train/ref.jpg",
        "retrieval_method": "dinov2_cosine_top1",
    }
    prediction = {
        "output": '{"anomaly":"normal"}',
        "error": None,
        **gt,
    }

    assert RUNNER.prediction_is_reusable(
        RUNNER.ONE_SHOT_PLUS_EXPERIMENT,
        prediction,
        gt,
    )
    assert not RUNNER.prediction_is_reusable(
        RUNNER.ONE_SHOT_PLUS_EXPERIMENT,
        {**prediction, "reference_image": "asset/train/other.jpg"},
        gt,
    )
    assert not RUNNER.prediction_is_reusable(
        RUNNER.ONE_SHOT_PLUS_EXPERIMENT,
        {**prediction, "retrieval_method": "other_method"},
        gt,
    )
    assert RUNNER.prediction_is_reusable(
        "one_shot",
        {"output": prediction["output"], "error": None},
        gt,
    )


def test_plus_output_contract_is_written_early_and_is_immutable_on_resume(tmp_path):
    rows = _one_shot_plus_rows(25)
    out_dir = tmp_path / "results"

    expected = RUNNER.ensure_one_shot_plus_output_contract(
        out_dir,
        rows,
        full=False,
        smoke_size=25,
    )
    contract_path = out_dir / RUNNER.RUN_CONTRACT_FILENAME
    original = contract_path.read_bytes()

    assert json.loads(original) == expected
    assert expected["experiment"] == RUNNER.ONE_SHOT_PLUS_EXPERIMENT
    assert expected["run_mode"] == "smoke"
    assert expected["smoke_size"] == 25
    assert expected["selected_qa_count"] == 25

    resumed = RUNNER.ensure_one_shot_plus_output_contract(
        out_dir,
        rows,
        full=False,
        smoke_size=25,
    )
    assert resumed == expected
    assert contract_path.read_bytes() == original


@pytest.mark.parametrize(
    ("full", "smoke_size", "mutate_row", "error"),
    [
        (True, 25, False, "incompatible run contract"),
        (False, 24, False, "incompatible run contract"),
        (False, 25, True, "incompatible run contract"),
    ],
)
def test_plus_output_contract_rejects_mode_size_or_selected_qa_changes_before_writes(
    tmp_path,
    full,
    smoke_size,
    mutate_row,
    error,
):
    rows = _one_shot_plus_rows(25)
    out_dir = tmp_path / "results"
    RUNNER.ensure_one_shot_plus_output_contract(
        out_dir,
        rows,
        full=False,
        smoke_size=25,
    )
    contract_path = out_dir / RUNNER.RUN_CONTRACT_FILENAME
    before = contract_path.read_bytes()
    attempted_rows = [dict(row) for row in rows]
    if mutate_row:
        attempted_rows[0]["reference_image"] = "other/train/reference.jpg"

    with pytest.raises(RuntimeError, match=error):
        RUNNER.ensure_one_shot_plus_output_contract(
            out_dir,
            attempted_rows,
            full=full,
            smoke_size=smoke_size,
        )

    assert contract_path.read_bytes() == before
    assert set(out_dir.iterdir()) == {contract_path}


def test_full_plus_snapshot_cannot_be_truncated_by_later_smoke_run(tmp_path):
    full_rows = _one_shot_plus_rows(30)
    smoke_rows = full_rows[:25]
    out_dir = tmp_path / "results"
    RUNNER.ensure_one_shot_plus_output_contract(
        out_dir,
        full_rows,
        full=True,
        smoke_size=25,
    )
    snapshot = out_dir / "run_qa" / "one_shot_plus.jsonl"
    RUNNER.ensure_one_shot_plus_snapshot(snapshot, full_rows)
    before = snapshot.read_bytes()

    with pytest.raises(RuntimeError, match="incompatible run contract"):
        RUNNER.ensure_one_shot_plus_output_contract(
            out_dir,
            smoke_rows,
            full=False,
            smoke_size=25,
        )

    assert snapshot.read_bytes() == before
    assert len(RUNNER.read_jsonl(snapshot)) == 30


def test_plus_output_contract_rejects_ambiguous_or_legacy_nonempty_dirs(tmp_path):
    rows = _one_shot_plus_rows(25)
    ambiguous = tmp_path / "ambiguous"
    ambiguous.mkdir()
    sentinel = ambiguous / "leaderboard.csv"
    sentinel.write_text("legacy-data", encoding="utf-8")

    with pytest.raises(RuntimeError, match="refusing nonempty"):
        RUNNER.ensure_one_shot_plus_output_contract(
            ambiguous,
            rows,
            full=False,
            smoke_size=25,
        )
    assert sentinel.read_text(encoding="utf-8") == "legacy-data"
    assert not (ambiguous / RUNNER.RUN_CONTRACT_FILENAME).exists()

    legacy = tmp_path / "legacy"
    legacy_prediction = legacy / "qwen3_5_4b" / "zero_shot.jsonl"
    legacy_prediction.parent.mkdir(parents=True)
    legacy_prediction.write_text("baseline-data", encoding="utf-8")
    with pytest.raises(RuntimeError, match="legacy output artifacts"):
        RUNNER.ensure_one_shot_plus_output_contract(
            legacy,
            rows,
            full=False,
            smoke_size=25,
        )
    assert legacy_prediction.read_text(encoding="utf-8") == "baseline-data"
    assert not (legacy / RUNNER.RUN_CONTRACT_FILENAME).exists()


@pytest.mark.parametrize("experiment", [RUNNER.LEGACY_EXPERIMENT, "other_experiment"])
def test_plus_output_contract_rejects_other_experiment_metadata_before_writes(
    tmp_path,
    experiment,
):
    rows = _one_shot_plus_rows(25)
    out_dir = tmp_path / experiment
    snapshot = out_dir / "run_qa" / "one_shot_plus.jsonl"
    RUNNER.write_jsonl(snapshot, rows)
    metadata_path = out_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "experiment": experiment,
                "run_mode": "smoke",
                "selected_qa_count": 25,
            }
        ),
        encoding="utf-8",
    )
    before_metadata = metadata_path.read_bytes()
    before_snapshot = snapshot.read_bytes()

    with pytest.raises(RuntimeError, match="incompatible run metadata"):
        RUNNER.ensure_one_shot_plus_output_contract(
            out_dir,
            rows,
            full=False,
            smoke_size=25,
        )

    assert metadata_path.read_bytes() == before_metadata
    assert snapshot.read_bytes() == before_snapshot
    assert not (out_dir / RUNNER.RUN_CONTRACT_FILENAME).exists()


def test_compatible_previous_plus_metadata_and_snapshot_are_migrated(tmp_path):
    rows = _one_shot_plus_rows(25)
    out_dir = tmp_path / "previous-plus"
    snapshot = out_dir / "run_qa" / "one_shot_plus.jsonl"
    RUNNER.write_jsonl(snapshot, rows)
    metadata = {
        "experiment": RUNNER.ONE_SHOT_PLUS_EXPERIMENT,
        "run_mode": "smoke",
        "selected_qa_count": 25,
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    expected = RUNNER.ensure_one_shot_plus_output_contract(
        out_dir,
        rows,
        full=False,
        smoke_size=25,
    )

    assert json.loads(
        (out_dir / RUNNER.RUN_CONTRACT_FILENAME).read_text(encoding="utf-8")
    ) == expected
    assert RUNNER.read_jsonl(snapshot) == rows


def test_existing_plus_snapshot_identity_is_checked_before_resume_writes(tmp_path):
    rows = _one_shot_plus_rows(25)
    out_dir = tmp_path / "results"
    RUNNER.ensure_one_shot_plus_output_contract(
        out_dir,
        rows,
        full=False,
        smoke_size=25,
    )
    snapshot = out_dir / "run_qa" / "one_shot_plus.jsonl"
    RUNNER.ensure_one_shot_plus_snapshot(snapshot, rows)
    changed = [dict(row) for row in rows]
    changed[0]["reference_image"] = "changed/train/reference.jpg"
    RUNNER.write_jsonl(snapshot, changed)
    before = snapshot.read_bytes()

    with pytest.raises(RuntimeError, match="different selected QA identity"):
        RUNNER.ensure_one_shot_plus_output_contract(
            out_dir,
            rows,
            full=False,
            smoke_size=25,
        )
    assert snapshot.read_bytes() == before


def test_runner_wires_distinct_plus_snapshot_prediction_and_evaluation_regime():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'snapshot_path = args.out_dir / "run_qa" / "one_shot_plus.jsonl"' in source
    assert 'pred_path = args.out_dir / model_key / f"{task}.jsonl"' in source
    assert '"retrieval_method": gt["retrieval_method"]' in source
    assert 'ONE_SHOT_PLUS_EXPERIMENT,' in source
    assert '"one_shot_plus_bacc"' in source
    assert '"one_shot_plus_asset_macro_bacc"' in source
    assert 'for key in ONE_SHOT_PLUS_MODELS' in source
    assert 'Aligned Core QA must be 1378/1378/1378' in source
    assert source.index("def main()") < source.index("import pandas as pd")

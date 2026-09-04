from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/lvlm_vision_expert_smoke_resumable_colab.ipynb")


def _load_notebook(root: Path) -> dict:
    return json.loads((root / NOTEBOOK).read_text(encoding="utf-8"))


def test_resumable_notebook_compiles_and_persists_every_stage() -> None:
    root = Path(__file__).resolve().parents[1]
    data = _load_notebook(root)

    assert data["nbformat"] == 4
    assert data["metadata"]["accelerator"] == "GPU"

    sources: list[str] = []
    for index, cell in enumerate(data["cells"]):
        source = "".join(cell.get("source", []))
        sources.append(source)
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
            compile(source, f"resumable_vision_smoke_cell_{index}", "exec")

    joined = "\n".join(sources)
    for expected in (
        'RUN_FOLDER = "vision_expert_smoke_v1"',
        'LVLM_CHECKPOINT_EVERY = 1',
        'FORCE_STAGES = set()',
        'STATE_PATH = RUN_ROOT / "run_state.json"',
        'EVENTS_PATH = LOG_ROOT / "events.jsonl"',
        'RUN_LOG_PATH = LOG_ROOT / "run.log"',
        'def atomic_write_text(path, text):',
        'def run_stage(name, action, validator):',
        '[RESUME]',
        'PERSISTENT_WORKSPACE_ROOT = RUN_ROOT / "workspace"',
        'PatchCore',
        'Dinomaly',
        'AnomalyCLIP',
        'LVLM_CHECKPOINT_EVERY',
        'run_summary_resumable.json',
        'screening_comparison.csv',
    ):
        assert expected in joined


def test_review_bundle_is_saved_to_drive_and_ready_for_chat_upload() -> None:
    root = Path(__file__).resolve().parents[1]
    data = _load_notebook(root)
    joined = "\n".join("".join(cell.get("source", [])) for cell in data["cells"])

    for expected in (
        'AUTO_DOWNLOAD_REVIEW_ZIP = True',
        'SAVE_FULL_RESULT_ZIP_TO_DRIVE = False',
        '# 11) 결과 확인용 ZIP 생성',
        'review_base.glob(f"{review_run_folder}*")',
        'screening_comparison.csv',
        'smoke_winner.json',
        'run_state.json',
        'logs/run_log_tail.txt',
        'review_manifest.json',
        'shutil.make_archive(',
        'temporary.replace(destination)',
        'vision_expert_smoke_REVIEW_LATEST.zip',
        'files.download(str(local_zip))',
    ):
        assert expected in joined

    export_cells = [
        "".join(cell.get("source", []))
        for cell in data["cells"]
        if "# 11) 결과 확인용 ZIP 생성" in "".join(cell.get("source", []))
    ]
    assert len(export_cells) == 1
    assert 'globals().get("RUN_ROOT")' in export_cells[0]
    assert 'globals().get("RUN_FOLDER", "vision_expert_smoke_v1")' in export_cells[0]


def test_stage_table_handles_empty_none_multiline_and_malformed_values() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "notebooks" / "lvlm_vision_expert_smoke_resumable_colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "def stage_table()" in "".join(cell.get("source", []))
    )
    namespace = {
        "STATE": {
            "stages": {
                "complete": {"status": "complete"},
                "empty": {"status": "failed", "error": ""},
                "none": {"status": "failed", "error": None},
                "multi": {"status": "failed", "error": "first line\nsecond line"},
                "malformed": "broken-state-value",
            }
        },
        "STATE_PATH": Path("state.json"),
        "RUN_LOG_PATH": Path("run.log"),
        "display": lambda _value: None,
    }
    exec(compile(source, "stage-table-cell", "exec"), namespace)
    table = namespace["stage_table"]().set_index("stage")
    assert table.loc["complete", "error"] == ""
    assert table.loc["empty", "error"] == ""
    assert table.loc["none", "error"] == ""
    assert table.loc["multi", "error"] == "first line"
    assert table.loc["malformed", "status"] == "invalid_state"
    assert table.loc["malformed", "error"] == "broken-state-value"

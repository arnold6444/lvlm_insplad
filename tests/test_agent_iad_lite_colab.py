import json
from pathlib import Path


def test_agent_iad_lite_notebook_compiles_and_uses_repo_cwd() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/lvlm_agent_iad_lite_smoke_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile(cell.get("source", ""), f"cell-{index}", "exec")

    setup = next(
        cell["source"]
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "tests/test_agent_iad_lite.py" in cell.get("source", "")
    )
    assert "def run_cmd(cmd, name, *, cwd=None):" in setup
    assert '"focused_tests", cwd=REPO_DIR' in setup
    assert 'REPO_REF = "main"' in notebook["cells"][1]["source"]


def test_agent_notebook_keeps_smoke_holdout_warning() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "notebooks/lvlm_agent_iad_lite_smoke_colab.ipynb").read_text(encoding="utf-8")
    )
    markdown = "\n".join(
        str(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    assert "train-normal calibration" in markdown
    assert "AgentIAD-lite" in markdown

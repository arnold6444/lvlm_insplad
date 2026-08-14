import json
from pathlib import Path


def test_luxia_probe_notebook_structure():
    root = Path(__file__).resolve().parents[1]
    notebook = root / "notebooks" / "luxia_vision_probe_colab.ipynb"
    data = json.loads(notebook.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    joined = []
    for index, cell in enumerate(data["cells"]):
        source = cell.get("source", "")
        joined.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"luxia_probe_cell_{index}", "exec")
    joined = "\n".join(joined)
    for expected in (
        "LUXIA_API_KEY",
        "GITHUB_TOKEN",
        "unsupervised_anomaly_detection.zip",
        "samples/anomaly_zero_shot_20.jsonl",
        "r.get('answer') == 'normal'",
        "r.get('answer') == 'anomalous'",
        "SRC_DIR = REPO_DIR / 'src'",
        "sys.path.insert(0, str(SRC_DIR))",
        "import insplad_benchmark",
        "call_luxia_document_ai",
        "luxia_probe.json",
        "document_ai_single_image",
        "comparable_to_prompted_vlm",
    ):
        assert expected in joined

    assert "r.get('answer', {}).get('anomaly')" not in joined

import json
from pathlib import Path


def test_one_shot_plus_colab_compiles_and_keeps_full_run_guarded():
    root = Path(__file__).resolve().parents[1]
    notebook = root / "notebooks" / "lvlm_one_shot_plus_colab.ipynb"
    data = json.loads(notebook.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    assert data["metadata"]["accelerator"] == "GPU"

    sources = []
    for index, cell in enumerate(data["cells"]):
        source = cell.get("source", "")
        sources.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"one_shot_plus_colab_cell_{index}", "exec")
    joined = "\n".join(sources)

    for expected in (
        "facebook/dinov2-base",
        "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        "reference_embeddings.npz",
        "one_shot_plus.jsonl",
        "zero_shot.jsonl",
        "structured.jsonl",
        "one_shot_plus_references.csv",
        "scripts/build_one_shot_plus.py",
        "scripts/run_hf_aligned_core_full.py",
        "--experiment', 'one_shot_plus",
        "Qwen3.5-4B / InternVL3.5-8B / MiniCPM-V-4.5",
        "qwen3_5_4b",
        "internvl3_5_8b",
        "minicpm_v_4_5",
        "hf_one_shot_plus_v1",
        "SMOKE_SIZE = 25",
        "RUN_FULL = False",
        "if RUN_FULL:",
        "full 1,378 LVLM inference was not run",
    ):
        assert expected in joined

    assert "llava_onevision_7b" not in joined
    assert "20 <= SMOKE_SIZE <= 50" in joined
    assert "assert len(retrievals) == 1378" in joined

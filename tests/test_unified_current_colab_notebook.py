import json
from pathlib import Path


def _joined_source() -> str:
    root = Path(__file__).resolve().parents[1]
    notebook = root / "notebooks" / "lvlm_current_all_colab.ipynb"
    data = json.loads(notebook.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    assert data["metadata"]["accelerator"] == "GPU"

    sources = []
    for index, cell in enumerate(data["cells"]):
        source = cell.get("source", "")
        sources.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"lvlm_current_all_cell_{index}", "exec")
    return "\n".join(sources)


def test_unified_current_notebook_runs_only_active_hf_models_plus_luxia():
    joined = _joined_source()
    for expected in (
        "Qwen/Qwen3.5-4B",
        "OpenGVLab/InternVL3_5-8B",
        "openbmb/MiniCPM-V-4_5",
        "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "scripts/run_luxia_smoke.py",
        "LUXIA_API_KEY",
        "combined_leaderboard.csv",
        "combined_run_manifest.json",
        "current4_plus_luxia",
        "files.download(str(zip_path))",
    ):
        assert expected in joined

    for removed in (
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
        "OpenGVLab/InternVL2_5-8B",
    ):
        assert removed not in joined


def test_unified_notebook_reuses_engine_without_legacy_registry_or_early_download():
    joined = _joined_source()
    assert "source.startswith('# 4) Hugging Face model registry')" in joined
    assert "source.startswith('# 9) Package results and download ZIP')" in joined
    assert "HF_MODELS_TO_RUN = list(MODEL_REGISTRY)" in joined
    assert "hf_current4" in joined
    assert "luxia_pipeline" in joined

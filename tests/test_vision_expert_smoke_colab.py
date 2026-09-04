from __future__ import annotations

import json
from pathlib import Path


def test_vision_expert_smoke_notebook_compiles_and_locks_screening_scope() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = root / "notebooks" / "lvlm_vision_expert_smoke_colab.ipynb"
    data = json.loads(notebook.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    assert data["metadata"]["accelerator"] == "GPU"

    sources = []
    for index, cell in enumerate(data["cells"]):
        source = cell.get("source", "")
        sources.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"vision_expert_smoke_cell_{index}", "exec")
    joined = "\n".join(sources)

    for expected in (
        "SMOKE_SIZE = 50",
        "SRC_DIR = str(REPO_DIR / 'src')",
        "REPO_REF = 'main'",
        "predictions = engine = model = datamodule = None",
        "text_features = prompt_learner = model = None",
        "MODELS_TO_RUN = ['qwen3_5_4b', 'internvl3_5_8b']",
        "EXPERTS_TO_RUN = ['patchcore', 'dinomaly', 'anomalyclip']",
        "anomalib==2.6.0",
        "DINOMALY_DECODER_DEPTH = 8",
        "facebook/dinov2-base",
        "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        "scripts/build_vision_expert_smoke_data.py",
        "tests/test_build_vision_expert_smoke_data.py",
        "stdout=subprocess.PIPE",
        "complete child-process output is shown above",
        "scripts/run_hf_vision_expert_smoke.py",
        "roi_plus_reference",
        "3911738c0867544f545a076ad78f3f11d9ecbfdf",
        "checkpoints/9_12_4_multiscale/epoch_15.pth",
        "assert len(selected_roi_rows) == SMOKE_SIZE * len(EXPERTS_TO_RUN)",
        "1,328장 holdout",
        "RAG와 AgentIAD-lite는 이번 notebook 범위가 아닙니다",
    ):
        assert expected in joined

    assert "feat/vision-expert-smoke-colab" not in joined
    assert "minicpm_v_4_5" not in joined
    assert "llava_onevision_7b" not in joined
    assert "RUN_FULL = True" not in joined
    assert "torch==2.0.0" not in joined


def test_checkout_cell_is_safe_when_rerun_from_deleted_repo_directory() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = root / "notebooks" / "lvlm_vision_expert_smoke_colab.ipynb"
    data = json.loads(notebook.read_text(encoding="utf-8"))
    checkout_source = next(
        cell["source"]
        for cell in data["cells"]
        if cell.get("cell_type") == "code"
        and "REPO_DIR = Path('/content/lvlm_insplad')" in cell.get("source", "")
    )
    chdir_index = checkout_source.index("os.chdir('/content')")
    repo_index = checkout_source.index("REPO_DIR = Path('/content/lvlm_insplad')")
    delete_index = checkout_source.index("shutil.rmtree(REPO_DIR)")
    assert chdir_index < repo_index < delete_index


def test_anomalyclip_runtime_compatibility_is_explicit() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = root / "notebooks" / "lvlm_vision_expert_smoke_colab.ipynb"
    data = json.loads(notebook.read_text(encoding="utf-8"))
    joined = "\n".join(cell.get("source", "") for cell in data["cells"])

    assert "tests/test_anomalyclip_compat.py" in joined
    assert "patch_official_anomalyclip_checkout" in joined
    assert "ANOMALYCLIP_MODEL_CACHE = Path('/content/anomalyclip_model_cache')" in joined
    assert "download_root=str(ANOMALYCLIP_MODEL_CACHE)" in joined
    assert "compat_patch_schema_version" in joined
    assert "compat_source_sha256" in joined


def test_anomalib_engines_use_barebones_without_checkpoint_conflict() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = root / "notebooks" / "lvlm_vision_expert_smoke_colab.ipynb"
    data = json.loads(notebook.read_text(encoding="utf-8"))
    joined = "\n".join(cell.get("source", "") for cell in data["cells"])
    assert joined.count("barebones=True") == 2
    assert "enable_checkpointing=False" not in joined


def test_anomalib_engines_disable_missing_validation_loop() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = root / "notebooks" / "lvlm_vision_expert_smoke_colab.ipynb"
    data = json.loads(notebook.read_text(encoding="utf-8"))
    joined = "\n".join(cell.get("source", "") for cell in data["cells"])
    assert joined.count("barebones=True") == 2
    assert joined.count("limit_val_batches=0") == 2

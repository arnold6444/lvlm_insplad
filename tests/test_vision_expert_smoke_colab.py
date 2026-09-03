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
        "MODELS_TO_RUN = ['qwen3_5_4b', 'internvl3_5_8b']",
        "EXPERTS_TO_RUN = ['patchcore', 'dinomaly', 'anomalyclip']",
        "anomalib==2.6.0",
        "DINOMALY_DECODER_DEPTH = 8",
        "facebook/dinov2-base",
        "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        "scripts/build_vision_expert_smoke_data.py",
        "scripts/run_hf_vision_expert_smoke.py",
        "roi_plus_reference",
        "3911738c0867544f545a076ad78f3f11d9ecbfdf",
        "checkpoints/9_12_4_multiscale/epoch_15.pth",
        "assert len(selected_roi_rows) == SMOKE_SIZE * len(EXPERTS_TO_RUN)",
        "1,328장 holdout",
        "RAG와 AgentIAD-lite는 이번 notebook 범위가 아닙니다",
    ):
        assert expected in joined

    assert "minicpm_v_4_5" not in joined
    assert "llava_onevision_7b" not in joined
    assert "RUN_FULL = True" not in joined
    assert "torch==2.0.0" not in joined

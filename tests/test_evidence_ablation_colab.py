from __future__ import annotations

import json
from pathlib import Path


def test_evidence_ablation_notebook_compiles_and_locks_scope() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/lvlm_evidence_ablation_smoke_colab.ipynb"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["metadata"]["accelerator"] == "GPU"
    joined = []
    for index, cell in enumerate(data["cells"]):
        source = cell.get("source", "")
        joined.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"evidence_ablation_cell_{index}", "exec")
    text = "\n".join(joined)
    for expected in (
        'PREVIOUS_RUN_FOLDER = "vision_expert_smoke_v1"',
        'EXPERTS_TO_RUN = ["patchcore", "dinomaly", "anomalyclip", "reconpatch_lite"]',
        'PRIMARY_CONDITIONS = ["score", "bbox", "heatmap"]',
        '"score": "reference_score"',
        '"bbox": "reference_bbox"',
        '"heatmap": "reference_heatmap"',
        "scripts/run_reconpatch_lite_smoke.py",
        "scripts/build_evidence_ablation.py",
        "scripts/run_hf_evidence_ablation.py",
        "best_fixed_by_model.json",
        "evidence_ablation_REVIEW_LATEST.zip",
        "1,328장 holdout",
    ):
        assert expected in text
    assert '"roi"' not in text
    assert "ROI crop`은 다시 추론하지 않고" in text

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
FULL = NB / "lvlm_gold_v2_vision_expert_full_final_colab.ipynb"
RESUME = NB / "lvlm_gold_v2_vision_expert_full_resume_colab.ipynb"
NEXT = NB / "lvlm_gold_v2_reference_expert_colab.ipynb"


def _code(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = [c["source"] if isinstance(c["source"], str) else "".join(c["source"])
             for c in nb["cells"] if c["cell_type"] == "code"]
    for i, src in enumerate(cells):
        compile(src, f"{path.name}:{i}", "exec")
    return cells


def test_notebooks_compile_and_checkpoint_every_20():
    for path in (FULL, RESUME, NEXT):
        assert "CHECKPOINT_EVERY = 20" in "\n".join(_code(path))


def test_merge_cell_reuses_existing_outputs_in_full_and_resume():
    for path in (FULL, RESUME):
        merge = next(c for c in _code(path) if c.startswith("# 4 expert manifest"))
        assert "merged_manifest_ready()" in merge
        assert "out.is_file() and out.stat().st_size > 0" in merge
        assert "tmp.replace(out)" in merge


def test_resume_notebook_skips_expert_stages():
    code = "\n".join(_code(RESUME))
    for skipped in ("anomalib==", "run_patchcore_cohort.py", "run_dinomaly_cohort.py",
                    "run_anomalyclip_cohort.py", "run_reconpatch_lite_smoke.py",
                    "build_vision_expert_full_data.py"):
        assert skipped not in code
    assert "run_gold_v2_vision_expert_full.py" in code
    assert "publish_results_to_github.py" in code


def test_next_stage_notebook_runs_reference_expert_after_full_coverage():
    code = "\n".join(_code(NEXT))
    assert "run_gold_v2_reference_expert.py" in code
    assert "Vision Expert full 80조건이 아직 다 끝나지 않았습니다" in code
    assert '"--prefix", "gold_v2_reference_expert"' in code

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_hf_vision_expert_smoke.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_hf_vision_expert_smoke", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_compiles_and_limits_screening_models() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    compile(source, str(SCRIPT), "exec")
    assert set(load_runner().MODEL_REGISTRY) == {"qwen3_5_4b", "internvl3_5_8b"}
    assert "minicpm_v_4_5" not in source
    assert "llava_onevision_7b" not in source


def test_evidence_input_order_and_non_forcing_prompt(tmp_path: Path) -> None:
    runner = load_runner()
    reference = tmp_path / "reference.png"
    query = tmp_path / "query.png"
    roi = tmp_path / "roi.png"

    images, prompt = runner.task_inputs(
        condition="roi",
        query_path=query,
        roi_path=roi,
        reference_path=None,
        expert="patchcore",
    )
    assert images == [query, roi]
    assert "can be normal" in prompt
    assert "classify only the TARGET" in prompt

    images, prompt = runner.task_inputs(
        condition="roi_plus_reference",
        query_path=query,
        roi_path=roi,
        reference_path=reference,
        expert="dinomaly",
    )
    assert images == [reference, query, roi]
    assert "Image 1 is a NORMAL reference" in prompt
    assert "Image 3 is a candidate close-up" in prompt


def test_resume_identity_rejects_changed_roi() -> None:
    runner = load_runner()
    row = {
        "output": '{"anomaly":"normal"}',
        "error": "",
        "condition": "roi",
        "expert": "patchcore",
        "query_image": "q.png",
        "reference_image": "",
        "roi_path": "/tmp/a.png",
    }
    expected = {
        "condition": "roi",
        "expert": "patchcore",
        "query_image": "q.png",
        "reference_image": "",
        "roi_path": "/tmp/a.png",
    }
    assert runner.reusable_prediction(row, expected)
    expected["roi_path"] = "/tmp/b.png"
    assert not runner.reusable_prediction(row, expected)

import json
from pathlib import Path


def _joined_source():
    root = Path(__file__).resolve().parents[1]
    notebook = root / 'notebooks' / 'lvlm_current_models_colab.ipynb'
    data = json.loads(notebook.read_text(encoding='utf-8'))
    assert data['nbformat'] == 4
    assert data['metadata']['accelerator'] == 'GPU'
    joined = []
    for index, cell in enumerate(data['cells']):
        source = cell.get('source', '')
        joined.append(source)
        if cell.get('cell_type') == 'code':
            compile(source, f'current_models_cell_{index}', 'exec')
    return '\n'.join(joined)


def test_current_runner_keeps_only_current_family_models():
    joined = _joined_source()
    for expected in (
        "MODEL_KEY = 'qwen3_5_4b'",
        'Qwen/Qwen3.5-4B',
        'OpenGVLab/InternVL3_5-8B',
        'openbmb/MiniCPM-V-4_5',
        'llava-hf/llava-onevision-qwen2-7b-ov-hf',
        "HF_MODELS_TO_RUN = [{MODEL_KEY!r}]",
        "lvlm_baselines_colab.ipynb",
    ):
        assert expected in joined

    for removed in (
        'Qwen/Qwen2.5-VL-7B-Instruct',
        'Qwen/Qwen3-VL-8B-Instruct',
        'OpenGVLab/InternVL2_5-8B',
    ):
        assert removed not in joined


def test_current_runner_executes_one_selected_model_only():
    joined = _joined_source()
    assert 'ALLOWED_MODEL_KEYS' in joined
    assert "if MODEL_KEY not in ALLOWED_MODEL_KEYS" in joined
    assert "HF_MODELS_TO_RUN = [{MODEL_KEY!r}]" in joined
    assert "source.startswith('# 4) Hugging Face model registry')" in joined

import json
from pathlib import Path


def test_hybrid_colab_notebook_is_valid_and_wired_to_benchmark():
    root = Path(__file__).resolve().parents[1]
    notebook = root / 'notebooks' / 'lvlm_baselines_colab.ipynb'
    data = json.loads(notebook.read_text(encoding='utf-8'))

    assert data['nbformat'] == 4
    assert data['metadata']['accelerator'] == 'GPU'

    sources = []
    for index, cell in enumerate(data['cells']):
        source = cell.get('source', '')
        sources.append(source)
        if cell.get('cell_type') == 'code':
            compile(source, f'lvlm_baselines_cell_{index}', 'exec')

    joined = '\n'.join(sources)
    for expected in (
        "BRANCH = 'main'",
        'GITHUB_TOKEN',
        'OPENAI_API_KEY',
        'GEMINI_API_KEY',
        'gpt-5.6-sol',
        'gemini-3.6-flash',
        "'backend': 'api'",
        "'backend': 'hf'",
        'API_MODELS_TO_RUN',
        'HF_MODELS_TO_RUN',
        'Qwen/Qwen3-VL-8B-Instruct',
        'OpenGVLab/InternVL3_5-8B',
        'openbmb/MiniCPM-V-4_5',
        'llava-hf/llava-onevision-qwen2-7b-ov-hf',
        'samples/anomaly_zero_shot_20.jsonl',
        'samples/anomaly_one_shot_20.jsonl',
        'samples/defect_20.jsonl',
        'scripts/evaluate.py',
        'leaderboard.csv',
        'prediction_review.csv',
        'run_metadata.json',
        'skipped_insufficient_gpu',
        'torch.cuda.empty_cache',
    ):
        assert expected in joined


def test_hybrid_notebook_keeps_api_and_hf_execution_separate():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / 'notebooks' / 'lvlm_baselines_colab.ipynb').read_text(encoding='utf-8'))
    joined = '\n'.join(cell.get('source', '') for cell in data['cells'])

    assert 'API_READY' in joined
    assert "'openai': bool(OPENAI_API_KEY)" in joined
    assert "'gemini': bool(GEMINI_API_KEY)" in joined
    assert 'HF_READY = cuda_available and (vram_gb >= 20 or ALLOW_HF_BELOW_20GB)' in joined
    assert 'Use an L4/A100-class runtime rather than silently changing the paper baseline with 4-bit quantization.' in joined

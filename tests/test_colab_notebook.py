import json
from pathlib import Path


def test_multimodel_colab_notebook_is_valid_and_wired_to_benchmark():
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
        'MODELS_TO_RUN',
        'torch.cuda.empty_cache',
    ):
        assert expected in joined

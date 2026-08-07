import json
from pathlib import Path


def test_qwen3_vl_colab_notebook_is_valid_and_wired_to_benchmark():
    root = Path(__file__).resolve().parents[1]
    notebook = root / 'notebooks' / 'qwen3_vl_baseline.ipynb'
    data = json.loads(notebook.read_text(encoding='utf-8'))

    assert data['nbformat'] == 4
    assert data['metadata']['accelerator'] == 'GPU'

    sources = []
    for index, cell in enumerate(data['cells']):
        source = cell.get('source', '')
        sources.append(source)
        if cell.get('cell_type') == 'code':
            compile(source, f'qwen3_vl_baseline_cell_{index}', 'exec')

    joined = '\n'.join(sources)
    for expected in (
        'Qwen/Qwen3-VL-8B-Instruct',
        'Qwen/Qwen3-VL-4B-Instruct',
        'samples/anomaly_zero_shot_20.jsonl',
        'samples/anomaly_one_shot_20.jsonl',
        'samples/defect_20.jsonl',
        'scripts/evaluate.py',
        'smoke_review.csv',
    ):
        assert expected in joined

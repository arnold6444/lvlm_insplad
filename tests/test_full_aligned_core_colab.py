import json
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_full_runner_compiles_and_has_expected_contract():
    root = _root()
    runner = (root / 'scripts' / 'run_hf_aligned_core_full.py').read_text(encoding='utf-8')
    compile(runner, 'run_hf_aligned_core_full.py', 'exec')

    for expected in (
        'Qwen/Qwen3.5-4B',
        'OpenGVLab/InternVL3_5-8B',
        'openbmb/MiniCPM-V-4_5',
        'llava-hf/llava-onevision-qwen2-7b-ov-hf',
        'Aligned Core QA must be 1378/1378/1378',
        'reusable_prediction',
        'checkpoint-every',
        'attempts: int = 3',
        'leaderboard.csv',
        'defect_macro_f1',
        'consistency_error',
    ):
        assert expected in runner


def test_full_colab_compiles_and_builds_same_image_aligned_core():
    root = _root()
    notebook = root / 'notebooks' / 'lvlm_baselines_full_colab.ipynb'
    data = json.loads(notebook.read_text(encoding='utf-8'))
    assert data['nbformat'] == 4
    assert data['metadata']['accelerator'] == 'GPU'

    joined = []
    for index, cell in enumerate(data['cells']):
        source = cell.get('source', '')
        joined.append(source)
        if cell.get('cell_type') == 'code':
            compile(source, f'full_colab_cell_{index}', 'exec')
    joined = '\n'.join(joined)

    for expected in (
        'Aligned Core 1,378',
        "assert len(core) == 1378",
        "reports/image_overlap_summary.json",
        "data/qa/anomaly_zero_shot.jsonl",
        "data/qa/anomaly_one_shot.jsonl",
        "data/qa/defect_qa.jsonl",
        "assert [r['id'] for r in zero_rows] == [r['id'] for r in one_rows]",
        'run_hf_aligned_core_full.py',
        'hf_full_aligned_core_v1',
        'GITHUB_TOKEN',
        'drive.mount',
    ):
        assert expected in joined

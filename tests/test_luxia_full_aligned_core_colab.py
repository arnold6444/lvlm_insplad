import json
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_full_luxia_runner_compiles_and_uses_aligned_core_contract():
    root = _root()
    runner = (root / 'scripts' / 'run_luxia_aligned_core_full.py').read_text(encoding='utf-8')
    compile(runner, 'run_luxia_aligned_core_full.py', 'exec')
    for expected in (
        'Aligned Core QA must be 1378/1378/1378',
        'infer_luxia_pipeline',
        'reusable_prediction',
        'attempts: int = 3',
        'checkpoint-every',
        'leaderboard.csv',
        'expected_vision_calls_without_resume',
        'expected_chat_calls_without_resume',
        'defect_macro_f1',
        'consistency_error',
    ):
        assert expected in runner
    assert 'os.environ.get("LUXIA_API_KEY"' in runner


def test_full_luxia_colab_compiles_and_rebuilds_same_aligned_core():
    root = _root()
    notebook = root / 'notebooks' / 'luxia_baselines_full_colab.ipynb'
    data = json.loads(notebook.read_text(encoding='utf-8'))
    assert data['nbformat'] == 4
    joined = []
    for index, cell in enumerate(data['cells']):
        source = cell.get('source', '')
        joined.append(source)
        if cell.get('cell_type') == 'code':
            compile(source, f'luxia_full_cell_{index}', 'exec')
    joined = '\n'.join(joined)
    for expected in (
        'Aligned Core 1,378',
        "assert len(core) == 1378",
        'run_luxia_aligned_core_full.py',
        'luxia_full_aligned_core_v1',
        'LUXIA_API_KEY',
        "run_env['LUXIA_API_KEY'] = LUXIA_API_KEY",
        'Chat 4,134회',
        'Vision(Document AI) 5,512회',
        'drive.mount',
    ):
        assert expected in joined
    assert "'--api-key'" not in joined

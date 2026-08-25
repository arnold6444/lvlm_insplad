import json
from pathlib import Path


def test_luxia_pipeline_runner_compiles_and_keeps_api_key_in_env():
    root = Path(__file__).resolve().parents[1]
    script = root / 'scripts' / 'run_luxia_smoke.py'
    source = script.read_text(encoding='utf-8')
    compile(source, str(script), 'exec')
    for expected in (
        'infer_luxia_pipeline',
        'luxia_document_ai_chat',
        'LUXIA_CHAT_MODEL',
        'expected_vision_calls',
        'expected_chat_calls',
        'leaderboard.csv',
        'prediction_review.csv',
        'run_status.json',
        'run_metadata.json',
        'reusable_prediction',
        'existing_by_id',
        'reused_predictions',
        'rows_called_this_run',
        'attempts: int = 3',
    ):
        assert expected in source
    assert 'os.environ.get("LUXIA_API_KEY"' in source

    # A row-level API failure must not suppress the entire score table.
    assert 'if row_error_count == 0:\n        evaluate_predictions' not in source
    assert 'evaluate_predictions(args.out_dir)\n    build_leaderboard(args.out_dir)' in source


def test_luxia_pipeline_colab_structure():
    root = Path(__file__).resolve().parents[1]
    notebook = root / 'notebooks' / 'luxia_pipeline_smoke_colab.ipynb'
    data = json.loads(notebook.read_text(encoding='utf-8'))
    assert data['nbformat'] == 4
    joined = []
    for index, cell in enumerate(data['cells']):
        source = cell.get('source', '')
        joined.append(source)
        if cell.get('cell_type') == 'code':
            compile(source, f'luxia_pipeline_cell_{index}', 'exec')
    joined = '\n'.join(joined)
    for expected in (
        'GITHUB_TOKEN',
        'LUXIA_API_KEY',
        'unsupervised_anomaly_detection.zip',
        'supervised_fault_classification.zip',
        'scripts/run_luxia_smoke.py',
        "run_env['LUXIA_API_KEY'] = LUXIA_API_KEY",
        "'--limit', '20'",
        'RESUME_LATEST = True',
        "glob('luxia_smoke_*.zip')",
        'archive.extractall(RUN_ROOT)',
        "print((RUN_ROOT / 'leaderboard.csv').read_text",
        'files.download(str(zip_path))',
    ):
        assert expected in joined

    # Do not expose the API key as a command-line argument.
    assert "'--api-key'" not in joined

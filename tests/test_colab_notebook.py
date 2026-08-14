import json
from pathlib import Path


def _joined_notebook_source() -> str:
    root = Path(__file__).resolve().parents[1]
    notebook = root / 'notebooks' / 'lvlm_baselines_colab.ipynb'
    data = json.loads(notebook.read_text(encoding='utf-8'))

    assert data['nbformat'] == 4
    assert data['metadata']['accelerator'] == 'GPU'
    assert len(data['cells']) >= 10

    sources = []
    for index, cell in enumerate(data['cells']):
        source = cell.get('source', '')
        sources.append(source)
        if cell.get('cell_type') == 'code':
            compile(source, f'lvlm_baselines_cell_{index}', 'exec')
    return '\n'.join(sources)


def test_colab_contains_full_benchmark_flow_inline():
    joined = _joined_notebook_source()
    for expected in (
        "BRANCH = 'main'",
        'GITHUB_TOKEN',
        'LUXIA_API_KEY',
        'LUXIA_EXTERNAL_CHAT_URL',
        'gpt-5.3-chat-latest',
        'gemini-3-flash-preview',
        'infer_luxia_external',
        'API_PROBE_STATUS',
        'skipped_external_route_missing',
        'skipped_api_probe_failed',
        'Qwen/Qwen3-VL-8B-Instruct',
        'OpenGVLab/InternVL3_5-8B',
        'openbmb/MiniCPM-V-4_5',
        'llava-hf/llava-onevision-qwen2-7b-ov-hf',
        "'adapter': 'internvl_chat'",
        "'adapter': 'minicpm_chat'",
        'load_hf_adapter',
        'infer_hf',
        'samples/anomaly_zero_shot_20.jsonl',
        'samples/anomaly_one_shot_20.jsonl',
        'samples/defect_20.jsonl',
        'scripts/evaluate.py',
        'leaderboard.csv',
        'prediction_review.csv',
        'run_metadata.json',
        'files.download(str(zip_path))',
    ):
        assert expected in joined

    assert 'run_luxia_colab.py' not in joined
    assert 'OPENAI_API_KEY' not in joined
    assert 'GEMINI_API_KEY' not in joined


def test_colab_repairs_live_kernel_import_path():
    joined = _joined_notebook_source()
    assert "SRC_DIR = REPO_DIR / 'src'" in joined
    assert 'sys.path.insert(0, str(SRC_DIR))' in joined
    assert 'import insplad_benchmark' in joined


def test_colab_does_not_use_luxia_native_chat_for_external_models():
    joined = _joined_notebook_source()
    assert 'The public /luxia/v1/chat route only accepts Luxia model IDs.' in joined
    assert "LUXIA_EXTERNAL_CHAT_URL = get_secret('LUXIA_EXTERNAL_CHAT_URL')" in joined
    assert "status = 'skipped_external_route_missing'" in joined


def test_colab_uses_runtime_uploads_instead_of_google_drive():
    joined = _joined_notebook_source()
    assert "Path('/content')" in joined
    assert 'unsupervised_anomaly_detection.zip' in joined
    assert 'supervised_fault_classification.zip' in joined
    assert "Path('/content/lvlm_insplad_results')" in joined
    assert 'drive.mount' not in joined
    assert '/content/drive' not in joined
    assert 'MyDrive' not in joined


def test_colab_preserves_model_specific_hf_adapters():
    joined = _joined_notebook_source()
    assert "'adapter': 'pipeline'" in joined
    assert "'adapter': 'internvl_chat'" in joined
    assert "'adapter': 'minicpm_chat'" in joined
    assert 'load_hf_adapter(spec)' in joined
    assert 'torch.cuda.empty_cache()' in joined


def test_colab_writes_diagnostic_csv_headers_when_nothing_completed():
    joined = _joined_notebook_source()
    assert 'No models completed.' in joined
    assert 'LEADERBOARD_COLUMNS' in joined
    assert 'REVIEW_COLUMNS' in joined

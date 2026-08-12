import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _joined_notebook_source() -> str:
    notebook = _repo_root() / 'notebooks' / 'lvlm_baselines_colab.ipynb'
    data = json.loads(notebook.read_text(encoding='utf-8'))

    assert data['nbformat'] == 4
    assert data['metadata']['accelerator'] == 'GPU'

    sources = []
    for index, cell in enumerate(data['cells']):
        source = cell.get('source', '')
        sources.append(source)
        if cell.get('cell_type') == 'code':
            compile(source, f'lvlm_baselines_cell_{index}', 'exec')
    return '\n'.join(sources)


def _runner_source() -> str:
    source = (_repo_root() / 'scripts' / 'run_luxia_colab.py').read_text(encoding='utf-8')
    compile(source, 'run_luxia_colab.py', 'exec')
    return source


def test_colab_notebook_bootstraps_main_and_luxia_runner():
    joined = _joined_notebook_source()
    for expected in (
        "BRANCH = 'main'",
        'GITHUB_TOKEN',
        'LUXIA_API_KEY',
        'scripts/run_luxia_colab.py',
        "COLAB_ROOT" if False else "Path('/content')",
    ):
        assert expected in joined

    assert 'OPENAI_API_KEY' not in joined
    assert 'GEMINI_API_KEY' not in joined
    assert 'drive.mount' not in joined
    assert '/content/drive' not in joined
    assert 'MyDrive' not in joined


def test_luxia_runner_keeps_common_benchmark_and_hf_adapters():
    runner = _runner_source()
    for expected in (
        'gpt-5.3-chat-latest',
        'gemini-3-flash-preview',
        'luxia_external',
        'LUXIA_API_KEY',
        'LUXIA_EXTERNAL_CHAT_URL',
        'infer_luxia_external',
        'api_probe_status',
        'skipped_api_probe_failed',
        'Qwen/Qwen3-VL-8B-Instruct',
        'OpenGVLab/InternVL3_5-8B',
        'openbmb/MiniCPM-V-4_5',
        'llava-hf/llava-onevision-qwen2-7b-ov-hf',
        "'adapter': 'internvl_chat'" if False else 'internvl_chat',
        'minicpm_chat',
        'load_hf_adapter',
        'infer_hf',
        'samples/anomaly_zero_shot_20.jsonl',
        'samples/anomaly_one_shot_20.jsonl',
        'samples/defect_20.jsonl',
        'leaderboard.csv',
        'prediction_review.csv',
        'run_metadata.json',
        'skipped_insufficient_gpu',
        'torch.cuda.empty_cache',
    ):
        assert expected in runner


def test_luxia_runner_uses_runtime_uploads_not_google_drive():
    runner = _runner_source()
    assert 'Path("/content")' in runner
    assert 'unsupervised_anomaly_detection.zip' in runner
    assert 'supervised_fault_classification.zip' in runner
    assert 'Path("/content/lvlm_insplad_results")' in runner
    assert 'files.download(str(zip_path))' in runner

    assert 'drive.mount' not in runner
    assert '/content/drive' not in runner
    assert 'MyDrive' not in runner

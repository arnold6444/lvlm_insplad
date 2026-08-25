import json
from pathlib import Path


def _joined_notebook_source() -> str:
    root = Path(__file__).resolve().parents[1]
    notebook = root / 'notebooks' / 'lvlm_baselines_colab.ipynb'
    data = json.loads(notebook.read_text(encoding='utf-8'))

    assert data['nbformat'] == 4
    assert data['metadata']['accelerator'] == 'GPU'
    assert len(data['cells']) >= 9

    sources = []
    for index, cell in enumerate(data['cells']):
        source = cell.get('source', '')
        sources.append(source)
        if cell.get('cell_type') == 'code':
            compile(source, f'lvlm_baselines_cell_{index}', 'exec')
    return '\n'.join(sources)


def test_colab_contains_four_family_representative_hf_flow():
    joined = _joined_notebook_source()
    for expected in (
        "BRANCH = 'main'",
        'GITHUB_TOKEN',
        'transformers==5.14.0',
        'Qwen/Qwen3.5-4B',
        'OpenGVLab/InternVL3_5-8B',
        'openbmb/MiniCPM-V-4_5',
        'llava-hf/llava-onevision-qwen2-7b-ov-hf',
        "'adapter': 'internvl_chat'",
        "'adapter': 'minicpm_chat'",
        "'adapter': 'llava_processor'",
        "'max_total_tiles': 6",
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

    for removed in (
        'Qwen/Qwen2.5-VL-7B-Instruct',
        'Qwen/Qwen3-VL-8B-Instruct',
        'OpenGVLab/InternVL2_5-8B',
        'OPENAI_API_KEY',
        'GEMINI_API_KEY',
        'LUXIA_API_KEY',
        'LUXIA_EXTERNAL_CHAT_URL',
        'gpt-5.3-chat-latest',
        'gemini-3-flash-preview',
        'infer_luxia_external',
    ):
        assert removed not in joined


def test_colab_uses_mmad_style_normal_reference_not_positive_reference():
    joined = _joined_notebook_source()
    assert 'Image 1 is a NORMAL reference' in joined
    assert "PATH_CACHE[row['reference_image']]" in joined


def test_colab_probes_models_before_full_smoke():
    joined = _joined_notebook_source()
    assert "for task in ('zero_shot', 'one_shot')" in joined
    assert "status': 'probe_failed'" in joined
    assert 'MODEL_PROBE_STATUS' in joined


def test_colab_repairs_live_kernel_import_path():
    joined = _joined_notebook_source()
    assert "SRC_DIR = REPO_DIR / 'src'" in joined
    assert 'sys.path.insert(0, str(SRC_DIR))' in joined
    assert 'import insplad_benchmark' in joined


def test_colab_uses_google_drive_for_source_zips():
    joined = _joined_notebook_source()
    assert "Path('/content/drive/MyDrive')" in joined
    assert 'unsupervised_anomaly_detection.zip' in joined
    assert 'supervised_fault_classification.zip' in joined
    assert 'drive.mount' in joined
    assert 'DRIVE_DATA_DIR' in joined


def test_colab_requires_clean_completion_before_scoring():
    joined = _joined_notebook_source()
    assert "status = 'completed' if row_error_count == 0 else 'completed_with_row_errors'" in joined
    assert "completed = [row for row in RUN_STATUS if row['status'] == 'completed']" in joined


def test_colab_preserves_failure_details_for_all_models():
    joined = _joined_notebook_source()
    assert "for model_key in HF_MODELS_TO_RUN:" in joined
    assert "'run_status': status.get('status', 'not_run')" in joined
    assert "'error': pred.get('error') or status.get('error')" in joined
    assert "'error_examples': error_examples" in joined
    assert "'peak_vram_gb': round(peak_vram_gb, 2)" in joined


def test_colab_writes_diagnostic_csv_headers_when_nothing_completed():
    joined = _joined_notebook_source()
    assert 'No cleanly completed models.' in joined
    assert 'LEADERBOARD_COLUMNS' in joined
    assert 'REVIEW_COLUMNS' in joined

import pytest

from insplad_benchmark import hf_adapters


def test_load_hf_adapter_dispatches_model_specific_paths(monkeypatch):
    monkeypatch.setattr(hf_adapters, '_load_pipeline_adapter', lambda spec: {'kind': 'pipeline'})
    monkeypatch.setattr(hf_adapters, '_load_internvl_adapter', lambda spec: {'kind': 'internvl_chat'})
    monkeypatch.setattr(hf_adapters, '_load_minicpm_adapter', lambda spec: {'kind': 'minicpm_chat'})
    monkeypatch.setattr(hf_adapters, '_load_llava_adapter', lambda spec: {'kind': 'llava_processor'})

    assert hf_adapters.load_hf_adapter({'adapter': 'pipeline'})['kind'] == 'pipeline'
    assert hf_adapters.load_hf_adapter({'adapter': 'internvl_chat'})['kind'] == 'internvl_chat'
    assert hf_adapters.load_hf_adapter({'adapter': 'minicpm_chat'})['kind'] == 'minicpm_chat'
    assert hf_adapters.load_hf_adapter({'adapter': 'llava_processor'})['kind'] == 'llava_processor'


def test_load_hf_adapter_rejects_unknown_adapter():
    with pytest.raises(ValueError, match='Unsupported HF adapter'):
        hf_adapters.load_hf_adapter({'adapter': 'not-a-real-adapter'})


def test_infer_hf_dispatches_loaded_adapter(monkeypatch):
    monkeypatch.setattr(hf_adapters, '_infer_pipeline', lambda adapter, images, prompt: 'pipeline')
    monkeypatch.setattr(hf_adapters, '_infer_qwen35', lambda adapter, images, prompt: 'qwen35')
    monkeypatch.setattr(hf_adapters, '_infer_internvl', lambda adapter, images, prompt: 'internvl')
    monkeypatch.setattr(hf_adapters, '_infer_minicpm', lambda adapter, images, prompt: 'minicpm')
    monkeypatch.setattr(hf_adapters, '_infer_llava', lambda adapter, images, prompt: 'llava')

    assert hf_adapters.infer_hf({'kind': 'pipeline'}, [], 'x') == 'pipeline'
    assert hf_adapters.infer_hf({'kind': 'qwen35_processor'}, [], 'x') == 'qwen35'
    assert hf_adapters.infer_hf({'kind': 'internvl_chat'}, [], 'x') == 'internvl'
    assert hf_adapters.infer_hf({'kind': 'minicpm_chat'}, [], 'x') == 'minicpm'
    assert hf_adapters.infer_hf({'kind': 'llava_processor'}, [], 'x') == 'llava'


def test_qwen35_pipeline_spec_uses_non_thinking_direct_adapter(monkeypatch):
    monkeypatch.setattr(
        hf_adapters,
        '_load_qwen35_adapter',
        lambda spec: {'kind': 'qwen35_processor', 'model_id': spec['model_id']},
    )
    loaded = hf_adapters._load_pipeline_adapter({'model_id': 'Qwen/Qwen3.5-4B'})
    assert loaded == {'kind': 'qwen35_processor', 'model_id': 'Qwen/Qwen3.5-4B'}


def test_transformers5_custom_model_compat_adds_missing_default():
    class LegacyRemoteModelBase:
        pass

    hf_adapters._ensure_transformers5_custom_model_compat(LegacyRemoteModelBase)
    assert LegacyRemoteModelBase.all_tied_weights_keys == {}


def test_transformers5_custom_model_compat_preserves_existing_mapping():
    existing = {'lm_head.weight': 'embed_tokens.weight'}

    class CurrentModelBase:
        all_tied_weights_keys = existing

    hf_adapters._ensure_transformers5_custom_model_compat(CurrentModelBase)
    assert CurrentModelBase.all_tied_weights_keys is existing


@pytest.mark.parametrize(
    ('total', 'count', 'expected'),
    [
        (6, 1, 6),
        (6, 2, 3),
        (6, 3, 2),
        (2, 3, 1),
    ],
)
def test_internvl_tile_budget_is_distributed_across_images(total, count, expected):
    assert hf_adapters._internvl_per_image_tile_budget(total, count) == expected


def test_internvl_tile_budget_rejects_zero_images():
    with pytest.raises(ValueError):
        hf_adapters._internvl_per_image_tile_budget(6, 0)

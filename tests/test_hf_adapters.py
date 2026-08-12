import pytest

from insplad_benchmark import hf_adapters


def test_load_hf_adapter_dispatches_model_specific_paths(monkeypatch):
    monkeypatch.setattr(hf_adapters, '_load_pipeline_adapter', lambda spec: {'kind': 'pipeline'})
    monkeypatch.setattr(hf_adapters, '_load_internvl_adapter', lambda spec: {'kind': 'internvl_chat'})
    monkeypatch.setattr(hf_adapters, '_load_minicpm_adapter', lambda spec: {'kind': 'minicpm_chat'})

    assert hf_adapters.load_hf_adapter({'adapter': 'pipeline'})['kind'] == 'pipeline'
    assert hf_adapters.load_hf_adapter({'adapter': 'internvl_chat'})['kind'] == 'internvl_chat'
    assert hf_adapters.load_hf_adapter({'adapter': 'minicpm_chat'})['kind'] == 'minicpm_chat'


def test_load_hf_adapter_rejects_unknown_adapter():
    with pytest.raises(ValueError, match='Unsupported HF adapter'):
        hf_adapters.load_hf_adapter({'adapter': 'not-a-real-adapter'})


def test_infer_hf_dispatches_loaded_adapter(monkeypatch):
    monkeypatch.setattr(hf_adapters, '_infer_pipeline', lambda adapter, images, prompt: 'pipeline')
    monkeypatch.setattr(hf_adapters, '_infer_internvl', lambda adapter, images, prompt: 'internvl')
    monkeypatch.setattr(hf_adapters, '_infer_minicpm', lambda adapter, images, prompt: 'minicpm')

    assert hf_adapters.infer_hf({'kind': 'pipeline'}, [], 'x') == 'pipeline'
    assert hf_adapters.infer_hf({'kind': 'internvl_chat'}, [], 'x') == 'internvl'
    assert hf_adapters.infer_hf({'kind': 'minicpm_chat'}, [], 'x') == 'minicpm'

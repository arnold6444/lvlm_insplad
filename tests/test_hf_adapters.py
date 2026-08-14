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


def test_generation_mixin_compat_restores_generate_and_generation_config():
    class LegacyCausalLM:
        def __init__(self):
            self.config = {'model_type': 'legacy'}

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {'input_ids': input_ids}

    class FakeGenerationMixin:
        def generate(self):
            return 'generated'

    class FakeGenerationConfig:
        @classmethod
        def from_model_config(cls, config):
            return {'from': config}

    model = LegacyCausalLM()
    original_id = id(model)
    result = hf_adapters._ensure_generation_mixin_compat(
        model,
        generation_mixin_cls=FakeGenerationMixin,
        generation_config_cls=FakeGenerationConfig,
    )
    assert result is model
    assert id(result) == original_id
    assert result.generate() == 'generated'
    assert result.generation_config == {'from': {'model_type': 'legacy'}}


def test_generation_mixin_compat_preserves_existing_generate():
    class CurrentCausalLM:
        def generate(self):
            return 'native'

    class UnexpectedMixin:
        def generate(self):
            return 'patched'

    model = CurrentCausalLM()
    original_cls = model.__class__
    result = hf_adapters._ensure_generation_mixin_compat(
        model,
        generation_mixin_cls=UnexpectedMixin,
        generation_config_cls=object,
    )
    assert result is model
    assert model.__class__ is original_cls
    assert model.generate() == 'native'


def test_generation_mixin_compat_requires_generation_inputs_hook():
    class NonGenerativeModel:
        pass

    with pytest.raises(RuntimeError, match='prepare_inputs_for_generation'):
        hf_adapters._ensure_generation_mixin_compat(
            NonGenerativeModel(),
            generation_mixin_cls=object,
            generation_config_cls=object,
        )


class _FakeInternVL25Tokenizer:
    def __init__(self, chat_shift=0):
        self.ids = {f'base-{i}': i for i in range(92538)}
        for index, token in enumerate(hf_adapters._INTERNVL25_CHAT_TOKENS):
            self.ids[token] = 92538 + chat_shift + index
        if chat_shift:
            for i in range(92538, 92538 + chat_shift):
                self.ids[f'extra-{i}'] = i
        self.model_max_length = None
        self.padding_side = 'right'

    def add_tokens(self, tokens, special_tokens=False):
        assert special_tokens is True
        next_id = max(self.ids.values()) + 1
        added = 0
        for token in tokens:
            if token not in self.ids:
                self.ids[token] = next_id
                next_id += 1
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self.ids[token]


def test_internvl25_compat_tokenizer_restores_checkpoint_token_ids():
    tokenizer = _FakeInternVL25Tokenizer()
    result = hf_adapters._prepare_internvl25_compat_tokenizer(tokenizer)
    assert result is tokenizer
    assert tokenizer.model_max_length == 16384
    assert tokenizer.padding_side == 'left'
    assert {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in hf_adapters._INTERNVL25_CHAT_TOKENS
    } == hf_adapters._INTERNVL25_EXPECTED_CHAT_TOKEN_IDS
    assert {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in hf_adapters._INTERNVL25_EXTRA_TOKENS
    } == hf_adapters._INTERNVL25_EXPECTED_TOKEN_IDS


def test_internvl25_compat_tokenizer_fails_on_shifted_chat_tokens():
    tokenizer = _FakeInternVL25Tokenizer(chat_shift=6)
    with pytest.raises(RuntimeError, match='chat-token IDs do not match'):
        hf_adapters._prepare_internvl25_compat_tokenizer(tokenizer)


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

import json
from pathlib import Path

import pytest

from insplad_benchmark import luxia_api


def test_build_multimodal_chat_payload_uses_image_data_urls(tmp_path):
    image = tmp_path / 'sample.png'
    image.write_bytes(b'png-bytes')

    payload = luxia_api.build_multimodal_chat_payload(
        'gpt-5.3-chat-latest', [image], 'classify this image'
    )

    assert payload['model'] == 'gpt-5.3-chat-latest'
    assert payload['stream'] is False
    assert payload['temperature'] == 0
    content = payload['messages'][0]['content']
    assert content[0]['type'] == 'image_url'
    assert content[0]['image_url']['url'].startswith('data:image/png;base64,')
    assert content[-1] == {'type': 'text', 'text': 'classify this image'}


def test_resolve_external_endpoint_requires_explicit_luxia_route(monkeypatch):
    monkeypatch.delenv('LUXIA_EXTERNAL_CHAT_URL', raising=False)
    with pytest.raises(RuntimeError, match='do not document a GPT/Gemini'):
        luxia_api.resolve_luxia_external_endpoint()


def test_resolve_external_endpoint_rejects_luxia_native_chat():
    with pytest.raises(RuntimeError, match='Refusing'):
        luxia_api.resolve_luxia_external_endpoint(luxia_api.LUXIA_NATIVE_CHAT_URL)


def test_infer_luxia_external_sends_apikey_and_parses_chat_response(monkeypatch, tmp_path):
    image = tmp_path / 'sample.jpg'
    image.write_bytes(b'jpeg-bytes')
    captured = {}
    endpoint = 'https://bridge.luxiacloud.com/external-model-route-from-luxia'

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {'choices': [{'message': {'role': 'assistant', 'content': '{"anomaly":"normal"}'}}]}
            ).encode('utf-8')

    def fake_urlopen(request, timeout):
        captured['url'] = request.full_url
        captured['headers'] = dict(request.header_items())
        captured['body'] = json.loads(request.data.decode('utf-8'))
        captured['timeout'] = timeout
        return FakeResponse()

    monkeypatch.setattr(luxia_api.urllib.request, 'urlopen', fake_urlopen)

    result = luxia_api.infer_luxia_external(
        api_key='secret-key',
        model_id='gemini-3-flash-preview',
        image_paths=[image],
        prompt='inspect',
        endpoint=endpoint,
    )

    assert result == '{"anomaly":"normal"}'
    assert captured['url'] == endpoint
    assert captured['headers']['Apikey'] == 'secret-key'
    assert captured['headers']['Content-type'] == 'application/json'
    assert captured['body']['model'] == 'gemini-3-flash-preview'
    assert captured['timeout'] == 120.0


def test_infer_luxia_external_requires_key(tmp_path):
    image = tmp_path / 'sample.jpg'
    image.write_bytes(b'jpeg-bytes')

    with pytest.raises(RuntimeError, match='LUXIA_API_KEY'):
        luxia_api.infer_luxia_external(
            api_key='',
            model_id='gpt-5.3-chat-latest',
            image_paths=[image],
            prompt='inspect',
            endpoint='https://bridge.luxiacloud.com/external',
        )

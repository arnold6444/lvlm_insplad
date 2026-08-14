import base64
from pathlib import Path

import pytest

from insplad_benchmark import luxia_adapter
from insplad_benchmark.luxia_adapter import (
    _extract_chat_content,
    _extract_content,
    image_to_data_uri,
)


def test_image_to_data_uri_png(tmp_path):
    path = tmp_path / "x.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nabc")
    uri = image_to_data_uri(path)
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == path.read_bytes()


def test_image_to_data_uri_rejects_unknown_type(tmp_path):
    path = tmp_path / "x.bmp"
    path.write_bytes(b"BM")
    with pytest.raises(ValueError, match="Unsupported Luxia image format"):
        image_to_data_uri(path)


def test_extract_content_top_level():
    assert _extract_content({"content": " hello "}) == "hello"


def test_extract_content_nested():
    assert _extract_content({"data": {"content": " nested "}}) == "nested"


def test_extract_content_unknown_shape():
    assert _extract_content({"result": "x"}) == ""


def test_extract_chat_content():
    payload = {"choices": [{"message": {"role": "assistant", "content": "  ok  "}}]}
    assert _extract_chat_content(payload) == "ok"


def test_infer_luxia_pipeline_uses_one_vision_call_per_image_and_chat(monkeypatch, tmp_path):
    images = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    for image in images:
        image.write_bytes(b"jpg")

    vision_calls = []
    chat_calls = []

    def fake_vision(path, api_key):
        vision_calls.append((Path(path).name, api_key))
        return {"content": f"description:{Path(path).name}", "latency_s": 1.0}

    def fake_chat(prompt, api_key, *, model=luxia_adapter.LUXIA_CHAT_MODEL, **kwargs):
        chat_calls.append((prompt, api_key, model))
        return {"content": '{"anomaly":"anomalous"}', "latency_s": 0.5, "model": model}

    monkeypatch.setattr(luxia_adapter, "call_luxia_document_ai", fake_vision)
    monkeypatch.setattr(luxia_adapter, "call_luxia_chat", fake_chat)

    result = luxia_adapter.infer_luxia_pipeline(
        images,
        'Return {"anomaly":"normal|anomalous"}.',
        "secret",
    )

    assert vision_calls == [("a.jpg", "secret"), ("b.jpg", "secret")]
    assert len(chat_calls) == 1
    assert "Image 1 visual description" in chat_calls[0][0]
    assert "Image 2 visual description" in chat_calls[0][0]
    assert result["output"] == '{"anomaly":"anomalous"}'
    assert result["latency_s"] == 2.5


def test_infer_luxia_pipeline_requires_image():
    with pytest.raises(ValueError, match="at least one image"):
        luxia_adapter.infer_luxia_pipeline([], "x", "secret")

import base64

import pytest

from insplad_benchmark.luxia_adapter import _extract_content, image_to_data_uri


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

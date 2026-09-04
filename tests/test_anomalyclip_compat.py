from __future__ import annotations

from pathlib import Path

import pytest

from insplad_benchmark.anomalyclip_compat import (
    MODEL_LOAD_RELATIVE_PATH,
    patch_official_anomalyclip_checkout,
)


LEGACY_SOURCE = """import hashlib
import os
import urllib
import warnings
from pkg_resources import packaging

import torch

if packaging.version.parse(torch.__version__) < packaging.version.parse(\"1.7.1\"):
    warnings.warn(\"old torch\")


def download(url):
    return urllib.request.urlopen(url)
"""


def _write_loader(root: Path, source: str = LEGACY_SOURCE) -> Path:
    path = root / MODEL_LOAD_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_patch_modernizes_official_loader_and_is_idempotent(tmp_path: Path) -> None:
    path = _write_loader(tmp_path)

    first = patch_official_anomalyclip_checkout(tmp_path)
    patched = path.read_text(encoding="utf-8")
    assert first["changed"] is True
    assert first["applied"] == [
        "pkg_resources_to_packaging",
        "packaging_version_reference",
        "urllib_request_import",
    ]
    assert first["sha256_before"] != first["sha256_after"]
    assert "from packaging import version" in patched
    assert "version.parse(torch.__version__)" in patched
    assert "version.parse(\"1.7.1\")" in patched
    assert "import urllib.request" in patched
    assert "from pkg_resources import packaging" not in patched

    second = patch_official_anomalyclip_checkout(tmp_path)
    assert second["changed"] is False
    assert second["applied"] == []
    assert second["sha256_before"] == second["sha256_after"]
    assert path.read_text(encoding="utf-8") == patched


def test_patch_fails_closed_for_missing_or_unknown_loader(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        patch_official_anomalyclip_checkout(tmp_path)

    _write_loader(tmp_path, "import torch\n")
    with pytest.raises(RuntimeError, match="Unsupported AnomalyCLIP"):
        patch_official_anomalyclip_checkout(tmp_path)

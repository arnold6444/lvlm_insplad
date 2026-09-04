from __future__ import annotations

from pathlib import Path

import pytest

from insplad_benchmark.anomalyclip_compat import (
    MODEL_LOAD_RELATIVE_PATH,
    MODEL_RELATIVE_PATH,
    patch_official_anomalyclip_checkout,
)


LEGACY_LOADER_SOURCE = """import hashlib
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

LEGACY_MODEL_SOURCE = """import torch
from torch import nn

from thop import profile
class AnomalyCLIP(nn.Module):
    pass
"""


def _write_sources(
    root: Path,
    *,
    loader_source: str = LEGACY_LOADER_SOURCE,
    model_source: str = LEGACY_MODEL_SOURCE,
) -> tuple[Path, Path]:
    loader_path = root / MODEL_LOAD_RELATIVE_PATH
    model_path = root / MODEL_RELATIVE_PATH
    loader_path.parent.mkdir(parents=True, exist_ok=True)
    loader_path.write_text(loader_source, encoding="utf-8")
    model_path.write_text(model_source, encoding="utf-8")
    return loader_path, model_path


def test_patch_modernizes_official_sources_and_is_idempotent(tmp_path: Path) -> None:
    loader_path, model_path = _write_sources(tmp_path)

    first = patch_official_anomalyclip_checkout(tmp_path)
    patched_loader = loader_path.read_text(encoding="utf-8")
    patched_model = model_path.read_text(encoding="utf-8")
    assert first["schema_version"] == 2
    assert first["changed"] is True
    assert first["applied"] == [
        "pkg_resources_to_packaging",
        "packaging_version_reference",
        "urllib_request_import",
        "remove_unused_thop_import",
    ]
    assert len(first["changed_paths"]) == 2
    assert first["sha256_before"] != first["sha256_after"]
    assert first["files"][MODEL_LOAD_RELATIVE_PATH.as_posix()]["changed"] is True
    assert first["files"][MODEL_RELATIVE_PATH.as_posix()]["changed"] is True
    assert "from packaging import version" in patched_loader
    assert "version.parse(torch.__version__)" in patched_loader
    assert "version.parse(\"1.7.1\")" in patched_loader
    assert "import urllib.request" in patched_loader
    assert "from pkg_resources import packaging" not in patched_loader
    assert "from thop import profile" not in patched_model
    assert "class AnomalyCLIP(nn.Module):" in patched_model

    second = patch_official_anomalyclip_checkout(tmp_path)
    assert second["changed"] is False
    assert second["already_compatible"] is True
    assert second["applied"] == []
    assert second["changed_paths"] == []
    assert second["sha256_before"] == second["sha256_after"]
    assert loader_path.read_text(encoding="utf-8") == patched_loader
    assert model_path.read_text(encoding="utf-8") == patched_model


def test_patch_fails_closed_for_missing_or_unknown_sources(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        patch_official_anomalyclip_checkout(tmp_path)

    _write_sources(tmp_path, loader_source="import torch\n")
    with pytest.raises(RuntimeError, match="Unsupported AnomalyCLIP"):
        patch_official_anomalyclip_checkout(tmp_path)


def test_patch_refuses_to_remove_profiler_when_profile_is_used(tmp_path: Path) -> None:
    model_source = LEGACY_MODEL_SOURCE + "\ndef measure(model, inputs):\n    return profile(model, inputs)\n"
    _write_sources(tmp_path, model_source=model_source)

    with pytest.raises(RuntimeError, match=r"profile\(\) is used"):
        patch_official_anomalyclip_checkout(tmp_path)

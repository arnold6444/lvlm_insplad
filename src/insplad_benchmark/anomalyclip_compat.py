from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final


COMPAT_SCHEMA_VERSION: Final[int] = 1
MODEL_LOAD_RELATIVE_PATH: Final[Path] = Path("AnomalyCLIP_lib/model_load.py")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def patch_official_anomalyclip_checkout(checkout_root: Path) -> dict[str, object]:
    """Patch the pinned official AnomalyCLIP checkout for modern Python packages.

    The official commit used by the benchmark imports ``packaging`` through the
    removed ``pkg_resources`` shim and imports only the top-level ``urllib``
    package before accessing ``urllib.request``. This deterministic source patch
    replaces those two legacy imports without changing model behavior.

    The function is idempotent and fails closed when the expected official source
    layout is absent or when an unknown source variant cannot be verified.
    """

    checkout_root = Path(checkout_root)
    model_load_path = checkout_root / MODEL_LOAD_RELATIVE_PATH
    if not model_load_path.is_file():
        raise FileNotFoundError(
            f"AnomalyCLIP model loader is missing: {model_load_path}"
        )

    original_bytes = model_load_path.read_bytes()
    try:
        source = original_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"AnomalyCLIP model loader is not valid UTF-8: {model_load_path}"
        ) from exc

    original_source = source
    applied: list[str] = []

    if "from pkg_resources import packaging" in source:
        source = source.replace(
            "from pkg_resources import packaging",
            "from packaging import version",
            1,
        )
        applied.append("pkg_resources_to_packaging")
    if "packaging.version.parse(" in source:
        source = source.replace("packaging.version.parse(", "version.parse(")
        applied.append("packaging_version_reference")

    if "import urllib\n" in source:
        source = source.replace("import urllib\n", "import urllib.request\n", 1)
        applied.append("urllib_request_import")

    required_fragments = (
        "from packaging import version",
        "version.parse(torch.__version__)",
        "version.parse(\"1.7.1\")",
        "import urllib.request",
        "urllib.request.urlopen(url)",
    )
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise RuntimeError(
            "Unsupported AnomalyCLIP model loader variant; compatibility patch "
            f"could not verify fragments: {missing}"
        )
    forbidden_fragments = (
        "from pkg_resources import packaging",
        "packaging.version.parse(",
        "import urllib\n",
    )
    remaining = [fragment for fragment in forbidden_fragments if fragment in source]
    if remaining:
        raise RuntimeError(
            "AnomalyCLIP compatibility patch left legacy fragments behind: "
            f"{remaining}"
        )

    patched_bytes = source.encode("utf-8")
    changed = patched_bytes != original_bytes
    if changed:
        temporary = model_load_path.with_suffix(model_load_path.suffix + ".tmp")
        temporary.write_bytes(patched_bytes)
        temporary.replace(model_load_path)

    return {
        "schema_version": COMPAT_SCHEMA_VERSION,
        "model_load_path": str(model_load_path.resolve()),
        "changed": changed,
        "applied": applied,
        "sha256_before": _sha256_bytes(original_bytes),
        "sha256_after": _sha256_bytes(patched_bytes),
        "already_compatible": not changed and original_source == source,
    }

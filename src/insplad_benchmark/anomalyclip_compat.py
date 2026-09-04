from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final


COMPAT_SCHEMA_VERSION: Final[int] = 3
MODEL_LOAD_RELATIVE_PATH: Final[Path] = Path("AnomalyCLIP_lib/model_load.py")
MODEL_RELATIVE_PATH: Final[Path] = Path("AnomalyCLIP_lib/AnomalyCLIP.py")
PROMPT_RELATIVE_PATH: Final[Path] = Path("prompt_ensemble.py")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_utf8(path: Path, *, label: str) -> tuple[bytes, str]:
    if not path.is_file():
        raise FileNotFoundError(f"AnomalyCLIP {label} is missing: {path}")
    original = path.read_bytes()
    try:
        return original, original.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"AnomalyCLIP {label} is not valid UTF-8: {path}") from exc


def _atomic_write(path: Path, value: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _modernize_packaging_import(
    source: str,
    *,
    label: str,
    applied: list[str],
) -> str:
    legacy_import = "from pkg_resources import packaging"
    if legacy_import in source:
        if source.count(legacy_import) != 1:
            raise RuntimeError(
                f"Unsupported AnomalyCLIP {label}: expected one pkg_resources import"
            )
        source = source.replace(legacy_import, "from packaging import version", 1)
        applied.append(f"{label}_pkg_resources_to_packaging")
    if "packaging.version.parse(" in source:
        source = source.replace("packaging.version.parse(", "version.parse(")
        applied.append(f"{label}_packaging_version_reference")
    return source


def patch_official_anomalyclip_checkout(checkout_root: Path) -> dict[str, object]:
    """Patch the pinned official AnomalyCLIP checkout for modern Colab.

    The pinned official commit has runtime-only compatibility problems:

    - two modules import ``packaging`` through the removed ``pkg_resources`` shim;
    - the model loader imports only top-level ``urllib`` before using
      ``urllib.request``;
    - the model imports the undeclared ``thop`` profiler even though ``profile``
      is not used by inference.

    This deterministic patch modernizes imports and removes only the unused
    profiler import. Model architecture, weights, prompts, and inference logic
    remain unchanged. The function is idempotent and fails closed when the
    expected source layout or source fragments cannot be verified.
    """

    checkout_root = Path(checkout_root)
    model_load_path = checkout_root / MODEL_LOAD_RELATIVE_PATH
    model_path = checkout_root / MODEL_RELATIVE_PATH
    prompt_path = checkout_root / PROMPT_RELATIVE_PATH

    model_load_original, model_load_source = _read_utf8(
        model_load_path,
        label="model loader",
    )
    model_original, model_source = _read_utf8(
        model_path,
        label="model source",
    )
    prompt_original, prompt_source = _read_utf8(
        prompt_path,
        label="prompt module",
    )

    applied: list[str] = []
    model_load_source = _modernize_packaging_import(
        model_load_source,
        label="model_loader",
        applied=applied,
    )
    prompt_source = _modernize_packaging_import(
        prompt_source,
        label="prompt_module",
        applied=applied,
    )

    if "import urllib\n" in model_load_source:
        model_load_source = model_load_source.replace(
            "import urllib\n",
            "import urllib.request\n",
            1,
        )
        applied.append("urllib_request_import")

    profiler_import = "from thop import profile\n"
    if profiler_import in model_source:
        if model_source.count(profiler_import) != 1:
            raise RuntimeError(
                "Unsupported AnomalyCLIP model source: expected one thop import"
            )
        if "profile(" in model_source.replace(profiler_import, ""):
            raise RuntimeError(
                "Refusing to remove AnomalyCLIP thop import because profile() is used"
            )
        model_source = model_source.replace(profiler_import, "", 1)
        applied.append("remove_unused_thop_import")

    required_model_load_fragments = (
        "from packaging import version",
        "version.parse(torch.__version__)",
        "version.parse(\"1.7.1\")",
        "import urllib.request",
        "urllib.request.urlopen(url)",
    )
    missing = [
        fragment
        for fragment in required_model_load_fragments
        if fragment not in model_load_source
    ]
    if missing:
        raise RuntimeError(
            "Unsupported AnomalyCLIP model loader variant; compatibility patch "
            f"could not verify fragments: {missing}"
        )

    required_prompt_fragments = (
        "from packaging import version",
        "version.parse(torch.__version__)",
        "version.parse(\"1.8.0\")",
        "class AnomalyCLIP_PromptLearner(nn.Module):",
    )
    prompt_missing = [
        fragment for fragment in required_prompt_fragments if fragment not in prompt_source
    ]
    if prompt_missing:
        raise RuntimeError(
            "Unsupported AnomalyCLIP prompt module variant; compatibility patch "
            f"could not verify fragments: {prompt_missing}"
        )

    forbidden_fragments = (
        "from pkg_resources import packaging",
        "packaging.version.parse(",
    )
    remaining_by_file = {
        MODEL_LOAD_RELATIVE_PATH.as_posix(): [
            fragment for fragment in forbidden_fragments if fragment in model_load_source
        ],
        PROMPT_RELATIVE_PATH.as_posix(): [
            fragment for fragment in forbidden_fragments if fragment in prompt_source
        ],
    }
    remaining_by_file = {
        path: fragments for path, fragments in remaining_by_file.items() if fragments
    }
    if remaining_by_file:
        raise RuntimeError(
            "AnomalyCLIP compatibility patch left legacy packaging fragments behind: "
            f"{remaining_by_file}"
        )
    if "import urllib\n" in model_load_source:
        raise RuntimeError(
            "AnomalyCLIP compatibility patch left the legacy urllib import behind"
        )
    if "from thop import profile" in model_source:
        raise RuntimeError(
            "AnomalyCLIP compatibility patch left the undeclared thop import behind"
        )
    if "class AnomalyCLIP(nn.Module):" not in model_source:
        raise RuntimeError(
            "Unsupported AnomalyCLIP model source: model class was not found"
        )

    patched_by_path: dict[Path, bytes] = {
        model_load_path: model_load_source.encode("utf-8"),
        model_path: model_source.encode("utf-8"),
        prompt_path: prompt_source.encode("utf-8"),
    }
    original_by_path = {
        model_load_path: model_load_original,
        model_path: model_original,
        prompt_path: prompt_original,
    }
    changed_paths: list[str] = []
    for path, patched in patched_by_path.items():
        if patched != original_by_path[path]:
            _atomic_write(path, patched)
            changed_paths.append(str(path.resolve()))

    files = {
        path.relative_to(checkout_root).as_posix(): {
            "sha256_before": _sha256_bytes(original_by_path[path]),
            "sha256_after": _sha256_bytes(patched_by_path[path]),
            "changed": patched_by_path[path] != original_by_path[path],
        }
        for path in patched_by_path
    }
    changed = bool(changed_paths)
    return {
        "schema_version": COMPAT_SCHEMA_VERSION,
        "model_load_path": str(model_load_path.resolve()),
        "model_path": str(model_path.resolve()),
        "prompt_path": str(prompt_path.resolve()),
        "changed": changed,
        "changed_paths": changed_paths,
        "applied": applied,
        # Retain these keys for the existing notebook metadata contract.
        "sha256_before": _sha256_bytes(model_load_original),
        "sha256_after": _sha256_bytes(patched_by_path[model_load_path]),
        "files": files,
        "already_compatible": not changed,
    }

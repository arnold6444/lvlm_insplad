from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_FILES = (
    "leaderboard.csv",
    "baseline_vs_reference.csv",
    "asset_baseline_vs_reference.csv",
    "metrics_by_asset.csv",
    "screening_summary.csv",
    "build_summary.json",
    "master_counts.csv",
    "manual_review_final.csv",
    "run_status.json",
    "run_metadata.json",
)
TEXT_SUFFIXES = {".json", ".csv", ".txt", ".md"}
MAX_PREVIEW_CHARS = 20_000


def find_my_drive_root(path: Path) -> Path:
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if candidate.name == "MyDrive":
            return candidate
    raise RuntimeError(f"MyDrive root not found above: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_preview(path: Path) -> str:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return ""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return ""
    if len(text) <= MAX_PREVIEW_CHARS:
        return text
    return text[:MAX_PREVIEW_CHARS] + "\n...[truncated]...\n"


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_registry(path: Path) -> dict:
    if not path.is_file():
        return {"schema_version": 2, "entries": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 2, "entries": {}}
    if not isinstance(value, dict):
        return {"schema_version": 2, "entries": {}}
    value.setdefault("schema_version", 2)
    value.setdefault("entries", {})
    return value


def _write_searchable_text_index(path: Path, registry: dict) -> None:
    lines = [
        "LVLM InsPLAD ChatGPT Result Index",
        f"updated_at_utc: {registry.get('updated_at_utc', '')}",
        "",
    ]
    entries = registry.get("entries", {})
    for prefix in sorted(entries):
        entry = entries[prefix]
        lines += [
            f"## {prefix}",
            f"source_dir: {entry.get('source_dir', '')}",
            f"published_at_utc: {entry.get('published_at_utc', '')}",
        ]
        for item in entry.get("files", []):
            lines += [
                f"### {item.get('source_name', '')}",
                f"root_copy: {item.get('root_copy', '')}",
                f"sha256: {item.get('sha256', '')}",
                f"size_bytes: {item.get('size_bytes', '')}",
            ]
            preview = str(item.get("preview", "")).strip()
            if preview:
                lines += ["--- content preview ---", preview, "--- end preview ---"]
        missing = entry.get("missing", [])
        if missing:
            lines.append("missing: " + ", ".join(missing))
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def publish(out_dir: Path, prefix: str, filenames: tuple[str, ...] = DEFAULT_FILES) -> dict:
    out_dir = out_dir.resolve()
    mydrive = find_my_drive_root(out_dir)
    copied: list[str] = []
    missing: list[str] = []
    file_entries: list[dict] = []

    for name in filenames:
        source = out_dir / name
        if not source.is_file():
            missing.append(name)
            continue
        suffix = source.suffix
        stem = Path(name).stem
        target = mydrive / f"{prefix}_{stem}_latest{suffix}"
        shutil.copy2(source, target)
        copied.append(str(target))
        file_entries.append(
            {
                "source_name": name,
                "source_path": str(source),
                "root_copy": str(target),
                "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source),
                "preview": text_preview(source),
            }
        )

    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "source_dir": str(out_dir),
        "prefix": prefix,
        "published_at_utc": now,
        "copied": copied,
        "missing": missing,
        "files": file_entries,
    }
    manifest_path = mydrive / f"{prefix}_publish_manifest_latest.json"
    _atomic_json(manifest_path, manifest)

    registry_path = mydrive / "lvlm_chatgpt_index_latest.json"
    registry = _load_registry(registry_path)
    registry["schema_version"] = 2
    registry["updated_at_utc"] = now
    registry.setdefault("entries", {})[prefix] = manifest
    _atomic_json(registry_path, registry)

    text_index_path = mydrive / "lvlm_chatgpt_index_latest.txt"
    _write_searchable_text_index(text_index_path, registry)

    return {
        **manifest,
        "manifest": str(manifest_path),
        "chatgpt_index_json": str(registry_path),
        "chatgpt_index_txt": str(text_index_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--files", nargs="*")
    args = parser.parse_args()
    filenames = tuple(args.files) if args.files else DEFAULT_FILES
    print(json.dumps(publish(args.out_dir, args.prefix, filenames), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

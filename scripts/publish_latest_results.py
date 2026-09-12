from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DEFAULT_FILES = (
    "leaderboard.csv",
    "baseline_vs_reference.csv",
    "asset_baseline_vs_reference.csv",
    "metrics_by_asset.csv",
    "run_status.json",
    "run_metadata.json",
)


def find_my_drive_root(path: Path) -> Path:
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if candidate.name == "MyDrive":
            return candidate
    raise RuntimeError(f"MyDrive root not found above: {path}")


def publish(out_dir: Path, prefix: str, filenames: tuple[str, ...] = DEFAULT_FILES) -> dict:
    out_dir = out_dir.resolve()
    mydrive = find_my_drive_root(out_dir)
    copied = []
    missing = []

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

    manifest = {
        "source_dir": str(out_dir),
        "prefix": prefix,
        "copied": copied,
        "missing": missing,
    }
    manifest_path = mydrive / f"{prefix}_publish_manifest_latest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest": str(manifest_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.out_dir, args.prefix), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

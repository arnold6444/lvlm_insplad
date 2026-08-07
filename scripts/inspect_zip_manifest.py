#!/usr/bin/env python3
"""List ZIP members without extracting images or assuming an InsPLAD folder schema."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from zipfile import BadZipFile, ZipFile


def inspect_zip(zip_path: Path, out_csv: Path, out_json: Path) -> None:
    with ZipFile(zip_path) as zf:
        infos = zf.infolist()
        rows = []
        extension_counts: Counter[str] = Counter()
        top_level_counts: Counter[str] = Counter()
        for info in infos:
            p = Path(info.filename)
            suffix = p.suffix.lower()
            if not info.is_dir():
                extension_counts[suffix or "<none>"] += 1
            parts = p.parts
            if parts:
                top_level_counts[parts[0]] += 1
            rows.append({"path": info.filename, "is_dir": info.is_dir(), "size": info.file_size, "compressed_size": info.compress_size, "extension": suffix, "crc32": f"{info.CRC:08x}"})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "is_dir", "size", "compressed_size", "extension", "crc32"])
        writer.writeheader(); writer.writerows(rows)

    summary = {"zip": str(zip_path), "members": len(rows), "files": sum(not row["is_dir"] for row in rows), "directories": sum(row["is_dir"] for row in rows), "top_level_entries": dict(sorted(top_level_counts.items())), "extension_counts": dict(sorted(extension_counts.items())), "note": "This manifest intentionally does not infer asset categories, splits, labels, or defects."}
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    try:
        inspect_zip(args.zip_path, args.out_csv, args.out_json)
    except (FileNotFoundError, BadZipFile) as exc:
        parser.error(str(exc))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

import csv
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


def test_zip_manifest_does_not_infer_dataset_schema(tmp_path: Path):
    archive = tmp_path / "sample.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("root/foo/a.jpg", b"abc")
        zf.writestr("root/bar/meta.json", b"{}")
    out_csv = tmp_path / "manifest.csv"; out_json = tmp_path / "summary.json"
    subprocess.run([sys.executable, "scripts/inspect_zip_manifest.py", str(archive), "--out-csv", str(out_csv), "--out-json", str(out_json)], check=True)
    summary = json.loads(out_json.read_text())
    assert summary["files"] == 2
    assert summary["top_level_entries"] == {"root": 2}
    assert "asset" not in summary
    with out_csv.open() as f: rows = list(csv.DictReader(f))
    assert {row["extension"] for row in rows} == {".jpg", ".json"}

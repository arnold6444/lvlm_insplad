from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_vision_expert_smoke_data.py"


def _png_bytes(value: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=(value % 255, (value * 3) % 255, (value * 7) % 255)).save(
        buffer,
        format="PNG",
    )
    return buffer.getvalue()


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    zip_path = tmp_path / "unsupervised_anomaly_detection.zip"
    query_rows: list[dict[str, object]] = []

    with ZipFile(zip_path, "w") as archive:
        for index in range(4):
            path = f"dataset/asset-a/train/good/train-{index:02d}.png"
            archive.writestr(path, _png_bytes(index + 1))

        for index in range(20):
            label_dir = "good" if index < 10 else "rust"
            answer = "normal" if index < 10 else "anomalous"
            path = f"dataset/asset-a/test/{label_dir}/query-{index:02d}.png"
            archive.writestr(path, _png_bytes(index + 30))
            query_rows.append(
                {
                    "id": f"query-{index:02d}",
                    "image": path,
                    "asset_category": "asset-a",
                    "answer": answer,
                }
            )

    manifest_path = tmp_path / "unsup.csv"
    with ZipFile(zip_path) as archive, manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "is_dir",
                "size",
                "compressed_size",
                "extension",
                "crc32",
            ],
        )
        writer.writeheader()
        for info in archive.infolist():
            writer.writerow(
                {
                    "path": info.filename,
                    "is_dir": False,
                    "size": info.file_size,
                    "compressed_size": info.compress_size,
                    "extension": Path(info.filename).suffix,
                    "crc32": f"{info.CRC:08x}",
                }
            )

    qa_path = tmp_path / "smoke.jsonl"
    qa_path.write_text(
        "".join(json.dumps(row) + "\n" for row in query_rows),
        encoding="utf-8",
    )
    return manifest_path, zip_path, qa_path


def _run_builder(
    manifest_path: Path,
    zip_path: Path,
    qa_path: Path,
    out_dir: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--anomaly-manifest",
            str(manifest_path),
            "--unsup-zip",
            str(zip_path),
            "--qa",
            str(qa_path),
            "--out-dir",
            str(out_dir),
            "--max-train-per-asset",
            "2",
            "--seed",
            "42",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def _manifest_rows(out_dir: Path) -> list[dict[str, str]]:
    with (out_dir / "workspace_manifest.csv").open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        return list(csv.DictReader(handle))


def test_builder_reuses_complete_workspace_and_repairs_missing_file(tmp_path: Path) -> None:
    manifest_path, zip_path, qa_path = _write_fixture(tmp_path)
    out_dir = tmp_path / "workspace"

    first = _run_builder(manifest_path, zip_path, qa_path, out_dir)
    assert first.returncode == 0, first.stdout + first.stderr
    config = json.loads((out_dir / "workspace_config.json").read_text(encoding="utf-8"))
    assert config["schema_version"] == 2
    assert config["qa_count"] == 20
    assert config["counts"]["query:asset-a"] == 20
    assert config["counts"]["train_normal:asset-a"] == 2

    second = _run_builder(manifest_path, zip_path, qa_path, out_dir)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "Reusing complete workspace" in second.stdout

    missing_relative = _manifest_rows(out_dir)[0]["local_path"]
    missing_path = out_dir / missing_relative
    missing_path.unlink()
    assert not missing_path.exists()

    repaired = _run_builder(manifest_path, zip_path, qa_path, out_dir)
    assert repaired.returncode == 0, repaired.stdout + repaired.stderr
    assert "Rebuilding stale/incomplete workspace" in repaired.stdout
    assert missing_path.is_file()
    assert not list(tmp_path.glob(".workspace.building-*"))


def test_builder_replaces_partial_or_mismatched_workspace_without_force(tmp_path: Path) -> None:
    manifest_path, zip_path, qa_path = _write_fixture(tmp_path)
    out_dir = tmp_path / "workspace"
    out_dir.mkdir()
    (out_dir / "leftover.txt").write_text("partial", encoding="utf-8")

    partial = _run_builder(manifest_path, zip_path, qa_path, out_dir)
    assert partial.returncode == 0, partial.stdout + partial.stderr
    assert "Rebuilding stale/incomplete workspace" in partial.stdout
    assert not (out_dir / "leftover.txt").exists()

    config_path = out_dir / "workspace_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["seed"] = 999
    config_path.write_text(json.dumps(config), encoding="utf-8")

    mismatched = _run_builder(manifest_path, zip_path, qa_path, out_dir)
    assert mismatched.returncode == 0, mismatched.stdout + mismatched.stderr
    assert "workspace contract mismatch" in mismatched.stdout
    repaired_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert repaired_config["seed"] == 42

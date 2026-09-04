from __future__ import annotations

import csv
import json
import subprocess
import sys
import zlib
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_vision_expert_smoke_data.py"


def make_fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[dict]]:
    zip_path = tmp_path / "unsupervised.zip"
    manifest_path = tmp_path / "unsupervised.csv"
    qa_path = tmp_path / "smoke.jsonl"

    members: dict[str, bytes] = {}
    for index in range(4):
        members[f"asset-a/train/good/train-{index}.jpg"] = f"train-{index}".encode()

    qa_rows: list[dict] = []
    for index in range(20):
        source_label = "good" if index < 10 else "rust"
        image_path = f"asset-a/test/{source_label}/query-{index}.jpg"
        members[image_path] = f"query-{index}".encode()
        qa_rows.append(
            {
                "id": f"q{index}",
                "image": image_path,
                "asset_category": "asset-a",
                "answer": "normal" if index < 10 else "anomalous",
            }
        )

    with ZipFile(zip_path, "w") as archive:
        for image_path, data in members.items():
            archive.writestr(image_path, data)

    manifest_rows = [
        {
            "path": image_path,
            "is_dir": False,
            "size": len(data),
            "compressed_size": len(data),
            "extension": ".jpg",
            "crc32": f"{zlib.crc32(data) & 0xFFFFFFFF:08x}",
        }
        for image_path, data in members.items()
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    qa_path.write_text(
        "".join(json.dumps(row) + "\n" for row in qa_rows),
        encoding="utf-8",
    )
    return zip_path, manifest_path, qa_path, qa_rows


def run_builder(
    zip_path: Path,
    manifest_path: Path,
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
            "3",
            "--seed",
            "42",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_stale_or_partial_workspace_is_repaired_automatically(tmp_path: Path) -> None:
    zip_path, manifest_path, qa_path, _ = make_fixture(tmp_path)
    out_dir = tmp_path / "workspace"

    first = run_builder(zip_path, manifest_path, qa_path, out_dir)
    assert first.returncode == 0, first.stderr

    config_path = out_dir / "workspace_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["seed"] = 999
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (out_dir / "sentinel.txt").write_text("stale", encoding="utf-8")

    second = run_builder(zip_path, manifest_path, qa_path, out_dir)
    assert second.returncode == 0, second.stderr
    assert "rebuilding safely" in second.stdout
    assert not (out_dir / "sentinel.txt").exists()

    repaired = json.loads(config_path.read_text(encoding="utf-8"))
    assert repaired["schema_version"] == 2
    assert repaired["seed"] == 42

    manifest_rows = list(
        csv.DictReader((out_dir / "workspace_manifest.csv").open(encoding="utf-8"))
    )
    missing_artifact = out_dir / manifest_rows[0]["local_path"]
    missing_artifact.unlink()

    third = run_builder(zip_path, manifest_path, qa_path, out_dir)
    assert third.returncode == 0, third.stderr
    assert "missing or empty" in third.stdout
    assert missing_artifact.is_file()


def test_failed_rebuild_keeps_last_valid_workspace(tmp_path: Path) -> None:
    zip_path, manifest_path, qa_path, qa_rows = make_fixture(tmp_path)
    out_dir = tmp_path / "workspace"
    assert run_builder(zip_path, manifest_path, qa_path, out_dir).returncode == 0

    before = (out_dir / "workspace_config.json").read_text(encoding="utf-8")
    qa_rows[0]["image"] = "asset-a/test/good/missing.jpg"
    bad_qa = tmp_path / "bad-smoke.jsonl"
    bad_qa.write_text(
        "".join(json.dumps(row) + "\n" for row in qa_rows),
        encoding="utf-8",
    )

    failed = run_builder(zip_path, manifest_path, bad_qa, out_dir)
    assert failed.returncode == 1
    assert "source ZIP does not match" in failed.stderr
    assert (out_dir / "workspace_config.json").read_text(encoding="utf-8") == before

import csv
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from insplad_benchmark.reference_retrieval import EmbeddingProvenance


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_one_shot_plus.py"
SPEC = importlib.util.spec_from_file_location("build_one_shot_plus", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PixelEncoder:
    provenance = EmbeddingProvenance(model_id="fake/dino", model_revision="test")

    def encode(self, images):
        vectors = []
        for image in images:
            red, green, _ = image.getpixel((0, 0))
            vectors.append([float(red), float(green)])
        return np.asarray(vectors, dtype=np.float32)


def _write_fixture(tmp_path: Path):
    image_specs = {
        "asset-a/train/good/red.jpg": (250, 1, 0),
        "asset-a/train/good/green.jpg": (1, 250, 0),
        "asset-a/train/rust/anomalous.jpg": (125, 125, 0),
        "asset-a/test/good/query-red.jpg": (240, 5, 0),
        "asset-a/test/rust/query-green.jpg": (5, 240, 0),
    }
    zip_path = tmp_path / "unsup.zip"
    member_bytes = {}
    for path, color in image_specs.items():
        image_path = tmp_path / Path(path).name
        Image.new("RGB", (4, 4), color).save(image_path, format="PNG")
        member_bytes[path] = image_path.read_bytes()
    with ZipFile(zip_path, "w") as archive:
        for path, data in member_bytes.items():
            archive.writestr(path, data)

    manifest_path = tmp_path / "manifest.csv"
    fields = ["path", "is_dir", "size", "compressed_size", "extension", "crc32"]
    rows = []
    for index, (path, data) in enumerate(member_bytes.items(), 1):
        rows.append({
            "path": path,
            "is_dir": False,
            "size": len(data),
            "compressed_size": len(data),
            "extension": ".jpg",
            "crc32": f"{index:08x}",
        })
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    base_rows = [
        {
            "id": "q-red", "task": "anomaly_detection",
            "image": "asset-a/test/good/query-red.jpg", "asset_category": "asset-a",
            "asset_category_raw": "asset-a", "split": "test",
            "question": "q", "options": ["normal", "anomalous"], "answer": "normal",
            "source_label_raw": "good", "reference_image": "asset-a/train/good/green.jpg",
            "reference_split": "train", "reference_source_label_raw": "good",
        },
        {
            "id": "q-green", "task": "anomaly_detection",
            "image": "asset-a/test/rust/query-green.jpg", "asset_category": "asset-a",
            "asset_category_raw": "asset-a", "split": "test",
            "question": "q", "options": ["normal", "anomalous"], "answer": "anomalous",
            "source_label_raw": "rust", "reference_image": "asset-a/train/good/red.jpg",
            "reference_split": "train", "reference_source_label_raw": "good",
        },
    ]
    base_path = tmp_path / "fixed.jsonl"
    base_path.write_text(
        "".join(json.dumps(row) + "\n" for row in base_rows), encoding="utf-8"
    )
    return zip_path, manifest_path, base_path, base_rows


def test_builder_writes_separate_qa_manifest_smoke_and_reuses_cache(tmp_path):
    zip_path, manifest_path, base_path, base_rows = _write_fixture(tmp_path)
    original_fixed = base_path.read_bytes()
    out_qa = tmp_path / "one_shot_plus.jsonl"
    out_csv = tmp_path / "references.csv"
    smoke = tmp_path / "smoke.jsonl"
    cache = tmp_path / "embeddings.npz"

    summary = MODULE.build_artifacts(
        anomaly_manifest=manifest_path,
        unsup_zip=zip_path,
        base_one_shot_qa=base_path,
        out_qa=out_qa,
        out_manifest=out_csv,
        cache=cache,
        encoder=PixelEncoder(),
        batch_size=2,
        smoke_out=smoke,
        smoke_size=2,
        expected_query_count=2,
        expected_assets=("asset-a",),
    )

    plus = MODULE.read_jsonl(out_qa)
    assert summary["query_count"] == 2
    assert [row["id"] for row in plus] == [row["id"] for row in base_rows]
    assert [row["answer"] for row in plus] == [row["answer"] for row in base_rows]
    assert plus[0]["reference_image"].endswith("red.jpg")
    assert plus[1]["reference_image"].endswith("green.jpg")
    assert {row["retrieval_method"] for row in plus} == {"dinov2_cosine_top1"}
    assert {row["retrieval_cache_sha256"] for row in plus} == {summary["cache_sha256"]}
    assert len(summary["cache_sha256"]) == 64
    assert len(MODULE.read_jsonl(smoke)) == 2
    csv_rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert len(csv_rows) == 2
    assert {row["retrieval_cache_sha256"] for row in csv_rows} == {summary["cache_sha256"]}
    assert cache.is_file()
    assert base_path.read_bytes() == original_fixed

    class FailingEncoder(PixelEncoder):
        def encode(self, images):
            raise AssertionError("cache should satisfy every embedding")

    MODULE.build_artifacts(
        anomaly_manifest=manifest_path,
        unsup_zip=zip_path,
        base_one_shot_qa=base_path,
        out_qa=out_qa,
        out_manifest=out_csv,
        cache=cache,
        encoder=FailingEncoder(),
        batch_size=2,
        expected_query_count=2,
        expected_assets=("asset-a",),
    )


def test_builder_help_compiles_without_loading_dinov2():
    compile(SCRIPT.read_text(encoding="utf-8"), str(SCRIPT), "exec")


def test_builder_requires_immutable_model_revision():
    assert MODULE.resolve_model_revision("facebook/dinov2-base", None) == (
        "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
    )
    with pytest.raises(ValueError, match="custom --model-id requires"):
        MODULE.resolve_model_revision("custom/model", None)
    with pytest.raises(ValueError, match="40-character immutable commit SHA"):
        MODULE.resolve_model_revision("facebook/dinov2-base", "main")


def test_builder_rejects_wrong_aligned_core_asset_set(tmp_path):
    zip_path, manifest_path, base_path, _ = _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="asset set differs"):
        MODULE.build_artifacts(
            anomaly_manifest=manifest_path,
            unsup_zip=zip_path,
            base_one_shot_qa=base_path,
            out_qa=tmp_path / "plus.jsonl",
            out_manifest=tmp_path / "references.csv",
            cache=tmp_path / "embeddings.npz",
            encoder=PixelEncoder(),
            expected_query_count=2,
        )


def test_align_base_rows_follows_aligned_core_order(tmp_path):
    core = tmp_path / "aligned_core.csv"
    core.write_text("anomaly_id\nq-green\nq-red\n", encoding="utf-8")
    rows = [{"id": "q-red"}, {"id": "q-green"}, {"id": "unused"}]

    selected = MODULE.align_base_rows(rows, core)

    assert [row["id"] for row in selected] == ["q-green", "q-red"]


@pytest.mark.parametrize("collision", ["out_qa", "smoke_out", "cache"])
def test_builder_rejects_source_or_output_path_collisions(tmp_path, collision):
    base = tmp_path / "fixed.jsonl"
    kwargs = {
        "anomaly_manifest": tmp_path / "manifest.csv",
        "unsup_zip": tmp_path / "source.zip",
        "base_one_shot_qa": base,
        "out_qa": tmp_path / "plus.jsonl",
        "out_manifest": tmp_path / "references.csv",
        "cache": tmp_path / "embeddings.npz",
        "smoke_out": tmp_path / "smoke.jsonl",
        "encoder": PixelEncoder(),
    }
    if collision == "out_qa":
        kwargs["out_qa"] = base
    elif collision == "smoke_out":
        kwargs["smoke_out"] = kwargs["out_qa"]
    else:
        kwargs["cache"] = kwargs["out_manifest"]

    with pytest.raises(ValueError, match="path collision"):
        MODULE.build_artifacts(**kwargs)

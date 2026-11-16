"""Human location boxes for the defect photos that train the tool-choice LoRA (SFT set).

The automatic DINOv2 boxes of build_agent_sft_data.py could not be learned: the trained policy cropped the same
corner in 341 of 365 crops (2026-11-16). The 300 anomalous training photos therefore get manual boxes. Save, draft,
resume and revision rules follow dev_missing_location_annotation.py. The screen shows the dataset's defect type,
which the annotator needs to know what to box; it never shows a file name, a model answer or an automatic box.
Training photos only: dev240 and gold are refused.
"""
from __future__ import annotations

import base64
from collections import Counter
import io
from pathlib import Path, PurePosixPath
import re
import time
import uuid
from zipfile import ZIP_DEFLATED, ZipFile

from dev_missing_location_annotation import atomic, box_regions, clean_boxes, digest, read, sha

PROTOCOL = "sft_train_location_annotation_v1"
GT_PROTOCOL = "sft_train_location_gt_v1"
PREFIX = "sft_train_location_annotation"
RESULT_ZIP = "sft_train_location_annotation_results.zip"
FINAL = ("boxed", "unlocatable", "disputed")
TARGET_KEYS = {"core_id", "image", "sha256", "size", "asset_category", "defect_gt", "anomaly_gt", "sft_split",
               "content_fingerprint"}
# Shown on screen: Korean name of the defect and what to wrap. The rule matches the gold boxes ("wrap the defect").
DEFECTS = {
    "missing-cap": ("누락 (유리 원판이 빠짐)", "원판이 빠진 자리(연결부·드러난 금속 몸통)를 감싸세요. 절연체 줄 전체를 크게 감싸지 마세요."),
    "rust": ("녹 / 부식", "녹슨 부분을 감싸세요. 부품 전체가 녹슬었으면 그 부품을 감싸는 박스 하나를 그리세요."),
    "bird-nest": ("새 둥지", "둥지(나뭇가지·풀 뭉치)를 감싸세요."),
}
CELL_NAMES = ("Upper-left", "Upper-center", "Upper-right", "Middle-left", "Center", "Middle-right",
              "Lower-left", "Lower-center", "Lower-right")


def validate_manifest(manifest):
    if set(manifest) != {"protocol", "source", "cohort", "targets"} or manifest["protocol"] != PROTOCOL:
        raise ValueError("expected an SFT training-photo annotation manifest")
    targets = manifest["targets"]
    if not targets or len({t["core_id"] for t in targets}) != len(targets) or len({t["image"] for t in targets}) != len(targets):
        raise ValueError("empty or duplicate targets")
    for t in targets:
        if set(t) != TARGET_KEYS:
            raise ValueError("identity-only manifest required; no model outputs or automatic boxes")
        if not isinstance(t["core_id"], str) or not re.fullmatch(r"sft-[A-Za-z0-9_-]{1,76}", t["core_id"]):
            raise ValueError("only SFT training photos (sft- ids), never dev or gold")
        name = t["image"]
        if not isinstance(name, str) or "\\" in name or ":" in name or name.startswith("/") or ".." in PurePosixPath(name).parts:
            raise ValueError("unsafe image member")
        if PurePosixPath(name).parts[:4] != ("defect_supervised", t["asset_category"], "train", t["defect_gt"]):
            raise ValueError("image must be a supervised train photo of its own asset and defect")
        if t["defect_gt"] not in DEFECTS or t["anomaly_gt"] != "anomalous" or t["sft_split"] not in ("train", "val"):
            raise ValueError("only anomalous photos of a known defect type")
        if len(t["size"]) != 2 or any(type(x) is not int or x < 3 for x in t["size"]):
            raise ValueError("invalid image dimensions")
        if len(t["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in t["sha256"]):
            raise ValueError("invalid image SHA")
    return digest(manifest)


def prepare(manifest, archive, out, cache=None, progress=None):
    """Lock the manifest in ``out`` and make verified copies of the photos in ``cache`` (default ``out/_images``)."""
    from PIL import Image
    out, archive = Path(out), Path(archive)
    cache = Path(cache) if cache else out / "_images"
    validate_manifest(manifest)
    if not out.name.startswith(PREFIX) or archive.resolve().is_relative_to(out.resolve()):
        raise ValueError("separate SFT annotation folder required")
    lock = out / "manifest.json"
    if lock.exists() and read(lock) != manifest:
        raise ValueError("대상 사진이 바뀌었습니다. 기존 파일을 덮어쓰지 말고 새 폴더를 사용하세요.")
    if not lock.exists() and out.exists() and any(p.name != "_images" for p in out.iterdir()):
        raise ValueError("알 수 없는 기존 폴더입니다. 새 폴더를 사용하세요.")
    paths, missing = {}, []
    for t in manifest["targets"]:
        p = cache / (sha(t["core_id"].encode()) + PurePosixPath(t["image"]).suffix)
        paths[t["core_id"]] = p
        if not (p.exists() and sha(p.read_bytes()) == t["sha256"]):
            missing.append(t)
    payloads = {}
    if missing:                                   # check every original byte before writing anything
        with ZipFile(archive) as z:
            for i, t in enumerate(missing, 1):
                b = z.read(t["image"])
                if sha(b) != t["sha256"]:
                    raise ValueError("원본 사진 SHA가 다릅니다: " + t["core_id"])
                with Image.open(io.BytesIO(b)) as im:
                    if list(im.size) != t["size"] or im.getexif().get(274, 1) != 1:
                        raise ValueError("사진 크기/EXIF 방향이 바뀌었습니다.")
                payloads[t["core_id"]] = b
                if progress and (i % 50 == 0 or i == len(missing)):
                    progress(i, len(missing))
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    atomic(lock, manifest)
    for core, b in payloads.items():
        paths[core].write_bytes(b)
    return Session(manifest, paths, out)


class Session:
    def __init__(self, manifest, paths, out):
        self.manifest, self.identity = manifest, validate_manifest(manifest)
        self.targets, self.paths, self.out = manifest["targets"], paths, Path(out)
        self.path = self.out / "annotations.json"
        self._state()
        self._status()

    def _state(self):
        state = read(self.path) if self.path.exists() else {"protocol": PROTOCOL, "manifest_identity": self.identity, "records": {}}
        if set(state) != {"protocol", "manifest_identity", "records"} or state["protocol"] != PROTOCOL \
                or state["manifest_identity"] != self.identity:
            raise ValueError("표시 파일의 대상 목록이 다릅니다.")
        by_id = {t["core_id"]: t for t in self.targets}
        if not isinstance(state["records"], dict) or set(state["records"]) - set(by_id):
            raise ValueError("알 수 없는 표시 ID입니다.")
        for core, r in state["records"].items():
            if set(r) != {"status", "boxes_xywh", "note", "revision", "image_sha256", "image_size", "saved_at"}:
                raise ValueError("잘못된 표시 기록 형식입니다.")
            if r["image_sha256"] != by_id[core]["sha256"] or r["image_size"] != by_id[core]["size"] \
                    or r["status"] not in ("draft", *FINAL):
                raise ValueError("표시의 사진/상태가 다릅니다.")
            if type(r["revision"]) is not int or r["revision"] < 1:
                raise ValueError("잘못된 표시 버전입니다.")
            if r["status"] in ("draft", "boxed"):
                clean_boxes(r["boxes_xywh"], r["image_size"], r["status"] == "draft")
            elif r["boxes_xywh"] or not r["note"].strip():
                raise ValueError("위치 미확정에는 사유가 필요합니다.")
        return state

    def progress(self):
        records = self._state()["records"]
        counts = Counter(r["status"] for r in records.values())
        first = next((i for i, t in enumerate(self.targets) if records.get(t["core_id"], {}).get("status") not in FINAL),
                     len(self.targets))
        return {"total": len(self.targets), "done": sum(counts[x] for x in FINAL), "boxed": counts["boxed"],
                "unlocatable": counts["unlocatable"], "disputed": counts["disputed"], "drafts": counts["draft"],
                "first_todo": first}

    def _status(self):
        progress = self.progress()
        atomic(self.out / "run_status.json", {"protocol": PROTOCOL, "phase": "human_annotation",
                                              "complete": progress["done"] == progress["total"], **progress,
                                              "neural_calls": 0, "gold_changed": False, "dev_changed": False})

    def item(self, index):
        from PIL import Image
        if type(index) is not int:
            raise ValueError("invalid image index")
        index = max(0, min(index, len(self.targets) - 1))
        target = self.targets[index]
        b = self.paths[target["core_id"]].read_bytes()
        if sha(b) != target["sha256"]:
            raise ValueError("사진이 바뀌었습니다.")
        with Image.open(io.BytesIO(b)) as im:
            mime = Image.MIME[im.format]
        rec = self._state()["records"].get(target["core_id"], {})
        label, guide = DEFECTS[target["defect_gt"]]
        # The dataset's defect type is shown; no core ID, file name, model answer, automatic box or grid answer.
        return {"index": index, **self.progress(), "image_url": "data:" + mime + ";base64," + base64.b64encode(b).decode(),
                "width": target["size"][0], "height": target["size"][1], "defect_label": label, "guide": guide,
                "boxes": rec.get("boxes_xywh", []), "status": rec.get("status", ""), "note": rec.get("note", ""),
                "revision": rec.get("revision", 0), "saved_at": rec.get("saved_at", "")}

    def save(self, index, boxes, status, note="", revision=0):
        if type(index) is not int or not 0 <= index < len(self.targets):
            raise ValueError("invalid image index")
        if status not in ("draft", *FINAL) or not isinstance(note, str):
            raise ValueError("invalid annotation state")
        if len(note) > 4000:
            raise ValueError("메모가 너무 깁니다.")
        target, state = self.targets[index], self._state()
        old = state["records"].get(target["core_id"])
        current = old["revision"] if old else 0
        if type(revision) is not int or revision != current:
            raise ValueError("다른 화면에서 표시가 변경됐습니다. 다시 불러오세요.")
        if status in ("draft", "boxed"):
            clean = clean_boxes(boxes, target["size"], status == "draft")
        else:
            if boxes or not note.strip():
                raise ValueError("위치 판단 불가/라벨 이견에는 박스 없이 사유를 입력하세요.")
            clean = []
        record = {"status": status, "boxes_xywh": clean, "note": note.strip(), "revision": current + 1,
                  "image_sha256": target["sha256"], "image_size": target["size"],
                  "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        if old:
            atomic(self.out / "history" / target["core_id"] / (str(current) + "_" + uuid.uuid4().hex + ".json"), old)
        atomic(self.path, {**state, "records": {**state["records"], target["core_id"]: record}})
        self._status()
        return {"revision": record["revision"], "status": status, "saved_at": record["saved_at"], **self.progress()}

    def export(self):
        records, rows = self._state()["records"], []
        for target in self.targets:
            rec = records.get(target["core_id"], {})
            known = rec.get("status") == "boxed"
            boxes = rec["boxes_xywh"] if known else []
            rows.append({"core_id": target["core_id"], "image": target["image"], "image_sha256": target["sha256"],
                         "asset_category": target["asset_category"], "defect_gt": target["defect_gt"],
                         "anomaly_gt": target["anomaly_gt"], "sft_split": target["sft_split"], "protocol": GT_PROTOCOL,
                         "regions_gt": box_regions(boxes) if known else None, "normalized_bboxes_xywh": boxes,
                         "location_gt_known": known, "annotation_status": rec.get("status", "pending"),
                         "label_basis": "human_train_box_centers" if known else "human_train_spatial_unknown",
                         "note": rec.get("note", ""), "annotation_revision": rec.get("revision", 0)})
        atomic(self.out / "sft_train_location_gt.json", rows)
        known = [r for r in rows if r["location_gt_known"]]
        cells = Counter(c for r in known for c in r["regions_gt"])
        centre_only = sum(r["regions_gt"] == [5] for r in known)
        progress = self.progress()
        summary = {**progress, "complete": progress["done"] == progress["total"],
                   "positive_spatial_support": len(known), "unknown_spatial_support": len(rows) - len(known),
                   "boxed_by_defect": dict(sorted(Counter(r["asset_category"] + ":" + r["defect_gt"] for r in known).items())),
                   # How concentrated the answers are; "always centre" scores 72/133 on the gold location set.
                   "region_counts": {CELL_NAMES[c - 1]: cells[c] for c in range(1, 10)},
                   "centre_only_images": centre_only, "multi_box_images": sum(len(r["normalized_bboxes_xywh"]) > 1 for r in known),
                   "manifest_identity": self.identity,
                   "annotations_sha256": sha(self.path.read_bytes()) if self.path.exists() else None,
                   "model_predictions_shown": False, "automatic_boxes_shown": False, "gold_changed": False,
                   "dev_changed": False, "neural_calls": 0}
        atomic(self.out / "annotation_summary.json", summary)
        self._status()
        result = self.out.parent / RESULT_ZIP
        with ZipFile(result, "w", ZIP_DEFLATED) as z:
            for p in sorted(self.out.rglob("*")):
                if p.is_file() and "_images" not in p.relative_to(self.out).parts:
                    z.write(p, "annotation/" + p.relative_to(self.out).as_posix())
        return result, summary


def register_colab(session):
    from google.colab import output
    from IPython.display import HTML, JSON, display
    output.register_callback("sftloc.item", lambda i: JSON(session.item(i)))
    output.register_callback("sftloc.progress", lambda: JSON(session.progress()))
    output.register_callback("sftloc.save", lambda i, b, s, n, v: JSON(session.save(i, b, s, n, v)))
    display(HTML(Path(__file__).with_name("sft_train_location_ui.html").read_text(encoding="utf-8")))

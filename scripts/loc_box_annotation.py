#!/usr/bin/env python3
"""Defect-location box annotation for gold-v2 anomalous images (Colab tool backend).

Only 20 of the 139 gold-v2 anomalous images have a location label (InsPLAD-seg masks, lightning-rod-suspension
rust). This module lists the remaining anomalous images, serves them to a small browser UI inside Colab, and
stores human-drawn boxes as normalized ``[x, y, w, h]`` in a JSON file on Drive, saved after every image so the
work can be resumed. ``scripts/build_gold_v2_loc_mmad.py`` turns the boxes into MMAD-style location questions.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import uuid
from pathlib import Path
from typing import Mapping, Sequence

Q2 = "defect_classification"
LOCALIZATION = "defect_localization"
STATUSES = ("boxed", "unlocatable")


def anomalous_images(rows: Sequence[Mapping]) -> list[dict]:
    """One record per anomalous gold image (from the Q2 rows), sorted by asset then core id."""
    out: dict[str, dict] = {}
    for row in rows:
        if row.get("question_type") != Q2 or str(row.get("defect_gt")) == "normal":
            continue
        core = str(row["core_id"])
        out[core] = {"core_id": core, "image": str(row["image"]), "asset_category": str(row["asset_category"]),
                     "defect_gt": str(row["defect_gt"])}
    return sorted(out.values(), key=lambda r: (r["asset_category"], r["defect_gt"], r["core_id"]))


def seg_boxes_by_core(loc_rows: Sequence[Mapping]) -> dict[str, list[list[float]]]:
    """Existing InsPLAD-seg boxes for anomalous images (normal images have empty masks and are skipped)."""
    out: dict[str, list[list[float]]] = {}
    for row in loc_rows:
        gt = row.get("localization_gt") or {}
        boxes = gt.get("normalized_bboxes_xywh") or []
        if gt.get("empty_mask") or not boxes:
            continue
        out[str(row["core_id"])] = [[float(v) for v in b] for b in boxes]
    return out


def annotation_targets(rows: Sequence[Mapping], loc_rows: Sequence[Mapping], include_seg: bool = False) -> list[dict]:
    seg = seg_boxes_by_core(loc_rows)
    targets = []
    for rec in anomalous_images(rows):
        if rec["core_id"] in seg:
            if not include_seg:
                continue
            rec = {**rec, "seg_boxes": seg[rec["core_id"]]}
        targets.append(rec)
    return targets


def validate_boxes(boxes: Sequence[Sequence[float]]) -> list[list[float]]:
    clean = []
    for box in boxes:
        if len(box) != 4:
            raise ValueError(f"box must be [x, y, w, h], got {box!r}")
        x, y, w, h = (float(v) for v in box)
        x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
        w, h = min(max(w, 0.0), 1.0 - x), min(max(h, 0.0), 1.0 - y)
        if w <= 0.002 or h <= 0.002:
            continue  # accidental click, not a box
        clean.append([round(x, 6), round(y, 6), round(w, 6), round(h, 6)])
    return clean


class AnnotationStore:
    """JSON file {core_id: record}; every save rewrites atomically so a Colab disconnect loses nothing."""

    def __init__(self, path: Path, annotator: str = "") -> None:
        self.path = Path(path)
        self.annotator = annotator
        self.data: dict[str, dict] = {}
        if self.path.is_file() and self.path.stat().st_size:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, target: Mapping, boxes: Sequence[Sequence[float]], status: str, note: str = "",
             image_size: Sequence[int] = ()) -> dict:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        clean = validate_boxes(boxes)
        if status == "boxed" and not clean:
            raise ValueError("draw at least one box, or mark the image as unlocatable")
        record = {
            "core_id": target["core_id"], "image": target["image"], "asset_category": target["asset_category"],
            "defect_gt": target["defect_gt"], "status": status, "boxes_xywh": clean if status == "boxed" else [],
            "note": note, "annotator": self.annotator, "image_size": list(image_size),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self.data[str(target["core_id"])] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)
        return record

    def done(self) -> set[str]:
        return set(self.data)


def encode_image(path: Path, max_side: int = 1600) -> tuple[str, int, int]:
    """JPEG data URL for the browser (resized for speed; boxes are stored normalized so size does not matter)."""
    from PIL import Image

    with Image.open(path) as handle:
        img = handle.convert("RGB")
    width, height = img.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        img = img.resize((round(width * scale), round(height * scale)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"), width, height


class AnnotationSession:
    """Backend for the Colab UI: ``item(i)`` returns what the page shows, ``save(i, ...)`` stores the boxes."""

    def __init__(self, targets: Sequence[Mapping], image_paths: Mapping[str, Path], store: AnnotationStore,
                 drafts: Mapping[str, Sequence[Sequence[float]]] | None = None) -> None:
        self.targets = list(targets)
        self.image_paths = image_paths
        self.store = store
        self.drafts = dict(drafts or {})

    def progress(self) -> dict:
        done = self.store.done()
        ids = [t["core_id"] for t in self.targets]
        return {"total": len(ids), "done": sum(c in done for c in ids),
                "first_todo": next((i for i, c in enumerate(ids) if c not in done), len(ids))}

    def item(self, index: int) -> dict:
        index = max(0, min(int(index), len(self.targets) - 1))
        target = self.targets[index]
        url, width, height = encode_image(self.image_paths[target["image"]])
        saved = self.store.data.get(target["core_id"], {})
        return {
            "index": index, **self.progress(), "core_id": target["core_id"],
            "asset_category": target["asset_category"], "defect_gt": target["defect_gt"],
            "image_url": url, "width": width, "height": height,
            "boxes": saved.get("boxes_xywh") or target.get("seg_boxes") or [],
            "status": saved.get("status", ""), "note": saved.get("note", ""),
            "draft": [list(b) for b in self.drafts.get(target["core_id"], [])],
        }

    def save(self, index: int, boxes: Sequence[Sequence[float]], status: str, note: str = "") -> dict:
        target = self.targets[int(index)]
        record = self.store.save(target, boxes, status, note)
        return {"saved": record["core_id"], "status": record["status"], "n_boxes": len(record["boxes_xywh"]),
                **self.progress()}


def draft_boxes_from_manifest(manifest_rows: Sequence[Mapping[str, str]], expert: str,
                              sizes: Mapping[str, tuple[int, int]]) -> dict[str, list[list[float]]]:
    """Expert ROI as a normalized draft box (shown dashed; never saved unless the annotator adopts it)."""
    out = {}
    for row in manifest_rows:
        if row.get("expert") != expert or str(row.get("query_id")) not in sizes:
            continue
        w, h = sizes[str(row["query_id"])]
        left, top = float(row["roi_left"]) / w, float(row["roi_top"]) / h
        right, bottom = float(row["roi_right"]) / w, float(row["roi_bottom"]) / h
        out[str(row["query_id"])] = [[round(v, 6) for v in (left, top, right - left, bottom - top)]]
    return out


UI_HTML = r"""
<div id="lb" style="font-family:sans-serif;max-width:1100px">
  <div id="lb-head" style="margin:6px 0;font-size:15px"></div>
  <div style="margin:6px 0;font-size:13px;color:#555">
    마우스로 끌어서 결함을 감싸는 박스를 그리세요 (여러 개 가능). 결함이 여러 곳이면 각각 그립니다.
    단축키: <b>Enter</b> 저장 후 다음 · <b>Z</b> 마지막 박스 지우기 · <b>←/→</b> 이전/다음(저장 안 함)
  </div>
  <canvas id="lb-canvas" style="border:1px solid #888;cursor:crosshair;max-width:100%"></canvas>
  <div style="margin:8px 0;display:flex;gap:6px;flex-wrap:wrap;align-items:center">
    <button id="lb-prev">◀ 이전</button>
    <button id="lb-undo">마지막 박스 지우기 (Z)</button>
    <button id="lb-clear">전체 지우기</button>
    <button id="lb-draft">초안 박스 채택</button>
    <button id="lb-save" style="font-weight:bold">저장 후 다음 (Enter)</button>
    <button id="lb-unloc">위치 표시 불가로 저장</button>
    <button id="lb-next">다음 ▶ (저장 안 함)</button>
    <button id="lb-todo">첫 미완료로 이동</button>
    <input id="lb-note" placeholder="메모 (선택)" style="flex:1;min-width:160px">
  </div>
  <div id="lb-msg" style="font-size:13px;min-height:18px"></div>
</div>
<script>
(async () => {
  const call = async (name, args) => {
    const r = await google.colab.kernel.invokeFunction(name, args, {});
    return r.data['application/json'];
  };
  const cv = document.getElementById('lb-canvas'), ctx = cv.getContext('2d');
  const head = document.getElementById('lb-head'), msg = document.getElementById('lb-msg');
  const note = document.getElementById('lb-note');
  let item = null, boxes = [], img = new Image(), drag = null, scale = 1, busy = false, lastMsg = '';
  const COLORS = ['#ff2d2d', '#00c853', '#2979ff', '#ff9100', '#d500f9'];
  function draw() {
    if (!item) return;
    ctx.drawImage(img, 0, 0, cv.width, cv.height);
    ctx.setLineDash([8, 6]); ctx.lineWidth = 2; ctx.strokeStyle = '#ffd600';
    for (const b of item.draft) ctx.strokeRect(b[0]*cv.width, b[1]*cv.height, b[2]*cv.width, b[3]*cv.height);
    ctx.setLineDash([]); ctx.lineWidth = 3;
    boxes.forEach((b, i) => { ctx.strokeStyle = COLORS[i % COLORS.length];
      ctx.strokeRect(b[0]*cv.width, b[1]*cv.height, b[2]*cv.width, b[3]*cv.height); });
    if (drag && drag.cur) { ctx.strokeStyle = '#ffffff';
      const [x0, y0] = drag.start, [x1, y1] = drag.cur;
      ctx.strokeRect(Math.min(x0,x1), Math.min(y0,y1), Math.abs(x1-x0), Math.abs(y1-y0)); }
  }
  function header() {
    const st = item.status ? ` · 저장됨(${item.status === 'boxed' ? '박스' : '표시 불가'})` : ' · 미완료';
    head.innerHTML = `<b>${item.index + 1} / ${item.total}</b> (완료 ${item.done}) · ${item.asset_category} · ` +
      `결함: <b>${item.defect_gt}</b> · ${item.core_id}${st}`;
  }
  async function load(i) {
    if (busy) return;
    busy = true;
    msg.textContent = '불러오는 중...';
    item = await call('lb.item', [i]);
    boxes = item.boxes.map(b => b.slice()); note.value = item.note || '';
    img = new Image();
    img.onload = () => {
      const avail = (cv.parentElement && cv.parentElement.clientWidth) || window.innerWidth || 1060;
      const maxW = Math.max(320, Math.min(1060, avail - 20));
      scale = Math.min(1, maxW / img.naturalWidth, 760 / img.naturalHeight);
      cv.width = Math.round(img.naturalWidth * scale); cv.height = Math.round(img.naturalHeight * scale);
      header(); draw();
      msg.textContent = [lastMsg, item.draft.length ? '노란 점선 = expert 초안 (정답 아님)' : ''].filter(Boolean).join(' · ');
      lastMsg = '';
      busy = false;
    };
    img.onerror = () => { msg.textContent = '이미지를 불러오지 못했습니다'; busy = false; };
    img.src = item.image_url;
  }
  const pos = e => { const r = cv.getBoundingClientRect();
    return [(e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height]; };
  cv.onmousedown = e => { drag = {start: pos(e), cur: null}; };
  cv.onmousemove = e => { if (drag) { drag.cur = pos(e); draw(); } };
  window.addEventListener('mouseup', e => {
    if (!drag) return;
    const [x0, y0] = drag.start, [x1, y1] = drag.cur || drag.start; drag = null;
    const w = Math.abs(x1 - x0), h = Math.abs(y1 - y0);
    if (w > 4 && h > 4) boxes.push([Math.min(x0,x1)/cv.width, Math.min(y0,y1)/cv.height, w/cv.width, h/cv.height]);
    draw();
  });
  async function save(status) {
    if (busy) return;
    if (status === 'boxed' && !boxes.length) { msg.textContent = '박스를 하나 이상 그리거나 "위치 표시 불가"를 누르세요.'; return; }
    busy = true;
    msg.textContent = '저장 중...';
    let r;
    try {
      r = await call('lb.save', [item.index, boxes, status, note.value]);
    } catch (err) { msg.textContent = '저장 실패: ' + err; busy = false; return; }
    busy = false;
    const done = `저장됨: ${r.saved} (${r.status}, 박스 ${r.n_boxes}개) · 완료 ${r.done}/${r.total}` +
      (r.done >= r.total ? ' · 전부 완료!' : '');
    lastMsg = done;
    await load(Math.min(item.index + 1, item.total - 1));
  }
  document.getElementById('lb-prev').onclick = () => load(item.index - 1);
  document.getElementById('lb-next').onclick = () => load(item.index + 1);
  document.getElementById('lb-undo').onclick = () => { boxes.pop(); draw(); };
  document.getElementById('lb-clear').onclick = () => { boxes = []; draw(); };
  document.getElementById('lb-draft').onclick = () => { boxes = boxes.concat(item.draft.map(b => b.slice())); draw(); };
  document.getElementById('lb-save').onclick = () => save('boxed');
  document.getElementById('lb-unloc').onclick = () => save('unlocatable');
  document.getElementById('lb-todo').onclick = () => load(item.first_todo >= item.total ? item.total - 1 : item.first_todo);
  document.addEventListener('keydown', e => {
    if (e.target === note) return;
    if (e.key === 'Enter') { e.preventDefault(); save('boxed'); }
    else if (e.key === 'z' || e.key === 'Z') { boxes.pop(); draw(); }
    else if (e.key === 'ArrowLeft') load(item.index - 1);
    else if (e.key === 'ArrowRight') load(item.index + 1);
  });
  const p = await call('lb.progress', []);
  await load(p.first_todo >= p.total ? 0 : p.first_todo);
})();
</script>
"""


def register_colab(session: AnnotationSession) -> None:
    """Expose the session to the page via google.colab callbacks, then render the UI."""
    from google.colab import output  # type: ignore
    from IPython.display import HTML, JSON, display

    output.register_callback("lb.item", lambda i: JSON(session.item(i)))
    output.register_callback("lb.save", lambda i, boxes, status, note="": JSON(session.save(i, boxes, status, note)))
    output.register_callback("lb.progress", lambda: JSON(session.progress()))
    display(HTML(UI_HTML))

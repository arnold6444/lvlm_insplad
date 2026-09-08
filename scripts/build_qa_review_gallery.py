#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import textwrap
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageFont


def read_jsonl(path: Path | None) -> list[dict]:
    if path is None or not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def question_html(row: dict) -> str:
    options = row.get("options") or {}
    option_lines = []
    for key, value in options.items():
        marker = " ← GT" if key == row.get("answer") else ""
        option_lines.append(
            f"<li><code>{html.escape(str(key))}</code> "
            f"{html.escape(str(value))}{html.escape(marker)}</li>"
        )
    options_html = "<ul>" + "".join(option_lines) + "</ul>" if option_lines else ""
    return (
        f"<div class='qa'>"
        f"<div class='qtype'>{html.escape(str(row.get('question_type', '')))}</div>"
        f"<div class='question'>{html.escape(str(row.get('question', '')))}</div>"
        f"{options_html}"
        f"<div class='answer'>GT: {html.escape(str(row.get('answer_text', row.get('answer', ''))))}</div>"
        f"</div>"
    )


def load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_thumbnail(
    raw: bytes,
    *,
    normalized_bboxes: list[list[float]] | None,
    max_width: int = 760,
) -> tuple[bytes, Image.Image, int, int]:
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    original_w, original_h = image.size

    if normalized_bboxes:
        draw = ImageDraw.Draw(image)
        line_width = max(3, round(min(original_w, original_h) * 0.006))
        for x, y, w, h in normalized_bboxes:
            xy = (
                round(float(x) * original_w),
                round(float(y) * original_h),
                round((float(x) + float(w)) * original_w),
                round((float(y) + float(h)) * original_h),
            )
            draw.rectangle(xy, outline="red", width=line_width)

    preview = image.copy()
    if preview.width > max_width:
        ratio = max_width / preview.width
        preview = preview.resize((max_width, max(1, round(preview.height * ratio))))

    out = io.BytesIO()
    preview.save(out, format="JPEG", quality=82, optimize=True)
    return out.getvalue(), preview, original_w, original_h


def review_text_lines(row: dict, qa_rows: list[dict], q3_region: str) -> list[str]:
    lines = [
        f"core_id: {row.get('core_id', '')}",
        f"asset: {row.get('asset_category', '')}",
        f"GT: anomaly={row.get('anomaly_gt', '')} | defect={row.get('defect_gt', '')}",
    ]
    if q3_region:
        lines.append(f"Q3 localization GT: {q3_region}")
    else:
        lines.append("Q3 localization GT: not available")
    lines.append("")
    for q in qa_rows:
        qtype = str(q.get("question_type", ""))
        lines.append(qtype)
        lines.append(str(q.get("question", "")))
        for key, value in (q.get("options") or {}).items():
            marker = "  <-- GT" if key == q.get("answer") else ""
            lines.append(f"{key}. {value}{marker}")
        lines.append(f"GT answer: {q.get('answer_text', q.get('answer', ''))}")
        lines.append("")
    return lines


def make_review_card(preview: Image.Image, lines: list[str], out_path: Path) -> None:
    card_width = 900
    font = load_font(19)
    small_font = load_font(17)
    margin = 24
    image = preview.copy()
    if image.width != card_width - 2 * margin:
        ratio = (card_width - 2 * margin) / image.width
        image = image.resize((card_width - 2 * margin, max(1, round(image.height * ratio))))

    wrapped: list[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(line, width=88, break_long_words=False) or [""])

    line_h = 27
    text_h = max(1, len(wrapped)) * line_h + 20
    card_h = margin + image.height + 18 + text_h + margin
    card = Image.new("RGB", (card_width, card_h), "white")
    card.paste(image, (margin, margin))
    draw = ImageDraw.Draw(card)
    y = margin + image.height + 18
    for line in wrapped:
        active_font = font if (
            line.startswith("core_id:")
            or line.startswith("asset:")
            or line.startswith("GT:")
            or line in {"anomaly_detection", "defect_classification", "defect_localization"}
        ) else small_font
        draw.text((margin, y), line, fill="black", font=active_font)
        y += line_h
    card.save(out_path, format="JPEG", quality=88, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a visual QA review gallery from the fixed InsPLAD QA subset."
    )
    parser.add_argument("--supervised-zip", type=Path, required=True)
    parser.add_argument("--master-images", type=Path, required=True)
    parser.add_argument("--master-qa", type=Path, required=True)
    parser.add_argument("--localization-images", type=Path)
    parser.add_argument("--localization-qa", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    master_images = read_jsonl(args.master_images)
    master_qa = read_jsonl(args.master_qa)
    localization_images = read_jsonl(args.localization_images)
    localization_qa = read_jsonl(args.localization_qa)

    if not master_images:
        raise ValueError("master images are empty")

    qa_by_core: dict[str, list[dict]] = defaultdict(list)
    for row in master_qa:
        qa_by_core[str(row["core_id"])].append(row)
    for row in localization_qa:
        if row.get("question_type") == "defect_localization":
            qa_by_core[str(row["core_id"])].append(row)

    loc_by_core = {str(row["core_id"]): row for row in localization_images}

    review_dir = args.out_dir / "review_images"
    cards_dir = args.out_dir / "review_cards"
    review_dir.mkdir(parents=True, exist_ok=True)
    cards_dir.mkdir(parents=True, exist_ok=True)
    gallery_path = args.out_dir / "review_gallery.html"
    index_path = args.out_dir / "review_index.csv"

    cards: list[str] = []
    index_rows: list[dict] = []

    with ZipFile(args.supervised_zip) as zf:
        for idx, row in enumerate(master_images, 1):
            core_id = str(row["core_id"])
            image_member = str(row["image"])
            raw = zf.read(image_member)
            loc = loc_by_core.get(core_id)
            bbox_list = None
            q3_region = ""
            if loc:
                gt = loc.get("localization_gt") or {}
                bbox_list = gt.get("normalized_bboxes_xywh") or []
                q3_region = str(gt.get("coarse_region", ""))

            thumb, preview, width, height = make_thumbnail(
                raw,
                normalized_bboxes=bbox_list,
            )
            image_name = f"{idx:03d}_{core_id}.jpg"
            image_path = review_dir / image_name
            image_path.write_bytes(thumb)

            qa_rows = sorted(
                qa_by_core.get(core_id, []),
                key=lambda x: {
                    "anomaly_detection": 1,
                    "defect_classification": 2,
                    "defect_localization": 3,
                }.get(str(x.get("question_type")), 99),
            )
            card_path = cards_dir / image_name
            make_review_card(
                preview,
                review_text_lines(row, qa_rows, q3_region),
                card_path,
            )

            encoded = base64.b64encode(thumb).decode("ascii")
            qa_html = "".join(question_html(q) for q in qa_rows)

            asset = str(row.get("asset_category", ""))
            anomaly = str(row.get("anomaly_gt", ""))
            defect = str(row.get("defect_gt", ""))
            q3_badge = (
                f"<span class='badge'>Q3: {html.escape(q3_region)}</span>"
                if loc
                else "<span class='badge muted'>Q3 GT 없음</span>"
            )

            cards.append(
                f"""
                <article class="card" data-asset="{html.escape(asset)}"
                         data-defect="{html.escape(defect)}"
                         data-q3="{'yes' if loc else 'no'}">
                  <div class="num">#{idx:03d}</div>
                  <img loading="lazy" src="data:image/jpeg;base64,{encoded}"
                       alt="{html.escape(core_id)}">
                  <div class="meta">
                    <code>{html.escape(core_id)}</code><br>
                    asset: {html.escape(asset)} ·
                    anomaly: {html.escape(anomaly)} ·
                    defect: {html.escape(defect)}<br>
                    {q3_badge}
                  </div>
                  {qa_html}
                </article>
                """
            )

            index_rows.append(
                {
                    "index": idx,
                    "core_id": core_id,
                    "asset_category": asset,
                    "anomaly_gt": anomaly,
                    "defect_gt": defect,
                    "q3_available": bool(loc),
                    "q3_region": q3_region,
                    "review_image": str(image_path.relative_to(args.out_dir)),
                    "review_card": str(card_path.relative_to(args.out_dir)),
                    "source_image": image_member,
                    "width": width,
                    "height": height,
                }
            )

    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)

    assets = sorted({r["asset_category"] for r in index_rows})
    defects = sorted({r["defect_gt"] for r in index_rows})

    page = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>InsPLAD QA Visual Review</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; background: #f6f6f6; }}
h1 {{ margin-bottom: 6px; }}
.controls {{ position: sticky; top: 0; background: #fff; padding: 12px; border: 1px solid #ddd; z-index: 10; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(420px,1fr)); gap: 18px; margin-top: 18px; }}
.card {{ background: white; border: 1px solid #ddd; border-radius: 10px; padding: 14px; }}
.card img {{ width: 100%; height: auto; border-radius: 6px; }}
.num {{ font-size: 18px; font-weight: 700; margin-bottom: 8px; }}
.meta {{ line-height: 1.6; margin: 10px 0; }}
.qa {{ border-top: 1px solid #eee; padding-top: 10px; margin-top: 10px; }}
.qtype {{ font-weight: 700; }}
.question {{ margin: 6px 0; }}
.answer {{ font-weight: 700; }}
.badge {{ display: inline-block; padding: 2px 7px; border: 1px solid #aaa; border-radius: 10px; }}
.muted {{ opacity: .65; }}
ul {{ margin-top: 6px; padding-left: 22px; }}
code {{ word-break: break-all; }}
</style>
</head>
<body>
<h1>InsPLAD QA Visual Review</h1>
<p>총 {len(index_rows)}장. Q1/Q2는 source-label gold, Q3는 실제 segmentation 매칭 이미지에만 표시됩니다.
Q3 bbox가 있는 이미지는 defect localization 영역을 박스로 표시합니다.</p>
<div class="controls">
  asset:
  <select id="asset">
    <option value="">전체</option>
    {''.join(f'<option>{html.escape(a)}</option>' for a in assets)}
  </select>
  defect:
  <select id="defect">
    <option value="">전체</option>
    {''.join(f'<option>{html.escape(d)}</option>' for d in defects)}
  </select>
  Q3:
  <select id="q3">
    <option value="">전체</option>
    <option value="yes">있음</option>
    <option value="no">없음</option>
  </select>
  <span id="visible"></span>
</div>
<div class="grid">
{''.join(cards)}
</div>
<script>
const cards=[...document.querySelectorAll('.card')];
const asset=document.getElementById('asset');
const defect=document.getElementById('defect');
const q3=document.getElementById('q3');
const visible=document.getElementById('visible');
function apply(){{
  let n=0;
  cards.forEach(c=>{{
    const ok=(!asset.value||c.dataset.asset===asset.value)
      &&(!defect.value||c.dataset.defect===defect.value)
      &&(!q3.value||c.dataset.q3===q3.value);
    c.style.display=ok?'block':'none';
    if(ok)n++;
  }});
  visible.textContent=` · 표시 ${{n}} / ${{cards.length}}`;
}}
[asset,defect,q3].forEach(x=>x.addEventListener('change',apply));
apply();
</script>
</body>
</html>
"""
    gallery_path.write_text(page, encoding="utf-8")

    print(json.dumps({
        "images": len(index_rows),
        "q3_images": sum(bool(r["q3_available"]) for r in index_rows),
        "gallery": str(gallery_path),
        "review_images": str(review_dir),
        "review_cards": str(cards_dir),
        "index_csv": str(index_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

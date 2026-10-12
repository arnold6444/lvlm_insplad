"""Blind paired physical-part annotation for a small, dev-only crop diagnosis."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import run_dev_prompt_diagnostic as diag
from loc_box_annotation import encode_image


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def box_pixels(box, size):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("draw one part box on each image")
    x, y, w, h = map(float, box)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or min(x, y) < 0 or min(w, h) <= 0 or x+w > 1.000001 or y+h > 1.000001:
        raise ValueError("invalid normalized [x,y,width,height] box")
    width, height = size
    pixels = (round(x*width), round(y*height), min(width, round((x+w)*width)), min(height, round((y+h)*height)))
    if pixels[2]-pixels[0] < 2 or pixels[3]-pixels[1] < 2:
        raise ValueError("box is too small")
    return pixels


def prepare(dev_qa, refs_csv, sup_zip, unsup_zip, out_dir, n_images=40, seed=42):
    from PIL import Image

    root = Path(out_dir)
    if not root.name.startswith("gold_v2_part_crop_diagnostic"):
        raise ValueError("use a separate gold_v2_part_crop_diagnostic* directory")
    rows = diag.select_subset(diag.agent.read_jsonl(Path(dev_qa)), n_images, seed)
    with Path(refs_csv).open(encoding="utf-8-sig", newline="") as h:
        refs = {r["image"]: r["retrieved_diffdate"] for r in csv.DictReader(h)}
    if any(not refs.get(r["image"]) for r in rows):
        raise ValueError("diffdate reference missing")
    identity = diag.digest({"rows": rows, "seed": seed, "refs": {r["image"]: refs[r["image"]] for r in rows}})
    meta_path = root / "prepare_metadata.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8"))["identity"] != identity:
        raise ValueError("different dev subset/references: use a new output folder")
    targets = diag.agent.base.prepare_images(rows, Path(sup_zip), root / "_images" / "target")
    normals = {r: diag.agent.base.safe_extract(Path(unsup_zip), r, root / "_images" / "reference")
               for r in sorted({refs[q["image"]] for q in rows})}
    pairs = []
    for q in rows[::2]:
        record = {"core_id": q["core_id"], "asset_category": q["asset_category"]}
        for role, member, path in (("target", q["image"], targets[q["image"]]),
                                   ("reference", refs[q["image"]], normals[refs[q["image"]]])):
            with Image.open(path) as image:
                record[role] = {"member": member, "path": str(path), "size": list(image.size),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        record["pair_identity"] = diag.digest({role: {k: record[role][k] for k in ("member", "size", "sha256")}
                                               for role in ("target", "reference")})
        pairs.append(record)
    atomic_json(meta_path, {"protocol": "dev_part_crop_diagnostic", "identity": identity, "seed": seed,
                            "images": n_images, "questions": len(rows), "annotation": "physical part, no GT/model answers shown"})
    diag.agent.base.write_jsonl(root / "subset_qa.jsonl", rows)
    atomic_json(root / "pairs.json", pairs)
    return rows, pairs


class PartSession:
    def __init__(self, pairs, path):
        self.pairs, self.path = list(pairs), Path(path)
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        for pair in self.pairs:
            saved = self.data.get(pair["core_id"])
            if saved and saved["pair_identity"] != pair["pair_identity"]:
                raise ValueError("annotation image identity changed; use a new folder")

    def progress(self):
        return {"total": len(self.pairs), "done": sum(p["core_id"] in self.data for p in self.pairs),
                "first_todo": next((i for i, p in enumerate(self.pairs) if p["core_id"] not in self.data), len(self.pairs))}

    def item(self, index):
        pair = self.pairs[max(0, min(int(index), len(self.pairs)-1))]
        index = self.pairs.index(pair)
        saved = self.data.get(pair["core_id"], {})
        # No filenames, GT, question text, model answers, or expert predictions in the UI payload.
        return {"index": index, **self.progress(), "asset_category": pair["asset_category"],
                "status": saved.get("status", ""), "note": saved.get("note", ""),
                **{role: {"image_url": encode_image(Path(pair[role]["path"]), 1200)[0],
                          "box": saved.get(role+"_box")} for role in ("reference", "target")}}

    def save(self, index, reference_box, target_box, status="boxed", note=""):
        if not 0 <= int(index) < len(self.pairs):
            raise ValueError("invalid pair index")
        pair = self.pairs[int(index)]
        if status == "boxed":
            for role, box in (("reference", reference_box), ("target", target_box)):
                box_pixels(box, pair[role]["size"])
        elif status != "unavailable" or not str(note).strip():
            raise ValueError("draw both boxes, or give a reason for an unavailable pair")
        record = {"pair_identity": pair["pair_identity"], "status": status, "note": str(note),
                  "reference_box": list(reference_box) if status == "boxed" else None,
                  "target_box": list(target_box) if status == "boxed" else None,
                  "saved_at": time.time()}
        previous = self.data.get(pair["core_id"])
        if previous:
            record["history"] = previous.get("history", []) + [{k: v for k, v in previous.items() if k != "history"}]
        updated = {**self.data, pair["core_id"]: record}
        atomic_json(self.path, updated)
        self.data = updated
        return self.progress()

    def completed(self):
        if self.progress()["done"] != len(self.pairs):
            raise ValueError("finish every pair annotation or mark it unavailable before inference")
        eligible, excluded = [], []
        for pair in self.pairs:
            saved = self.data[pair["core_id"]]
            if saved["pair_identity"] != pair["pair_identity"]:
                raise ValueError("stale annotation")
            if saved["status"] == "boxed":
                for role in ("reference", "target"):
                    box_pixels(saved[role+"_box"], pair[role]["size"])
                eligible.append(pair)
            elif saved["status"] == "unavailable":
                excluded.append({"core_id": pair["core_id"], "asset_category": pair["asset_category"], "reason": saved["note"]})
            else:
                raise ValueError("unknown annotation status")
        if not eligible:
            raise ValueError("no annotated pairs available")
        return eligible, excluded


UI_HTML = r'''
<div id="part-ui" style="font:15px sans-serif;max-width:1100px;color:#17202a;background:#fff;padding:12px">
 <h3>같은 검사 부품 지정</h3><div id="part-head"></div>
 <p>양쪽에서 같은 종류의 부품 전체를 드래그하세요. 결함만 좁게 자르지 말고 주변 구조도 포함하세요.</p>
 <div style="display:flex;flex-wrap:wrap;gap:16px">
  <section style="flex:1;min-width:260px"><b>정상 참조</b><canvas id="part-reference" aria-label="정상 참조 부품 박스" style="display:block;width:100%;touch-action:none;cursor:crosshair"></canvas><button id="part-reset-reference">참조 박스 지우기</button></section>
  <section style="flex:1;min-width:260px"><b>검사 대상</b><canvas id="part-target" aria-label="검사 대상 부품 박스" style="display:block;width:100%;touch-action:none;cursor:crosshair"></canvas><button id="part-reset-target">대상 박스 지우기</button></section>
 </div>
 <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:12px">
  <button id="part-prev">이전</button><button id="part-save">저장 후 다음</button>
  <button id="part-skip">대응 부품 지정 불가로 저장</button><button id="part-todo">첫 미완료</button>
  <input id="part-note" aria-label="지정 불가 사유 또는 메모" placeholder="지정 불가 사유 또는 메모" style="min-width:200px;flex:1">
 </div><p id="part-msg" role="status"></p>
</div>
<script>
(async()=>{
 const root=document.getElementById('part-ui'), head=root.querySelector('#part-head'), msg=root.querySelector('#part-msg'), note=root.querySelector('#part-note');
 const call=async(name,args)=>{const r=await google.colab.kernel.invokeFunction(name,args,{});return r.data['application/json'];};
 let item=null,busy=false; const panels={};
 for(const role of ['reference','target']){
  const canvas=root.querySelector('#part-'+role), ctx=canvas.getContext('2d');
  const panel={canvas,ctx,box:null,image:null,start:null}; panels[role]=panel;
  const point=e=>{const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))];};
  panel.draw=()=>{if(!panel.image)return;ctx.drawImage(panel.image,0,0,canvas.width,canvas.height);if(panel.box){const b=panel.box;ctx.strokeStyle='#d32f2f';ctx.lineWidth=3;ctx.strokeRect(b[0]*canvas.width,b[1]*canvas.height,b[2]*canvas.width,b[3]*canvas.height);}};
  canvas.onpointerdown=e=>{if(busy||!panel.image)return;panel.start=point(e);canvas.setPointerCapture(e.pointerId);};
  canvas.onpointermove=e=>{if(!panel.start)return;const q=point(e),p=panel.start;panel.box=[Math.min(p[0],q[0]),Math.min(p[1],q[1]),Math.abs(p[0]-q[0]),Math.abs(p[1]-q[1])];panel.draw();};
  canvas.onpointerup=()=>{panel.start=null;}; canvas.onpointercancel=()=>{panel.start=null;};
  root.querySelector('#part-reset-'+role).onclick=()=>{if(!busy){panel.box=null;panel.draw();}};
 }
 async function load(index){
  if(busy)return;busy=true;msg.textContent='불러오는 중...';
  try{
   item=await call('part.item',[index]);note.value=item.note||'';
   await Promise.all(['reference','target'].map(role=>new Promise((resolve,reject)=>{
    const p=panels[role],im=new Image();p.box=item[role].box;p.start=null;
    im.onload=()=>{p.image=im;p.canvas.width=im.naturalWidth;p.canvas.height=im.naturalHeight;p.draw();resolve();};im.onerror=()=>reject(new Error('이미지 로딩 실패'));im.src=item[role].image_url;
   })));
   head.textContent=`${item.index+1}/${item.total} · ${item.asset_category} · 저장 ${item.done}/${item.total}`;
   msg.textContent=item.done===item.total?'표시 완료. 아래 완료 확인 셀을 실행하세요.':(item.status?'이 사진은 저장돼 있습니다.':'양쪽 부품 박스를 지정하세요.');
  }catch(e){msg.textContent='불러오기 실패: '+e;}finally{busy=false;}
 }
 async function save(status){
  if(busy||!item)return;
  if(status==='boxed'&&(!panels.reference.box||!panels.target.box)){msg.textContent='양쪽에 박스를 지정하세요.';return;}
  if(status==='unavailable'&&!note.value.trim()){msg.textContent='지정 불가 사유를 입력하세요.';return;}
  busy=true;msg.textContent='저장 중...';
  try{
   const progress=await call('part.save',[item.index,panels.reference.box,panels.target.box,status,note.value]);
   busy=false;await load(progress.first_todo>=progress.total?item.index:progress.first_todo);
  }catch(e){msg.textContent='저장 실패: '+e;busy=false;}
 }
 root.querySelector('#part-save').onclick=()=>save('boxed');root.querySelector('#part-skip').onclick=()=>save('unavailable');
 root.querySelector('#part-prev').onclick=()=>{if(item)load(item.index-1);};
 root.querySelector('#part-todo').onclick=async()=>{if(!busy){const p=await call('part.progress',[]);load(p.first_todo>=p.total?0:p.first_todo);}};
 const p=await call('part.progress',[]);await load(p.first_todo>=p.total?0:p.first_todo);
})();
</script>
'''


def register_colab(session):
    from google.colab import output
    from IPython.display import HTML, JSON, display

    output.register_callback("part.item", lambda i: JSON(session.item(i)))
    output.register_callback("part.save", lambda i, ref, target, status, note: JSON(session.save(i, ref, target, status, note)))
    output.register_callback("part.progress", lambda: JSON(session.progress()))
    display(HTML(UI_HTML))

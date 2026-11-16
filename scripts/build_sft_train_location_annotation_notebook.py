"""Build the CPU-only Colab where a person boxes the defect in the anomalous SFT training photos.

The target list is fixed here from the data folder of the trained adapter (``sft_samples.jsonl`` hash = the ``data``
field of its ``train_status.json``) and from the original supervised ZIP. No model output or automatic box goes in.
"""
import argparse
import ast
import base64
from collections import Counter, defaultdict
import csv
import io
import json
from pathlib import Path
import textwrap
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import sft_train_location_annotation as annotation

ROOT = Path(__file__).resolve().parents[1]
DATA_SHA = "a12ae43330bf6a5e0d190c42027b6f528a0d723b8b649ae28ea762e9567b9b55"   # sft_samples.jsonl of run sft_v1
SUP_ZIP_BYTES = 874753167
BLOCK = 12            # photos of one class in a row; five classes x 12 = one balanced round of 60
BUNDLE = ("scripts/sft_train_location_annotation.py", "scripts/dev_missing_location_annotation.py",
          "scripts/sft_train_location_ui.html")


def source_bytes(name):
    """Repository text with LF endings, so the bundle is the same from a Windows (CRLF) and a Linux checkout."""
    return (ROOT / name).read_bytes().replace(bytes([13, 10]), bytes([10]))


def targets(train_qa, samples, boxes_csv, archive, expected_sha=DATA_SHA, per_class=60):
    from PIL import Image
    if annotation.sha(Path(samples).read_bytes()) != expected_sha:
        raise ValueError("sft_samples.jsonl is not the data the adapter was trained on")
    rows = {}
    for line in Path(train_qa).read_text(encoding="utf-8").splitlines():
        r = json.loads(line) if line.strip() else None
        if r and r["question_type"] == "anomaly_detection" and r["anomaly_gt"] == "anomalous":
            rows[r["core_id"]] = r
    with Path(boxes_csv).open(encoding="utf-8-sig", newline="") as stream:
        split = {r["core_id"]: r["split"] for r in csv.DictReader(stream)}
    by_class = defaultdict(list)
    with ZipFile(archive) as z:
        for core, r in rows.items():
            info = z.getinfo(r["image"])
            size, crc = r["content_fingerprint"].split(":")
            if (int(size), int(crc, 16)) != (info.file_size, info.CRC):
                raise ValueError("training photo differs from the original ZIP: " + core)
            b = z.read(info)
            with Image.open(io.BytesIO(b)) as im:
                if im.getexif().get(274, 1) != 1:
                    raise ValueError("unsupported EXIF rotation: " + core)
                dims = list(im.size)
            by_class[r["asset_category"], r["defect_gt"]].append(dict(
                core_id=core, image=r["image"], sha256=annotation.sha(b), size=dims, asset_category=r["asset_category"],
                defect_gt=r["defect_gt"], anomaly_gt="anomalous", sft_split=split[core],
                content_fingerprint=r["content_fingerprint"]))
    if any(len(v) != per_class for v in by_class.values()) or not by_class:
        raise ValueError(f"expected {per_class} anomalous photos per class: " + str({k: len(v) for k, v in by_class.items()}))
    for v in by_class.values():
        v.sort(key=lambda t: annotation.sha(("sft-location/" + t["core_id"]).encode()))
    # Rounds of BLOCK photos per class: any finished round is class-balanced, and one defect type stays on screen
    # for a stretch so the annotator applies one rule at a time.
    ordered = [t for start in range(0, per_class, BLOCK) for key in sorted(by_class) for t in by_class[key][start:start + BLOCK]]
    manifest = dict(protocol=annotation.PROTOCOL, source=dict(sft_samples_sha256=expected_sha, run_tag="sft_v1"),
                    cohort=f"all_{len(ordered)}_anomalous_sft_training_photos", targets=ordered)
    annotation.validate_manifest(manifest)
    return manifest


def build(args):
    manifest = targets(args.train_qa, args.samples, args.boxes_csv, args.sup_zip)
    n = len(manifest["targets"])
    counts = Counter(t["asset_category"] + ":" + t["defect_gt"] for t in manifest["targets"])
    payloads = {name: source_bytes(name) for name in BUNDLE}
    payloads["inputs/manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    hashes = {name: annotation.sha(b) for name, b in payloads.items()}
    buf = io.BytesIO()
    with ZipFile(buf, "w", ZIP_DEFLATED) as z:
        for name, b in sorted(payloads.items()):
            info = ZipInfo(name, date_time=(2026, 11, 16, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            z.writestr(info, b)
    payload = buf.getvalue()
    source_sha = annotation.sha(payload)
    cells = []

    def cell(kind, text):
        c = {"id": "sft-location-label-" + str(len(cells)), "cell_type": kind, "metadata": {},
             "source": textwrap.dedent(text).strip().splitlines(keepends=True)}
        if kind == "code":
            ast.parse("".join(c["source"]))
            c.update(execution_count=None, outputs=[])
        cells.append(c)

    cell("markdown", f"""
    # 학습 사진 결함 위치 표시 — {n}장

    파일: `lvlm_sft_train_location_annotation_colab.ipynb`

    **GPU 불필요(일반 CPU). 위에서부터 실행 → 표시 화면에서 박스 표시 → 마지막 결과 저장 셀을 실행하세요.**

    LoRA 학습에 쓴 결함 사진 {n}장입니다(누락 60, 녹 180, 새 둥지 60). dev240이나 gold 사진이 아니며, 둘 다 바꾸지 않습니다.
    화면에는 사진과 **결함 종류**만 나옵니다. 모델 답이나 자동으로 만든 박스는 보이지 않습니다.

    - 마우스로 끌어서 **결함을 감싸는 박스**를 그리세요. 결함이 여러 곳이면 각각 그립니다.
    - 화면 위 빨간 줄에 이 사진의 결함 종류와 무엇을 감싸야 하는지 나옵니다.
    - 위치를 정하기 어려우면 `위치 판단 불가`, 결함 라벨 자체가 이상하면 `라벨 이견`을 누르고 사유를 적으세요. 이 사진들은 위치 학습에서 빠집니다.
    - 박스를 그리거나 지우면 Drive에 **초안이 자동 저장**됩니다. `저장 후 다음(Enter)`을 눌러야 확정입니다. `Z`는 마지막 박스 지우기입니다.
    - 중간에 나가도 됩니다. `초안 저장됨` 표시를 확인하고 나가면, 다음에 같은 노트북을 실행했을 때 첫 미완료 사진부터 이어집니다.
    - 같은 종류가 {BLOCK}장씩 묶여 나옵니다. 60장을 끝낼 때마다 다섯 종류가 같은 수만큼 채워집니다.

    결과 파일: `{annotation.RESULT_ZIP}`. 다 못 끝내도 마지막 셀로 중간 백업을 만들 수 있습니다.
    """)
    cell("code", f"""
    RUN_FOLDER = "lvlm_insplad_results/{annotation.PREFIX}_sft_v1"
    SUP_ZIP_PATH = ""  # 자동 검색이 안 될 때만 Drive의 실제 ZIP 경로를 입력
    """)
    cell("code", f"""
    import base64,hashlib,io,json,sys,zipfile
    from pathlib import Path
    from google.colab import drive
    drive.mount('/content/drive',force_remount=False)
    DRIVE=Path('/content/drive/MyDrive')
    SOURCE_SHA256={source_sha!r}
    SOURCE_MANIFEST={hashes!r}
    SOURCE_BYTES=base64.b64decode({base64.b64encode(payload).decode()!r})
    assert hashlib.sha256(SOURCE_BYTES).hexdigest()==SOURCE_SHA256
    CODE=Path('/content/sft_train_location_'+SOURCE_SHA256[:12])
    with zipfile.ZipFile(io.BytesIO(SOURCE_BYTES)) as archive:
        assert set(archive.namelist())==set(SOURCE_MANIFEST)
        for name in archive.namelist():
            dest=CODE/name;data=archive.read(name)
            assert dest.resolve().is_relative_to(CODE.resolve())
            assert hashlib.sha256(data).hexdigest()==SOURCE_MANIFEST[name]
            if dest.exists() and dest.read_bytes()!=data:raise RuntimeError('내장 코드와 다른 파일입니다. 새 런타임을 사용하세요.')
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    sys.path.insert(0,str(CODE/'scripts'))
    import sft_train_location_annotation as annotation
    def find_zip():
        if SUP_ZIP_PATH:
            p=Path(SUP_ZIP_PATH)
            if not p.is_file():raise FileNotFoundError(str(p))
            return p
        for p in [DRIVE/'supervised_fault_classification.zip',DRIVE/'lvlm_insplad_data/supervised_fault_classification.zip']:
            if p.is_file() and p.stat().st_size=={SUP_ZIP_BYTES}:return p
        matches=[p for p in DRIVE.rglob('supervised_fault_classification.zip') if p.is_file() and p.stat().st_size=={SUP_ZIP_BYTES}]
        if not matches:raise FileNotFoundError('supervised_fault_classification.zip 원본을 Drive에서 찾지 못했습니다.')
        return sorted(matches)[0]
    OUT=DRIVE/RUN_FOLDER
    assert OUT.name.startswith(annotation.PREFIX)
    manifest=annotation.read(CODE/'inputs/manifest.json')
    print('사진 {n}장을 원본 ZIP에서 꺼내 확인합니다. 1~2분 걸릴 수 있습니다.')
    session=annotation.prepare(manifest,find_zip(),OUT,cache=Path('/content/sft_train_location_images'),
        progress=lambda i,n:print(f'  사진 확인 {{i}}/{{n}}',flush=True))
    (OUT/'source_bundle.zip').write_bytes(SOURCE_BYTES)
    annotation.atomic(OUT/'source_manifest.json',SOURCE_MANIFEST)
    print('원본 SHA 확인 완료. 저장/재개 상태:',session.progress())
    print('저장 위치:',OUT/'annotations.json')
    """)
    cell("code", """
    # 이 화면에서 박스를 표시합니다. 박스 추가/삭제는 초안 저장, Enter는 확정 후 다음.
    annotation.register_colab(session)
    """)
    cell("markdown", f"""
    ## 표시를 마친 뒤(또는 중간 백업할 때) 아래 셀 실행

    화면에서 `저장 후 다음`으로 확정한 사진만 위치 정답으로 들어갑니다.
    아래 셀은 Drive에 저장된 상태로 결과를 만들고, 종류별 표시 수와 **위치가 9칸 중 어디에 몰려 있는지**를 출력합니다.
    """)
    cell("code", """
    from google.colab import files
    result,summary=session.export()
    print('확정',summary['done'],'/',summary['total'],'· 박스',summary['boxed'],'· 판단 불가',summary['unlocatable'],'· 이견',summary['disputed'])
    print('종류별 박스 수:',summary['boxed_by_defect'])
    print('위치 분포(박스 중심이 놓인 칸):',summary['region_counts'])
    print('가운데 칸만 정답인 사진:',summary['centre_only_images'],'/',summary['positive_spatial_support'])
    if not summary['complete']:print('미확정 사진이 있습니다. 중간 백업이며, 표시를 끝낸 뒤 이 셀을 다시 실행하세요.')
    print('ZIP:',result)
    files.download(str(result))
    """)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}, "colab": {"provenance": []}},
          "nbformat": 4, "nbformat_minor": 5}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    receipt = {"source_bundle_sha256": source_sha, "source_files": hashes,
               "notebook_sha256": annotation.sha(args.output.read_bytes()), "images": n, "images_by_class": dict(sorted(counts.items())),
               "manifest_identity": annotation.digest(manifest), "sft_samples_sha256": DATA_SHA, "code_cells_compiled": 4,
               "neural_calls": 0, "annotation_completed": False}
    if args.evidence_dir:
        annotation.atomic(args.evidence_dir / "manifest.json", manifest)
        annotation.atomic(args.evidence_dir / "notebook_build.json", receipt)
    print(json.dumps(receipt, indent=2))
    return nb


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for key in ("train-qa", "samples", "boxes-csv", "sup-zip"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "notebooks/lvlm_sft_train_location_annotation_colab.ipynb")
    p.add_argument("--evidence-dir", type=Path)
    build(p.parse_args())

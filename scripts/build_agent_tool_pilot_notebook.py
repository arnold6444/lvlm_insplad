"""Build a self-contained, single-run Colab; this does not execute inference."""
from __future__ import annotations

import argparse
import base64
import hashlib
import inspect
import io
import json
from pathlib import Path
import re
import subprocess
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def source_bundle(commit):
    """Only immutable tracked source/config; never local data, weights or credentials."""
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit,
                                     "--", "scripts", "src", "configs", "pyproject.toml"], cwd=ROOT, text=True).splitlines()
    names = sorted(n for n in names if n.endswith((".py", ".json", ".toml")))
    files = {n: subprocess.check_output(["git", "show", f"{commit}:{n}"], cwd=ROOT) for n in names}
    if "scripts/run_agent_tool_pilot.py" not in files:
        raise ValueError("source commit does not contain the pilot runner")
    receipt = dict(commit=commit, files={n: hashlib.sha256(data).hexdigest() for n, data in files.items()})
    files[".agent_pilot_source.json"] = (json.dumps(receipt, sort_keys=True)+"\n").encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return buffer.getvalue()


def extract_source_bundle(encoded, expected_sha, destination, commit):
    """Colab bootstrap: validate all bytes and refuse changed existing source."""
    import base64
    import hashlib
    import io
    import json
    from pathlib import Path
    import zipfile
    payload = base64.b64decode(encoded, validate=False)
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError("embedded source archive hash mismatch")
    root = Path(destination).resolve()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate source member")
        receipt = json.loads(archive.read(".agent_pilot_source.json"))
        if receipt["commit"] != commit or set(names) != set(receipt["files"]) | {".agent_pilot_source.json"}:
            raise ValueError("source receipt/commit mismatch")
        for name in names:
            target = (root / name).resolve()
            if not target.is_relative_to(root) or "\\" in name:
                raise ValueError("unsafe source member")
            data = archive.read(name)
            if name in receipt["files"] and hashlib.sha256(data).hexdigest() != receipt["files"][name]:
                raise ValueError("source member hash mismatch")
            if target.exists() and target.read_bytes() != data:
                raise ValueError("existing source changed; use a fresh runtime")
        for name in names:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(archive.read(name))
    return receipt


def cell(kind, source):
    result = dict(cell_type=kind, metadata={}, source=textwrap.dedent(source).strip()+"\n")
    if kind == "code":
        compile(result["source"], "<colab cell>", "exec")
        result.update(execution_count=None, outputs=[])
    return result


def build(execution_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", execution_commit):
        raise ValueError("execution commit must be a full immutable SHA")
    bundle = source_bundle(execution_commit)
    encoded = base64.b64encode(bundle).decode("ascii")
    cells = [cell("markdown", """
        # 도구 선택 비교 실험 — 240장 한 번에 실행
        L4/A100 GPU(20 GiB 이상)에서 **모두 실행**하세요. 추가 학습·웹 검색·gold 실행·결과 GitHub 게시는 하지 않습니다.

        기존 Drive의 FROZEN dev 세트와 원본 ZIP 2개를 사용합니다. **설정 변경 없이 모두 실행 한 번으로 240장 전체 비교와 결과 ZIP 다운로드까지 진행합니다.** 실행 소스를 파일 안에 포함해 GitHub 업로드나 git push가 필요 없습니다.

        정상참조 직접 질문 / 고정 확대+참조 / 모델의 도구 선택 / 동일 호출 수 재질문을 비교합니다. 최대 5,760회 호출이며 실제 시간과 성능은 미측정입니다. 중단되면 같은 파일을 다시 실행해 저장한 결과를 재사용합니다.

        **dev에는 결함 위치 정답이 없습니다. Q3 위치 정확도와 세 질문 위치 포함 종합 성공률은 계산하지 않습니다.** 영역 유무 종합 점수를 과거 11.7%/37.9%와 비교하지 마세요.
        마지막 셀은 실제 오류·무효·도구 호출을 확인하고 결과 ZIP을 내려받습니다. 테스트 통과는 GPU 성능 개선 결과가 아닙니다.
        """), cell("code", f'''
        #@title 1. 실행 설정 (수정 불필요)
        EXECUTION_COMMIT = "{execution_commit}"
        SOURCE_ZIP_SHA256 = "{hashlib.sha256(bundle).hexdigest()}"
        SOURCE_ZIP_B64 = "{encoded}"
        N_IMAGES = 240
        MODEL = "internvl3_5_8b"
        RUN_TAG = "once_v2"
        MAX_CALLS = 3
        DEV_FOLDER = "lvlm_insplad_qa/gold_v2_dev"
        SUP_ZIP_PATH = ""  # 자동 탐색이 모호할 때만 Drive의 절대 경로 입력
        UNSUP_ZIP_PATH = ""
        '''), cell("code", inspect.getsource(extract_source_bundle) + '\n' + textwrap.dedent('''
        import csv, json, os, re, subprocess, sys, zipfile
        from pathlib import Path
        from google.colab import drive

        def run(command, cwd=None, check=True):
            command = [str(x) for x in command]
            print("+", " ".join(command[:10]), flush=True)
            process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
            result = process.wait()
            if result and check:
                raise RuntimeError("명령 실패: 위 오류를 확인하세요. 결과를 성공으로 처리하지 않습니다.")
            return result

        if N_IMAGES != 240 or not re.fullmatch(r"[A-Za-z0-9_-]+", RUN_TAG):
            raise ValueError("240장 한 번 실행 설정을 유지하세요.")
        if MODEL not in ("internvl3_5_8b", "qwen3_5_4b") or MAX_CALLS != 3:
            raise ValueError("고정 pilot 설정을 유지하세요.")
        drive.mount("/content/drive", force_remount=False)
        DRIVE = Path("/content/drive/MyDrive")
        REPO_DIR = Path("/content/insplad_agent_tool_pilot_once")
        def verify_code():
            extract_source_bundle(SOURCE_ZIP_B64, SOURCE_ZIP_SHA256, REPO_DIR, EXECUTION_COMMIT)
        verify_code()
        run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.14.0", "accelerate==1.14.0",
             "sentencepiece==0.2.2", "numpy>=1.26", "pillow>=10", "pandas>=2.2", "tqdm>=4.66", "timm"])
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3:
            raise RuntimeError("VRAM 20 GiB 이상 L4/A100 GPU를 선택하세요. 양자화로 우회하지 않습니다.")
        print("GPU:", torch.cuda.get_device_name(0), "source:", EXECUTION_COMMIT)
        DEV_DIR = DRIVE / DEV_FOLDER
        summary = json.loads((DEV_DIR / "build_summary.json").read_text(encoding="utf-8"))
        if summary.get("quality_status") != "FROZEN":
            raise RuntimeError("기존 dev 세트가 FROZEN이 아닙니다.")
        DEV_QA = (DEV_DIR / summary["file"]).resolve()
        if not DEV_QA.is_relative_to(DEV_DIR.resolve()) or not DEV_QA.is_file():
            raise ValueError("dev QA 경로를 확인하세요.")
        RUN_ROOT = DRIVE / "lvlm_insplad_results" / f"agent_tool_pilot_dev{N_IMAGES}_{MODEL}_{RUN_TAG}"

        def find_zip(name, size, override):
            if override:
                p = Path(override)
                if not p.is_file() or p.stat().st_size != size:
                    raise ValueError(f"ZIP 경로/크기 불일치: {p}")
                return p
            preferred = [DRIVE / name, DRIVE / "lvlm_insplad_data" / name]
            for p in preferred:
                if p.is_file() and p.stat().st_size == size:
                    return p
            matches = [p for p in DRIVE.rglob(name) if p.is_file() and p.stat().st_size == size]
            if len(matches) != 1:
                raise RuntimeError(f"{name}: 후보 {len(matches)}개. 설정 셀에 정확한 경로를 입력하세요.")
            return matches[0]
        SUP_ZIP = find_zip("supervised_fault_classification.zip", 874_753_167, SUP_ZIP_PATH)
        UNSUP_ZIP = find_zip("unsupervised_anomaly_detection.zip", 1_170_967_719, UNSUP_ZIP_PATH)
        REFS_CSV = DEV_DIR / "refs/retrieved_refs.csv"
        has_diffdate = False
        if REFS_CSV.is_file():
            with REFS_CSV.open(encoding="utf-8-sig", newline="") as stream:
                has_diffdate = "retrieved_diffdate" in (csv.DictReader(stream).fieldnames or [])
        if not has_diffdate:
            prepared = DRIVE / "lvlm_insplad_results" / f"agent_tool_pilot_reference_{RUN_TAG}"
            run([sys.executable, REPO_DIR / "scripts/build_retrieved_refs.py", "--qa", DEV_QA,
                 "--sup-zip", SUP_ZIP, "--unsup-zip", UNSUP_ZIP, "--out-dir", prepared, "--device", "cuda"])
            REFS_CSV = prepared / "retrieved_refs.csv"
        if not REFS_CSV.is_file():
            raise FileNotFoundError("다른 촬영일 정상 참조 CSV가 없습니다.")
        print("출력:", RUN_ROOT)
        ''')), cell("code", '''
        verify_code()
        INFERENCE_EXIT_CODE = run([sys.executable, REPO_DIR / "scripts/run_agent_tool_pilot.py", "--dev-qa", DEV_QA,
             "--refs-csv", REFS_CSV, "--sup-zip", SUP_ZIP, "--unsup-zip", UNSUP_ZIP,
             "--out-dir", RUN_ROOT, "--model", MODEL, "--n-images", N_IMAGES, "--max-calls", MAX_CALLS], check=False)
        '''), cell("code", '''
        import pandas as pd
        from google.colab import files
        def read_receipt(name):
            p = RUN_ROOT / name
            return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
        status = read_receipt("run_status.json")
        metadata = read_receipt("run_metadata.json")
        verified_complete = (INFERENCE_EXIT_CODE == 0 and status.get("complete") and metadata.get("execution") == "real_hf"
                and status.get("run_id") == metadata.get("run_id")
                and metadata.get("benchmark_commit") == EXECUTION_COMMIT)
        if not verified_complete:
            print("실행 오류 또는 완료 미확인: 아래 ZIP은 진단용이며 성공 결과가 아닙니다.")
        if (metadata and (metadata.get("execution") != "real_hf"
                or status.get("run_id") != metadata.get("run_id")
                or metadata.get("benchmark_commit") != EXECUTION_COMMIT)):
            print("실행 출처/메타데이터를 확인하지 못했습니다.")
        print(json.dumps(status, ensure_ascii=False, indent=2))
        for name in ("leaderboard.csv", "q2_by_defect.csv", "significance.csv"):
            p = RUN_ROOT / name
            if p.is_file() and p.stat().st_size:
                display(pd.read_csv(p))
        print("Q3 위치 정확도/위치 포함 종합 성공률은 미평가입니다.")
        if not status.get("pilot_format_gate"):
            print("도구 형식 성공률이 부족합니다. 무효 응답을 포함한 결과로 해석하세요.")
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        (RUN_ROOT / "notebook_execution.json").write_text(json.dumps(dict(
            exit_code=INFERENCE_EXIT_CODE, verified_complete=bool(verified_complete),
            execution_commit=EXECUTION_COMMIT, status_run_id=status.get("run_id"),
            metadata_run_id=metadata.get("run_id")), indent=2), encoding="utf-8")
        archive = RUN_ROOT.parent / (RUN_ROOT.name + "_results.zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(RUN_ROOT.rglob("*")):
                if p.is_file() and not any(part.startswith("_") for part in p.relative_to(RUN_ROOT).parts):
                    z.write(p, p.relative_to(RUN_ROOT))
        files.download(str(archive))
        print("비교 완료 결과" if verified_complete else "미완료 실행 진단", "ZIP 저장:", archive)
        ''')]
    cells[1]["metadata"]["cellView"] = "form"
    return dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", name="python3"),
                                          language_info=dict(name="python"), accelerator="GPU", colab=dict(provenance=[])),
                nbformat=4, nbformat_minor=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks/lvlm_agent_tool_pilot_colab.ipynb")
    args = parser.parse_args()
    notebook = build(args.execution_commit)
    args.output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    print(f"Built {args.output}; {sum(c['cell_type']=='code' for c in notebook['cells'])} code cells compiled")


if __name__ == "__main__":
    main()

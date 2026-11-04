"""Build the SHA-pinned Colab pilot; this does not execute neural inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = dict(cell_type=kind, metadata={}, source=textwrap.dedent(source).strip()+"\n")
    if kind == "code":
        compile(result["source"], "<colab cell>", "exec")
        result.update(execution_count=None, outputs=[])
    return result


def build(execution_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", execution_commit):
        raise ValueError("execution commit must be a full immutable SHA")
    cells = [cell("markdown", """
        # 학습 없는 도구 선택 실험 — dev40 → dev240
        L4/A100 GPU(20 GiB 이상)에서 **모두 실행**하세요. 추가 학습·웹 검색·gold 실행·결과 GitHub 게시는 하지 않습니다.

        기존 Drive의 FROZEN dev 세트와 원본 ZIP 2개를 사용합니다. 40장 점검 뒤 유용할 때만 N_IMAGES=240으로 변경해 같은 설정으로 한 번 비교합니다. 두 결과는 독립 반복이 아닙니다.

        정상참조 직접 질문 / 고정 확대+참조 / 모델의 도구 선택 / 동일 호출 수 재질문을 비교합니다. 최대 40장 960회, 240장 5,760회 호출이며 실제 시간과 성능은 미측정입니다.

        **dev에는 결함 위치 정답이 없습니다. Q3 위치 정확도와 세 질문 위치 포함 종합 성공률은 계산하지 않습니다.** 영역 유무 종합 점수를 과거 11.7%/37.9%와 비교하지 마세요.
        마지막 셀은 실제 오류·무효·도구 호출을 확인하고 결과 ZIP을 내려받습니다. 테스트 통과는 GPU 성능 개선 결과가 아닙니다.
        """), cell("code", f'''
        EXECUTION_COMMIT = "{execution_commit}"
        N_IMAGES = 40  # 40 또는 240; 설정 탐색은 dev에서만
        MODEL = "internvl3_5_8b"
        RUN_TAG = "v1"  # 설정/코드/입력이 바뀌면 새 태그
        MAX_CALLS = 3
        DEV_FOLDER = "lvlm_insplad_qa/gold_v2_dev"
        SUP_ZIP_PATH = ""  # 자동 탐색이 모호할 때만 Drive의 절대 경로 입력
        UNSUP_ZIP_PATH = ""
        '''), cell("code", '''
        import csv, json, os, re, subprocess, sys, zipfile
        from pathlib import Path
        from google.colab import drive

        def run(command, cwd=None):
            command = [str(x) for x in command]
            print("+", " ".join(command[:10]), flush=True)
            process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
            if process.wait():
                raise RuntimeError("명령 실패: 위 오류를 확인하세요. 결과를 성공으로 처리하지 않습니다.")

        if N_IMAGES not in (40, 240) or not re.fullmatch(r"[A-Za-z0-9_-]+", RUN_TAG):
            raise ValueError("N_IMAGES=40/240, 안전한 RUN_TAG를 사용하세요.")
        if MODEL not in ("internvl3_5_8b", "qwen3_5_4b") or MAX_CALLS != 3:
            raise ValueError("고정 pilot 설정을 유지하세요.")
        drive.mount("/content/drive", force_remount=False)
        DRIVE = Path("/content/drive/MyDrive")
        REPO_DIR = Path("/content/insplad_agent_tool_pilot")
        REPO_URL = "https://github.com/arnold6444/lvlm_insplad.git"
        if not REPO_DIR.exists():
            run(["git", "clone", "--no-checkout", REPO_URL, REPO_DIR])
            run(["git", "checkout", "--detach", EXECUTION_COMMIT], cwd=REPO_DIR)
        def verify_code():
            remote = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=REPO_DIR, text=True).strip()
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True).strip()
            dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_DIR, text=True).strip()
            if remote != REPO_URL or head != EXECUTION_COMMIT or dirty:
                raise RuntimeError("소스/커밋/수정 상태가 다릅니다. 새 런타임을 사용하세요.")
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
        '''), cell("code", '''
        verify_code()
        run([sys.executable, REPO_DIR / "scripts/run_agent_tool_pilot.py", "--dev-qa", DEV_QA,
             "--refs-csv", REFS_CSV, "--sup-zip", SUP_ZIP, "--unsup-zip", UNSUP_ZIP,
             "--out-dir", RUN_ROOT, "--model", MODEL, "--n-images", N_IMAGES, "--max-calls", MAX_CALLS])
        '''), cell("code", '''
        import pandas as pd
        from google.colab import files
        status = json.loads((RUN_ROOT / "run_status.json").read_text(encoding="utf-8"))
        metadata = json.loads((RUN_ROOT / "run_metadata.json").read_text(encoding="utf-8"))
        if (not status.get("complete") or metadata.get("execution") != "real_hf"
                or status.get("run_id") != metadata.get("run_id")
                or metadata.get("benchmark_commit") != EXECUTION_COMMIT):
            raise RuntimeError("실제 GPU 추론 완료를 확인하지 못했습니다. 상태/오류를 확인하세요.")
        print(json.dumps(status, ensure_ascii=False, indent=2))
        display(pd.read_csv(RUN_ROOT / "leaderboard.csv"))
        display(pd.read_csv(RUN_ROOT / "q2_by_defect.csv"))
        display(pd.read_csv(RUN_ROOT / "significance.csv"))
        print("Q3 위치 정확도/위치 포함 종합 성공률은 미평가입니다.")
        if not status.get("pilot_format_gate"):
            print("도구 형식 성공률이 부족합니다. dev240으로 자동 확대하지 마세요.")
        archive = RUN_ROOT.parent / (RUN_ROOT.name + "_results.zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(RUN_ROOT.rglob("*")):
                if p.is_file() and not any(part.startswith("_") for part in p.relative_to(RUN_ROOT).parts):
                    z.write(p, p.relative_to(RUN_ROOT))
        files.download(str(archive))
        print("이 ZIP은 GPU 실행 결과이며 GitHub에는 자동 게시하지 않습니다.")
        ''')]
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

#!/usr/bin/env python3
"""Write the MiniCPM greedy re-run Colab notebook, pinning the git blob IDs of every code file it runs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_gold_v2_matched_call_notebook import cell  # noqa: E402

NOTEBOOK = ROOT / "notebooks/lvlm_gold_v2_minicpm_greedy_colab.ipynb"
EXTRA_FILES = ("src/insplad_benchmark/hf_adapters.py", "results/latest/gold_v2_reference/reference_manifest.csv",
               "results/latest/gold_v2_consistency_metrics/consistency_leaderboard.csv")
PROBE = """
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(root / "scripts"), str(root / "src")]
import run_gold_v2_minicpm_greedy
files = {Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, "__file__", None)}
print(json.dumps(sorted(f.relative_to(root).as_posix() for f in files if f.suffix == ".py" and root in f.parents)))
"""

INTRO = """# MiniCPM-V-4.5 greedy 재실행 (gold, 참조 4조건)

**새 L4/A100 런타임 → 런타임 / 모두 실행 → Drive 승인 → 결과 ZIP 전달.** 토큰·정답 입력이 필요 없고, 결과를 GitHub에 자동으로 올리지 않습니다.

왜 돌리나: 논문 표 2·3의 MiniCPM gold 수치는 어댑터가 샘플링(temperature 0.7)하던 시절(PR #47 이전)에 나왔습니다. 같은 입력을 greedy로 다시 돌려 그 행을 바꿉니다.

- 조건 4개: 이미지만 / + 정상 참조 / + 결함 참조 / + 정상·결함 참조
- 각 조건에서 534문항(Q1·Q2 각 246, Q3 42)을 한 번씩 물어 총 **2,136회** 호출합니다. L4 기준 대략 1~1.5시간입니다.
- 프롬프트·이미지 순서·참조 이미지는 원래 실행과 같습니다. 결함 참조는 저장소의 `results/latest/gold_v2_reference/reference_manifest.csv`를 씁니다. 바뀌는 건 디코딩(greedy)뿐입니다.
- 20회마다 Drive에 저장합니다. 끊기면 **추론 셀만 다시 실행**하면 이어서 합니다. 오류가 남은 조건은 완료로 인정하지 않습니다.

출력: `MyDrive/lvlm_insplad_results/gold_v2_minicpm_greedy_v1/`와 `gold_minicpm_greedy_results.zip`. 기존 결과 폴더는 건드리지 않습니다.
"""

CONFIG_CELL = """REPO = "arnold6444/lvlm_insplad"
EXECUTION_REF = "main"
CODE_BLOBS = {code_blobs}
QA_FOLDER = "lvlm_insplad_qa/gold_v2_final"
RUN_FOLDER = "lvlm_insplad_results/gold_v2_minicpm_greedy_v1"
CHECKPOINT_EVERY = 20
UNSUP_ZIP_NAME = "unsupervised_anomaly_detection.zip"
UNSUP_ZIP_SIZE = 1_170_967_719
UNSUP_ZIP_PATH = ""
SUP_ZIP_NAME = "supervised_fault_classification.zip"
SUP_ZIP_SIZE = 874_753_167
SUP_ZIP_PATH = ""
"""

SETUP_CELL = '''# Drive 연결 + 공개 저장소 받기(고정 blob 확인) + 패키지. 토큰은 쓰지 않으므로 자동 게시가 불가능합니다.
import json, os, subprocess, sys
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive", force_remount=False)
DRIVE_ROOT = Path("/content/drive/MyDrive")
QA_DIR = DRIVE_ROOT / QA_FOLDER
RUN_ROOT = DRIVE_ROOT / RUN_FOLDER
if not RUN_ROOT.name.startswith("gold_v2_minicpm_greedy"):
    raise ValueError("새 실험 전용 RUN_FOLDER를 사용하세요.")
REPO_DIR = Path("/content/lvlm_minicpm_greedy_code")
os.environ.pop("GITHUB_TOKEN", None)  # never publish from this notebook

def run(cmd, *, check=True):
    """Run a command and stream its output into this cell."""
    cmd = [str(x) for x in cmd]
    print("+", " ".join(cmd[:12]) + (" ..." if len(cmd) > 12 else ""), flush=True)
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    if check and p.returncode:
        raise RuntimeError(f"command failed with exit code {p.returncode} (마지막 출력 위 참고)")
    return p.returncode

if not REPO_DIR.exists():
    run(["git", "clone", "--depth", "1", "--branch", EXECUTION_REF, f"https://github.com/{REPO}.git", REPO_DIR])

def verify_code():
    """Run only the exact code that was tested and merged."""
    for rel, blob in CODE_BLOBS.items():
        actual = subprocess.check_output(["git", "-C", REPO_DIR, "rev-parse", f"HEAD:{rel}"], text=True).strip()
        if actual != blob:
            raise RuntimeError(f"{rel} 코드가 이 노트북과 다릅니다. 저장소의 최신 노트북을 다시 여세요.")
    if subprocess.check_output(["git", "-C", REPO_DIR, "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise RuntimeError("받은 코드가 수정됐습니다. 새 런타임을 사용하세요.")
    return subprocess.check_output(["git", "-C", REPO_DIR, "rev-parse", "HEAD"], text=True).strip()

print("code commit:", verify_code())
run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.14.0", "accelerate==1.14.0",
     "sentencepiece==0.2.2", "pandas>=2.2", "numpy>=1.26", "pillow>=10", "tqdm>=4.66", "timm"])
sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR / "scripts"))
import pandas as pd
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음")
if not torch.cuda.is_available():
    raise RuntimeError("GPU Runtime(L4/A100)을 사용하세요.")
if torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3:
    raise RuntimeError("VRAM 20GB 이상 L4/A100이 필요합니다.")
'''

INPUT_CELL = '''# 입력 확인: 원본 ZIP, FROZEN gold QA(534문항), 저장소의 참조 목록
def find_zip(filename, expected_size, override=""):
    candidates = [Path(override)] if override else []
    candidates += [DRIVE_ROOT / filename, DRIVE_ROOT / "lvlm_insplad_data" / filename]
    for p in candidates:
        if p.is_file() and p.stat().st_size == expected_size:
            return p
    print(f"{filename}: Drive 전체 검색 중 (몇 분 걸릴 수 있음)", flush=True)
    for p in DRIVE_ROOT.rglob(filename):
        if p.is_file() and p.stat().st_size == expected_size:
            return p
    raise FileNotFoundError(f"{filename} ({expected_size} bytes) not found")

SUP_ZIP = find_zip(SUP_ZIP_NAME, SUP_ZIP_SIZE, SUP_ZIP_PATH)
UNSUP_ZIP = find_zip(UNSUP_ZIP_NAME, UNSUP_ZIP_SIZE, UNSUP_ZIP_PATH)
print("SUP:", SUP_ZIP); print("UNSUP:", UNSUP_ZIP)

import run_gold_v2_minicpm_greedy as greedy
summary = json.loads((QA_DIR / "build_summary.json").read_text(encoding="utf-8"))
if summary.get("quality_status") != "FROZEN" or summary.get("final_master_images") != 246:
    raise RuntimeError("원래 FROZEN gold246 QA가 필요합니다.")
GOLD_ROWS = greedy.v2.load_gold_qa_v2(QA_DIR)
if len(GOLD_ROWS) != 534:
    raise RuntimeError(f"gold 문항수가 534가 아닙니다: {len(GOLD_ROWS)}")
greedy.load_manifest(greedy.REFERENCE_MANIFEST, GOLD_ROWS)  # same targets and fixed normals as the QA
print("QA 534문항, 참조 목록 일치. 조건:", ", ".join(greedy.CONDITIONS), "→ 총", 534 * len(greedy.CONDITIONS), "회")
'''

RUN_CELL = '''print("code commit:", verify_code())
# 추론 / 자동 채점. 중단되거나 오류가 남으면 이 셀만 다시 실행.
run([sys.executable, REPO_DIR / "scripts/run_gold_v2_minicpm_greedy.py",
     "--qa-dir", QA_DIR, "--sup-zip", SUP_ZIP, "--unsup-zip", UNSUP_ZIP,
     "--out-dir", RUN_ROOT, "--checkpoint-every", str(CHECKPOINT_EVERY)], check=False)
'''

CHECK_CELL = '''# 자동 결과 확인: 네 조건이 모두 오류 없이 끝나야 완료
pd.set_option("display.max_rows", 100)
status = json.loads((RUN_ROOT / "run_status.json").read_text(encoding="utf-8"))
meta = json.loads((RUN_ROOT / "run_metadata.json").read_text(encoding="utf-8"))
display(pd.DataFrame(status["conditions"]))
if not status.get("complete") or meta.get("row_errors") or len(status["conditions"]) != 4:
    raise RuntimeError("오류/누락이 남았습니다. 추론 셀을 다시 실행하세요.")
print("새 호출:", meta["called_this_run"], "저장분 재사용:", meta["reused_calls"], "무효 출력(오답 처리):", meta["invalid_outputs"])
lb = pd.read_csv(RUN_ROOT / "leaderboard.csv")
display(lb[["condition", "q1_anomaly_f1", "q2_defect_macro_f1", "joint_accuracy", "consistency_error",
            "false_defect_rate", "q3_accuracy", "q1_invalid_rate", "q2_invalid_rate"]].round(3))
print("greedy vs 이전 샘플링(발표값)")
cmp = pd.read_csv(RUN_ROOT / "comparison_vs_sampled.csv")
display(cmp.pivot_table(index="condition", columns="metric", values=["greedy", "sampled_published"]).round(3))
print("이미지만 대비 각 참조 조건 (paired bootstrap 95% CI)")
display(pd.read_csv(RUN_ROOT / "significance.csv").round(3))
print("완료. GitHub 자동 게시 없음. 결과 ZIP을 전달하세요:", RUN_ROOT)
'''

EXPORT_CELL = '''# 성공/실패 산출물 모두 내려받을 수 있습니다. 실패 ZIP은 완료 결과가 아닙니다.
from zipfile import ZipFile, ZIP_DEFLATED
from google.colab import files
archive = Path("/content/gold_minicpm_greedy_results.zip")
with ZipFile(archive, "w", ZIP_DEFLATED) as z:
    for p in RUN_ROOT.rglob("*"):
        if p.is_file() and "_images" not in p.relative_to(RUN_ROOT).parts:
            z.write(p, p.relative_to(RUN_ROOT))
files.download(str(archive))
'''


def code_files() -> list[str]:
    found = json.loads(subprocess.check_output([sys.executable, "-c", PROBE, str(ROOT)], text=True))
    return sorted(set(found) | set(EXTRA_FILES))


def code_blobs() -> dict[str, str]:
    return {rel: subprocess.check_output(["git", "-C", str(ROOT), "hash-object", rel], text=True).strip()
            for rel in code_files()}


def build() -> dict:
    blobs = json.dumps(code_blobs(), indent=4)
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "cells": [cell("markdown", INTRO), cell("code", CONFIG_CELL.replace("{code_blobs}", blobs)),
                      cell("code", SETUP_CELL), cell("code", INPUT_CELL), cell("code", RUN_CELL),
                      cell("code", CHECK_CELL), cell("code", EXPORT_CELL)]}


if __name__ == "__main__":
    NOTEBOOK.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(NOTEBOOK)

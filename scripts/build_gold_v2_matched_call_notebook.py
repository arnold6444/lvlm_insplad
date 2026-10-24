#!/usr/bin/env python3
"""Write the matched-call gold Colab notebook, pinning the git blob IDs of every code file it runs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/lvlm_gold_v2_matched_call_colab.ipynb"
EXTRA_FILES = ("configs/gold_v2_matched_call_v1.json", "src/insplad_benchmark/hf_adapters.py")


PROBE = """
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(root / "scripts"), str(root / "src")]
import run_gold_v2_matched_call
files = {Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, "__file__", None)}
print(json.dumps(sorted(f.relative_to(root).as_posix() for f in files if f.suffix == ".py" and root in f.parents)))
"""


def code_files() -> list[str]:
    """Every repository module the runner imports (fresh interpreter), plus its lock and the lazy adapter."""
    found = json.loads(subprocess.check_output([sys.executable, "-c", PROBE, str(ROOT)], text=True))
    return sorted(set(found) | set(EXTRA_FILES))

INTRO = """# gold 호출 수 맞춘 대조 실험 (matched-call control)

**새 L4/A100 런타임 → 런타임 / 모두 실행 → Drive 승인 → 결과 ZIP 전달.** 정답 입력·박스 그리기 없음. GitHub 토큰 없이 공개 저장소를 받습니다.

질문: gold에서 검증 agent가 좋아진 이유가 **구조(정의·유무→위치 단계)** 때문인지, 단순히 **더 많이 물어봐서**인지.

- 기존 세 조건(직접 질문 / 기존 agent / 수정 후보)은 PR56·57 gold와 같은 프롬프트입니다. Drive에 기존 `gold_v2_q23_candidate_v2` 결과가 있으면 이미지 바이트와 프롬프트가 같은 답만 그대로 재사용합니다.
- 새 대조군 `matched_reask`: 이미지마다 기존 agent와 **같은 호출 수**를 쓰지만 새 근거는 없습니다. Q2는 자기 답을 보여주고 한 번 다시 묻기(정의 없음), Q3는 단계형 Q3가 쓴 호출 수(1~2회)만큼 자기 답을 보여주고 다시 묻기.
- Q1 재질문 0회. gold 점수로 설정을 고르거나 바꾸지 않습니다(분석용 대조). Q3는 유무만 채점하고 위치 정확도는 null.

새 호출: 기존 결과(1,606답)를 재사용하면 **622회**(Q2 246 + Q3 376). 이전 속도 기준 추론 약 10분, 설치·모델 로딩 포함 대략 20~30분. 기존 결과가 없으면 최대 약 2,000회.
20단계마다 Drive 저장. 중단되면 **추론 셀만 다시 실행**하면 이어서 합니다. 실패·무효가 남은 조건은 비교에서 빠지고 완료로 인정하지 않습니다.

출력: `MyDrive/lvlm_insplad_results/gold_v2_matched_call_v1/` 와 `gold_matched_call_results.zip`. 기존 결과 폴더는 읽기만 합니다.
"""

CONFIG_CELL = """REPO = "arnold6444/lvlm_insplad"
EXECUTION_REF = "main"
CODE_BLOBS = {code_blobs}
QA_FOLDER = "lvlm_insplad_qa/gold_v2_final"
OLD_GOLD_FOLDER = "lvlm_insplad_results/gold_v2_main_rerun"
CANDIDATE_RUN_FOLDERS = ["lvlm_insplad_results/gold_v2_q23_candidate_v2", "lvlm_insplad_results/gold_v2_q23_candidate_v1"]
RUN_FOLDER = "lvlm_insplad_results/gold_v2_matched_call_v1"
CHECKPOINT_EVERY = 20
UNSUP_ZIP_NAME = "unsupervised_anomaly_detection.zip"
UNSUP_ZIP_SIZE = 1_170_967_719
UNSUP_ZIP_PATH = ""
SUP_ZIP_NAME = "supervised_fault_classification.zip"
SUP_ZIP_SIZE = 874_753_167
SUP_ZIP_PATH = ""
"""

SETUP_CELL = '''# Drive 연결 + 코드 받기(고정 blob 확인) + 패키지
import base64, json, os, subprocess, sys
from pathlib import Path
from google.colab import drive, userdata

drive.mount("/content/drive", force_remount=False)
DRIVE_ROOT = Path("/content/drive/MyDrive")
QA_DIR = DRIVE_ROOT / QA_FOLDER
OLD_GOLD_DIR = DRIVE_ROOT / OLD_GOLD_FOLDER
RUN_ROOT = DRIVE_ROOT / RUN_FOLDER
if not RUN_ROOT.name.startswith("gold_v2_matched_call"):
    raise ValueError("새 실험 전용 RUN_FOLDER를 사용하세요.")
REPO_DIR = Path("/content/lvlm_matched_call_code")

def run(cmd, *, env=None, check=True):
    """Run a command and stream its output into this cell."""
    cmd = [str(x) for x in cmd]
    print("+", " ".join(cmd[:12]) + (" ..." if len(cmd) > 12 else ""), flush=True)
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    if check and p.returncode:
        raise RuntimeError(f"command failed with exit code {p.returncode} (마지막 출력 위 참고)")
    return p.returncode

git_env = os.environ.copy()
git_env["GIT_TERMINAL_PROMPT"] = "0"
try:
    token = userdata.get("GITHUB_TOKEN")
except Exception:
    token = ""
if token:  # optional: only needed if the repository becomes private again
    auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    git_env.update({"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.extraHeader",
                    "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {auth}"})
if not REPO_DIR.exists():
    run(["git", "clone", "--depth", "1", "--branch", EXECUTION_REF, f"https://github.com/{REPO}.git", REPO_DIR], env=git_env)

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

INPUT_CELL = '''# 입력 준비: 원본 ZIP, FROZEN gold QA, 기존 참조·기준선·후보 결과(읽기 전용)
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

import run_gold_v2_matched_call as matched
candidate = matched.candidate
summary = json.loads((QA_DIR / "build_summary.json").read_text(encoding="utf-8"))
if summary.get("quality_status") != "FROZEN" or summary.get("final_master_images") != 246:
    raise RuntimeError("원래 FROZEN gold246 QA가 필요합니다.")
ALL_GOLD_ROWS = candidate.agent.v2.load_gold_qa_v2(QA_DIR)
GOLD_ROWS = candidate.gold_rows(ALL_GOLD_ROWS)
if len(GOLD_ROWS) != 492:
    raise RuntimeError("gold Q1/Q2 문항수가 다릅니다.")
GOLD_SOURCE_HASH = candidate.old.diagnostic.digest(ALL_GOLD_ROWS)
LOCK_PATH = REPO_DIR / "configs/gold_v2_matched_call_v1.json"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
GOLD_QA = RUN_ROOT / "retrieval_input_qa.jsonl"
if GOLD_QA.is_file() and candidate.agent.read_jsonl(GOLD_QA) != GOLD_ROWS:
    raise RuntimeError("QA가 바뀌었습니다. 기존 결과를 유지하고 새 RUN_FOLDER를 사용하세요.")
if not GOLD_QA.is_file():
    candidate.save_records(GOLD_QA, GOLD_ROWS)

# Reuse the saved candidate run with the most steps; its references are the ones those answers used.
REUSE_DIR = REUSE_STEPS = REUSE_MANIFEST = None
for folder in CANDIDATE_RUN_FOLDERS:
    steps = DRIVE_ROOT / folder / "predictions/internvl3_5_8b/steps.jsonl"
    manifest = DRIVE_ROOT / folder / "image_manifest.csv"
    if steps.is_file() and manifest.is_file():
        n = len(steps.read_text(encoding="utf-8").splitlines())
        if REUSE_STEPS is None or n > len(REUSE_STEPS.read_text(encoding="utf-8").splitlines()):
            REUSE_DIR, REUSE_STEPS, REUSE_MANIFEST = DRIVE_ROOT / folder, steps, manifest
print("기존 gold 후보 결과 재사용:", REUSE_STEPS or "없음 → 기존 세 조건도 새로 생성")

ref_sources = ([REUSE_DIR / "refs_gold" / "retrieved_refs.csv"] if REUSE_DIR else []) + [OLD_GOLD_DIR / "refs_gold" / "retrieved_refs.csv"]
new_refs_dir = RUN_ROOT / "refs_gold"
REFS_CSV = next((p for p in ref_sources if p.is_file()), new_refs_dir / "retrieved_refs.csv")
if not REFS_CSV.is_file():
    run([sys.executable, REPO_DIR / "scripts/build_retrieved_refs.py", "--qa", GOLD_QA,
         "--sup-zip", SUP_ZIP, "--unsup-zip", UNSUP_ZIP, "--out-dir", new_refs_dir])
REFS = candidate.read_references(REFS_CSV, GOLD_ROWS)
BASELINE_PREDICTIONS = OLD_GOLD_DIR / "predictions/internvl3_5_8b/normal_ref_retrieved_diffdate.jsonl"
if BASELINE_PREDICTIONS.is_file():
    reused, _ = candidate.cached_baseline(BASELINE_PREDICTIONS, GOLD_ROWS, REFS, "internvl3_5_8b")
    print("기존 gold 기준선 유효 답:", len(reused), "/492")
else:
    print("별도 기준선 파일 없음: 후보 결과의 기준선 답을 재사용하거나 새로 만듭니다.")
print("정상 참조:", REFS_CSV)
'''

RUN_CELL = '''print("code commit:", verify_code())
# 추론 / 자동 채점. 중단되거나 실패가 남으면 이 셀만 다시 실행.
cmd = [sys.executable, REPO_DIR / "scripts/run_gold_v2_matched_call.py",
       "--qa-dir", QA_DIR, "--refs-csv", REFS_CSV,
       "--sup-zip", SUP_ZIP, "--unsup-zip", UNSUP_ZIP,
       "--out-dir", RUN_ROOT, "--config", LOCK_PATH,
       "--checkpoint-every", str(CHECKPOINT_EVERY)]
if BASELINE_PREDICTIONS.is_file():
    cmd += ["--baseline-predictions", BASELINE_PREDICTIONS]
if REUSE_STEPS:
    cmd += ["--reuse-steps", REUSE_STEPS, "--reuse-manifest", REUSE_MANIFEST]
import uuid
INVOCATION_ID = uuid.uuid4().hex  # 아래 확인 셀은 이 실행 ID가 적힌 결과 파일만 받아들입니다.
RUN_RC = None  # 셀이 중간에 끊기면 이전 실행의 종료 코드가 남지 않게 비워 둡니다.
RUN_RC = run(cmd + ["--invocation-id", INVOCATION_ID], check=False)
print("종료 코드:", RUN_RC, "실행 ID:", INVOCATION_ID)
# The original gold QA must remain identical.
assert GOLD_SOURCE_HASH == candidate.old.diagnostic.digest(candidate.agent.v2.load_gold_qa_v2(QA_DIR))
'''

CHECK_CELL = '''# 자동 결과 확인: 이번 실행이 종료 코드 0으로 끝나고, 결과 파일이 같은 실행에서 나왔을 때만 성공 인정
pd.set_option("display.max_rows", 100)
if globals().get("RUN_RC") != 0:
    raise RuntimeError(f"이번 추론 실행이 실패했습니다(종료 코드 {globals().get('RUN_RC')}). 이전 실행의 결과 파일은 쓰지 않습니다. 추론 셀을 다시 실행하세요.")
status = json.loads((RUN_ROOT / "run_status.json").read_text(encoding="utf-8"))
meta = json.loads((RUN_ROOT / "run_metadata.json").read_text(encoding="utf-8"))
if status.get("invocation_id") != INVOCATION_ID or meta.get("invocation_id") != INVOCATION_ID:
    raise RuntimeError("결과 파일이 이번 실행에서 나온 것이 아닙니다. 추론 셀을 다시 실행하세요.")
display(pd.DataFrame(status["policies"]))
if (not status.get("complete") or not meta.get("complete") or meta.get("row_errors") or meta.get("invalid_answers")
    or len(status["policies"]) != 4 or not all(p["eligible_for_comparison"] for p in status["policies"])):
    raise RuntimeError("실패/무효/누락이 남았습니다. 추론 셀을 다시 실행하세요. 실패 조건은 비교에서 제외됩니다.")
answers = pd.read_csv(RUN_ROOT / "answers.csv")
q1 = answers[answers.question_type == "anomaly_detection"]
assert len(q1) == meta["images"]*4 and (q1.predicted_choice == q1.cached_baseline_choice).all(), "Q1 답 변경/누락"
budget = pd.read_csv(RUN_ROOT / "call_budget.csv").set_index("policy")
assert (budget.loc["matched_reask", ["q2_calls", "q3_calls"]] == budget.loc["previous_agent", ["q2_calls", "q3_calls"]]).all(), "호출 수 불일치"
print("Q1 유지:", len(q1), "행 모두 동일. 기존 agent와 대조군 호출 수 동일")
print("새 호출:", meta["called_this_run"], "기존 후보 결과 재사용:", meta["reused_candidate_steps"],
      "이 폴더 저장분 재사용:", meta["reused_calls"], "기존 기준선 반입:", meta["imported_baseline_calls"])
display(budget.round(3))
lb = pd.read_csv(RUN_ROOT / "leaderboard.csv")
display(lb[["policy", "q1_anomaly_f1", "q2_defect_macro_f1", "q2_defect_balanced_accuracy", "q2_accuracy", "joint_accuracy",
            "q3_presence_balanced_accuracy", "q3_normal_false_region_rate", "q3_anomaly_false_none_rate"]].round(3))
print("결함별 Q2 정답수/지원수")
display(pd.read_csv(RUN_ROOT / "q2_by_defect.csv"))
sig = pd.read_csv(RUN_ROOT / "significance.csv")
print("핵심 비교: 기존 agent − 호출 수 맞춘 대조 (paired bootstrap 95% CI)")
display(sig[(sig.baseline == "matched_reask") & (sig.condition == "previous_agent")].round(3))
print("전체 사전 고정 비교")
display(sig.round(3))
print("완료. gold에서 설정 재선택 없음. 결과 ZIP을 전달하세요:", RUN_ROOT)
'''

EXPORT_CELL = '''# 성공/실패 산출물 모두 내려받을 수 있습니다. 실패 ZIP은 완료 결과가 아닙니다.
from zipfile import ZipFile, ZIP_DEFLATED
from google.colab import files
archive = Path("/content/gold_matched_call_results.zip")
with ZipFile(archive, "w", ZIP_DEFLATED) as z:
    for p in RUN_ROOT.rglob("*"):
        if p.is_file() and "_images" not in p.relative_to(RUN_ROOT).parts:
            z.write(p, p.relative_to(RUN_ROOT))
files.download(str(archive))
'''


def code_blobs() -> dict[str, str]:
    """Blob IDs as committed (git applies the same line-ending filter Colab's checkout sees)."""
    return {rel: subprocess.check_output(["git", "-C", str(ROOT), "hash-object", rel], text=True).strip()
            for rel in code_files()}


def cell(kind, source):
    lines = source.splitlines(keepends=True)
    base = {"cell_type": kind, "metadata": {}, "source": lines}
    return {**base, "execution_count": None, "outputs": []} if kind == "code" else base


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

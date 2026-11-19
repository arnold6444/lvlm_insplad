"""Build the one-run Colab for LoRA SFT of the tool-choice agent; this does not execute training or inference."""
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
import sys
import textwrap
import zipfile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_agent_tool_pilot_notebook as pilot_nb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
cell = pilot_nb.cell
GOLD = "sft_v2_human_gold"         # the human-box run plus gold scoring in the same single execution
NOTEBOOKS = {"sft_v1": "notebooks/lvlm_agent_sft_lora_colab.ipynb",
             "sft_v2_human": "notebooks/lvlm_agent_sft_lora_human_boxes_colab.ipynb",
             GOLD: "notebooks/lvlm_agent_sft_lora_human_boxes_gold_colab.ipynb"}
# The frozen 133-photo location set is embedded in the notebook, so gold scoring needs no extra Drive file.
LOC_QA = "results/latest/gold_v2_loc_mmad/localization_mmad_qa_133.jsonl"
# Drive folders written by the two box-label Colabs, and the photos each must hold.
LABEL_RUNS = {"sft_train_location_annotation_sft_v1": 300, "sft_train_location_annotation_sft_v2_polymer": 60}
# annotations.json of the 300 boxes checked on 2026-11-16; a different file only warns, the builder re-checks every box.
FIRST_LABELS_SHA256 = "f4c9ce51e23c8fb6050dac00462f80e821b2b85c3f2edb0c68b540839c607270"

HUMAN_INTRO = """
    # 도구 선택 에이전트 LoRA SFT 2차 — 사람이 그린 박스로 다시 학습
    **A100 또는 L4(20 GiB 이상) GPU**에서 **모두 실행**하세요. 4bit 양자화는 쓰지 않습니다(BF16).

    순서: ① 박스 표시가 끝났는지 확인 → ② 학습 데이터 만들기 → ③ Qwen3.5-4B LoRA 학습 → ④ dev240 평가 → ⑤ 비교표와 결과 ZIP 다운로드.

    - 1차(`sft_v1`)와 다른 점은 두 가지입니다. 확대 박스와 위치 답을 **사람이 그린 박스**에서 가져오고, polymer 절연체 녹 사진 60장을 학습에 넣습니다.
    - 결함 사진: 정상 사진 요청 → 사람 박스 확대 → 답. 정상 사진: 정상 사진 요청 → 답(확대 없음).
    - 박스 표시 Colab 두 개(300장, polymer 60장)가 **모두 끝나 있어야** 시작합니다. 덜 끝났으면 첫 셀에서 몇 장 남았는지 알려 주고 멈춥니다.
    - 학습 설정과 dev240 평가는 1차와 같습니다. 학습 전 모델과 1차 모델은 다시 돌리지 않고 Drive의 지난 결과를 표에 같이 보여 줍니다.
    - dev240에는 위치 정답이 거의 없어 Q3는 유무만 채점합니다. 위치를 배웠는지는 확대 박스가 한 곳에 몰리는지로 먼저 확인합니다.

    예상 시간(L4, 1차 실측 기준): 학습 6~7시간, 평가 2시간 안팎. 끊기면 같은 노트북을 다시 **모두 실행**하세요. 학습 checkpoint와 평가 호출을 이어서 씁니다.
    """

HUMAN_CHECK = '''
    # 박스 표시가 끝났는지 먼저 확인합니다. 덜 끝난 사진이 있으면 학습을 시작하지 않습니다.
    import hashlib
    LABEL_DIRS = [RESULTS / name for name in LABEL_RUNS]
    for folder, expected in zip(LABEL_DIRS, LABEL_RUNS.values()):
        if not (folder / "manifest.json").is_file() or not (folder / "annotations.json").is_file():
            raise RuntimeError(f"박스 표시 결과가 없습니다: {folder}. 표시 Colab을 먼저 끝내세요.")
        targets = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["targets"]
        records = json.loads((folder / "annotations.json").read_text(encoding="utf-8"))["records"]
        done = sum(records.get(t["core_id"], {}).get("status") in ("boxed", "unlocatable", "disputed") for t in targets)
        print(f"{folder.name}: 표시 {done}/{len(targets)}장")
        if len(targets) != expected or done != len(targets):
            raise RuntimeError(f"{folder.name}: 표시가 {done}/{len(targets)}장입니다. 표시 Colab에서 남은 사진을 끝낸 뒤 "
                               "이 노트북을 다시 모두 실행하세요. 방금 끝냈다면 Drive 반영까지 1~2분 기다리세요.")
    if hashlib.sha256((LABEL_DIRS[0] / "annotations.json").read_bytes()).hexdigest() != FIRST_LABELS_SHA256:
        print("경고: 300장 표시가 2026-11-16에 확인한 파일과 다릅니다(나중에 고친 사진이 있음). 학습은 계속하고, "
              "쓴 표시 파일은 결과 ZIP에 들어갑니다.")
    PREVIOUS = {"학습 전 (1차 실행)": RESULTS / f"agent_tool_pilot_dev{N_IMAGES}_qwen3_5_4b_base_sft_v1",
                "lora sft_v1 (자동 박스)": RESULTS / f"agent_tool_pilot_dev{N_IMAGES}_qwen3_5_4b_lora_sft_v1"}
    '''

HUMAN_DATA = '''
    #@title 2. 학습 데이터 만들기 (사람 박스·궤적)
    verify_code()
    run([sys.executable, REPO_DIR / "scripts/build_agent_sft_data.py", "--dev-qa", DEV_QA,
         "--sup-zip", SUP_ZIP, "--unsup-zip", UNSUP_ZIP, "--out-dir", DATA_DIR,
         "--per-class", PER_CLASS, "--device", "cuda", "--add-polymer-corrosion", "--human-boxes", *LABEL_DIRS])
    built = json.loads((DATA_DIR / "build_summary.json").read_text(encoding="utf-8"))
    if built["protocol"] != "agent_sft_train_v2_human_boxes" or built["trajectories"].get("crop") != sum(
            r["status"].get("boxed", 0) for r in built["human_box_runs"]):
        raise RuntimeError("사람 박스로 만든 학습 데이터가 아닙니다. 결과를 쓰지 마세요.")
    print(json.dumps({k: built[k] for k in ("images", "samples", "trajectories", "q3_cells", "distinct_crop_boxes",
                                            "grown_small_boxes", "unlocated_photos_left_out")}, ensure_ascii=False))
    '''

HUMAN_EVAL = '''
    #@title 4. 학습 후 dev240 평가 (지난번과 같은 4개 조건)
    verify_code()
    EXIT_CODES = {"lora": run([sys.executable, REPO_DIR / "scripts/run_agent_tool_pilot.py", "--dev-qa", DEV_QA,
                               "--refs-csv", REFS_CSV, "--sup-zip", SUP_ZIP, "--unsup-zip", UNSUP_ZIP,
                               "--model", "qwen3_5_4b", "--n-images", N_IMAGES, "--max-calls", 3,
                               "--out-dir", SFT_RUN, "--adapter-dir", TRAIN_DIR / "adapter"], check=False)}
    print(EXIT_CODES)
    '''

HUMAN_REPORT = '''
    #@title 5. 비교표와 결과 ZIP
    from collections import Counter
    import pandas as pd
    from google.colab import files
    def receipt(root, name):
        p = root / name
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    status, meta = receipt(SFT_RUN, "run_status.json"), receipt(SFT_RUN, "run_metadata.json")
    VERIFIED = bool(EXIT_CODES.get("lora") == 0 and status.get("complete") and meta.get("execution") == "real_hf"
                    and status.get("run_id") == meta.get("run_id")
                    and meta.get("benchmark_commit") == EXECUTION_COMMIT and meta.get("training") is True)
    print("lora", RUN_TAG, "완료 확인" if VERIFIED else "실행 오류 또는 완료 미확인", json.dumps(status, ensure_ascii=False))
    boards = []
    for label, root in ((f"lora {RUN_TAG} (사람 박스)", SFT_RUN), *PREVIOUS.items()):
        board = root / "leaderboard.csv"
        if board.is_file() and board.stat().st_size:
            boards.append(pd.read_csv(board).assign(model=label))
    if boards:
        table = pd.concat(boards)
        cols = ["model", "policy", "q1_balanced_accuracy", "q2_defect_macro_f1", "q2_invalid_rate", "joint_accuracy",
                "false_defect_rate", "q3_presence_balanced_accuracy", "q3_normal_false_region_rate", "invalid_episodes", "calls"]
        display(table[[c for c in cols if c in table]])
        print("지난 실행의 행은 Drive에 남은 결과를 읽은 것입니다(이번에 다시 돌리지 않음).")
    # 1차 모델은 확대 365회 중 341회가 같은 왼쪽 위 구석이었습니다. 이번에는 박스가 사진마다 달라졌는지 봅니다.
    crops = Counter()
    for p in sorted((SFT_RUN / "episodes/adaptive_tools").glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                action = json.loads(json.loads(line)["output"])
            except (ValueError, KeyError, TypeError):
                continue
            if isinstance(action, dict) and action.get("action") == "crop":
                crops[json.dumps(action.get("bbox"))] += 1
    if crops:
        print(f"확대 요청 {sum(crops.values())}회 / 서로 다른 박스 {len(crops)}개 / 가장 흔한 박스 {crops.most_common(1)[0][1]}회")
    train_meta = receipt(TRAIN_DIR, "train_metadata.json")
    print("학습:", {k: train_meta.get(k) for k in ("optimizer_steps", "val_loss_before", "val_loss_after")})
    print("Q3 위치 정확도는 미평가입니다(dev에 위치 정답이 거의 없음). dev240은 여러 번 본 개발 세트입니다.")
    (RESULTS / f"agent_sft_{RUN_TAG}_execution.json").write_text(json.dumps(dict(
        exit_codes=EXIT_CODES, verified=dict(lora=VERIFIED), execution_commit=EXECUTION_COMMIT,
        distinct_crop_boxes=len(crops), crop_requests=sum(crops.values())), indent=2), encoding="utf-8")
    archive = RESULTS / f"agent_sft_{RUN_TAG}_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(RESULTS / f"agent_sft_{RUN_TAG}_execution.json", "notebook_execution.json")
        for folder in LABEL_DIRS:                      # the exact label files this run trained on
            for name in ("manifest.json", "annotations.json"):
                z.write(folder / name, f"labels/{folder.name}/{name}")
        for label, root in (("data", DATA_DIR), ("train", TRAIN_DIR), ("lora", SFT_RUN)):
            if not root.is_dir():
                continue
            for p in sorted(root.rglob("*")):
                rel = p.relative_to(root)
                private = any(part.startswith("_") for part in rel.parts) or rel.parts[0] == "checkpoint"
                if p.is_file() and not private and p.suffix != ".safetensors":
                    z.write(p, f"{label}/{rel.as_posix()}")
    files.download(str(archive))
    print("평가 완료" if VERIFIED else "미완료(진단용)", "ZIP:", archive)
    '''

GPU_GATE = '''
    # GPU를 Drive·pip보다 먼저 확인합니다. 런타임이 잘못 잡혔으면 몇 초 만에 멈추고, 지금 연결된 GPU를 알려 줍니다.
    try:
        probe = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                               capture_output=True, text=True)
        GPU_NOW = probe.stdout.strip().splitlines()[0] if probe.returncode == 0 and probe.stdout.strip() else ""
    except OSError:
        GPU_NOW = ""
    GPU_NOW = GPU_NOW or "GPU 없음(CPU 런타임)"
    found = re.search(r"(\\d+) MiB", GPU_NOW)
    if not found or int(found.group(1)) < 20 * 1024:
        raise RuntimeError(f"지금 연결된 GPU: {GPU_NOW}. VRAM 20 GiB 이상인 L4 또는 A100이 필요합니다(양자화로 우회하지 않음). "
                           "런타임 > 런타임 유형 변경에서 L4를 고르고 저장한 다음, 런타임 > 런타임 연결 해제 및 삭제를 누르고 "
                           "다시 모두 실행하세요. L4를 골랐는데도 T4나 GPU 없음으로 나오면 컴퓨팅 단위가 남아 있는지 확인하세요.")
    print("GPU:", GPU_NOW)
    '''

GOLD_INTRO = """
    # 도구 선택 에이전트 LoRA SFT 2차 + 시험(gold) 채점 — 한 번에 실행
    **L4 또는 A100(20 GiB 이상) GPU**에서 **모두 실행**을 한 번 누르면 학습부터 시험 채점까지 끝납니다. 4bit 양자화는 쓰지 않습니다(BF16).

    순서: ① GPU·박스 표시·시험 입력 확인 → ② 학습 데이터 만들기 → ③ Qwen3.5-4B LoRA 학습 → ④ 시험(gold) 채점: 2차 모델 → ⑤ 작은 분류기 비교 → ⑥ 시험 채점: 학습 전 모델 → ⑦ dev240 평가: 2차 모델 → ⑧ 시험 채점: 1차 모델 → ⑨ 시험 채점: 2차 모델의 대조 조건 → ⑩ 표와 결과 ZIP.

    - **Drive에 이미 끝난 단계는 다시 하지 않습니다.** 2차 학습 데이터와 학습된 어댑터가 Drive에 있으면(2026-11-17 실행) ②와 ③은 몇 초 만에 지나가고 바로 채점으로 갑니다.
    - 중요한 것부터 돌립니다. 중간에 끊겨도 끝난 단계의 결과는 Drive에 남고, 같은 노트북을 다시 **모두 실행**하면 이어서 합니다.
    - **결과 ZIP은 단계가 하나 끝날 때마다 Drive에 다시 씁니다.** 끝까지 못 가도 그때까지의 ZIP이 `lvlm_insplad_results`에 있습니다. ZIP에는 모델이 낸 답 원문이 전부 들어 있어, 표에 없는 숫자도 나중에 GPU 없이 다시 계산할 수 있습니다.
    - 모든 단계의 출력은 Drive의 `agent_sft_gold_sft_v2_human/logs`에도 남습니다. 화면이 닫혀 오류 글이 사라져도 ZIP 안의 로그로 원인을 볼 수 있습니다.
    - ⑤ 작은 분류기 비교는 같은 학습 사진 660장과 같은 라벨로 작은 전용 분류기(DINOv2 특징 + 선형 분류기)를 학습해, 같은 시험 문제의 Q1·Q2를 풀게 합니다. "LVLM 대신 작은 분류기를 쓰면?"에 답하기 위한 것이고, 위치(Q3)는 답하지 못합니다. 설정은 고르지 않고 기본값 하나로 한 번만 학습합니다.
    - ⑨ 대조 조건은 "도구가 도움이 됐다"는 결과가 나올 때 필요합니다. 가운데를 늘 같은 크기로 확대해 준 경우, 도구 없이 같은 횟수만큼 다시 물은 경우와 비교합니다.
    - 첫 셀이 몇 초 안에 GPU를 확인합니다. L4가 아니면 지금 잡힌 GPU 이름을 알려 주고 멈춥니다.
    - 학습을 시작하기 전에 시험 채점에 쓸 파일을 끝까지 점검합니다. 문제가 있으면 학습 시간을 쓰기 전에 멈춥니다.
    - 시험 채점 규칙은 실행 전에 고정했습니다(`docs/agent_sft_gold_eval.md`). gold로 설정을 고르지 않고, 돌린 조건은 전부 보고합니다.
      - 주 결과: 2차 모델, 한 번에 답 조건, 학습 사진과 같은 드론 사진을 뺀 gold 사진.
      - 위치: 사람 박스가 있는 결함 133장에서 답한 칸이 맞는지 채점하고, "늘 가운데"라고 답한 경우와 비교합니다.
    - 2차 학습이 1차와 다른 점: 확대 박스와 위치 답을 **사람이 그린 박스**에서 가져오고, polymer 절연체 녹 사진 60장을 학습에 넣습니다.
    - 설정 셀의 `RUN_CLASSIFIER_BASELINE`, `RUN_GOLD_BASE`, `RUN_DEV240`, `RUN_GOLD_SFT_V1`, `RUN_GOLD_CONTROLS`를 `False`로 바꾸면 그 단계를 건너뜁니다. 건너뛴 단계는 표에 없습니다.
    - dev240 평가는 새 폴더에 두 조건으로 돌립니다. 11/17에 네 조건으로 돌리다 끊긴 실행은 건드리지 않고, 그때 저장된 답을 결과 ZIP에 같이 넣습니다.
    - 다른 버전의 노트북으로 시작한 시험 채점이 Drive에 있으면 그 폴더는 그대로 두고 새 폴더에서 다시 합니다(그 폴더도 ZIP에 들어갑니다).

    예상 시간(L4, 1차 실측에서 어림한 값이고 아직 재지 않았습니다): 시험 채점 2차 모델 1.5시간, 작은 분류기 15분, 학습 전 모델 1시간 이내, dev240 1.5시간, 시험 채점 1차 모델 1.5시간, 대조 조건 1시간. 학습이 Drive에 끝나 있으면 합계 6~7시간, 학습부터 하면 6~7시간이 더 듭니다.
    """

GOLD_PREP = '''
    # 시험(gold) 채점 입력을 학습 전에 준비합니다: FROZEN gold QA, 위치 정답 133장(이 노트북에 들어 있음), 정상 참조.
    import base64, zlib
    GOLD_QA_DIR = DRIVE / GOLD_QA_FOLDER
    gold_summary = json.loads((GOLD_QA_DIR / "build_summary.json").read_text(encoding="utf-8"))
    if gold_summary.get("quality_status") != "FROZEN" or gold_summary.get("final_master_images") != 246:
        raise RuntimeError(f"FROZEN gold 246장 QA가 필요합니다: {GOLD_QA_DIR}")
    GOLD_MASTER = sorted(GOLD_QA_DIR.glob("master_qa_*.jsonl"))[-1]
    GOLD_ROOT = RESULTS / f"agent_sft_gold_{RUN_TAG}"
    (GOLD_ROOT / "inputs").mkdir(parents=True, exist_ok=True)
    LOC_QA = GOLD_ROOT / "inputs/localization_mmad_qa_133.jsonl"
    loc_bytes = zlib.decompress(base64.b64decode(LOC_QA_B64))
    if hashlib.sha256(loc_bytes).hexdigest() != LOC_QA_SHA256:
        raise RuntimeError("노트북에 들어 있는 위치 정답 파일이 손상됐습니다. 노트북을 다시 받으세요.")
    if not LOC_QA.is_file() or LOC_QA.read_bytes() != loc_bytes:
        LOC_QA.write_bytes(loc_bytes)
    gold_images = {json.loads(line)["image"] for line in GOLD_MASTER.read_text(encoding="utf-8").splitlines() if line.strip()}
    def covers_gold(path):
        if not path.is_file():
            return False
        with path.open(encoding="utf-8-sig", newline="") as stream:
            found = {r["image"]: r.get("retrieved_diffdate") for r in csv.DictReader(stream)}
        return all(found.get(image) for image in gold_images)
    GOLD_REFS = next((p for p in (RESULTS / "gold_v2_matched_call_v1/refs_gold/retrieved_refs.csv",
                                  RESULTS / "gold_v2_main_rerun/refs_gold/retrieved_refs.csv",
                                  GOLD_ROOT / "inputs/refs_gold/retrieved_refs.csv") if covers_gold(p)), None)
    if GOLD_REFS is None:
        run([sys.executable, REPO_DIR / "scripts/build_retrieved_refs.py", "--qa", GOLD_MASTER, "--sup-zip", SUP_ZIP,
             "--unsup-zip", UNSUP_ZIP, "--out-dir", GOLD_ROOT / "inputs/refs_gold", "--device", "cuda"])
        GOLD_REFS = GOLD_ROOT / "inputs/refs_gold/retrieved_refs.csv"
    V1_ADAPTER = RESULTS / "agent_sft_lora_qwen3_5_4b_sft_v1/adapter"
    V1_TRAIN_QA = RESULTS / f"agent_sft_data_pc{PER_CLASS}_sft_v1/sft_train_qa.jsonl"
    HAS_V1 = (V1_ADAPTER / "adapter_model.safetensors").is_file() and (V1_ADAPTER / "adapter_config.json").is_file()
    GOLD_RUNS = {"gold_lora": GOLD_ROOT / f"agent_sft_gold_eval_lora_{RUN_TAG}",
                 "gold_base": GOLD_ROOT / "agent_sft_gold_eval_base",
                 "gold_v1": GOLD_ROOT / "agent_sft_gold_eval_lora_sft_v1"}
    # dev240 goes to its own folder: SFT_RUN may hold the four-condition run of 2026-11-17 that stopped midway, and a
    # run made by other source cannot be resumed by this runner. That folder is left as it is and zipped with the rest.
    DEV_RUN = RESULTS / f"agent_tool_pilot_dev{N_IMAGES}_qwen3_5_4b_lora_{RUN_TAG}_two_conditions"
    CLASSIFIER_RUN = GOLD_ROOT / "sft_label_classifier_baseline"
    print("gold 참조:", GOLD_REFS)
    print("1차 모델:", "있음" if HAS_V1 else "Drive에 없음. 1차 모델 시험 채점은 건너뜁니다.")
    print("이미 끝난 학습:", "있음. 학습은 건너뜁니다." if (TRAIN_DIR / "adapter/adapter_model.safetensors").is_file()
          else "없음. 학습부터 합니다(6~7시간).")
    '''

# Helpers of the one-run notebook. The result ZIP is rewritten after every step and holds every raw model output,
# so an interrupted session still leaves a usable ZIP and no number has to be produced by a second GPU run.
GOLD_TOOLS = '''
    # 로그, 결과 ZIP, 채점 실행 도우미입니다. 결과 ZIP은 단계가 끝날 때마다 다시 씁니다.
    import io, shutil, time
    LOG_DIR = GOLD_ROOT / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG = LOG_DIR / time.strftime("run_%Y%m%d_%H%M%S.log")
    def keep_log(text):
        try:
            with RUN_LOG.open("a", encoding="utf-8") as stream:
                stream.write(text)
        except OSError:
            pass
    keep_log(f"notebook source {EXECUTION_COMMIT}, GPU {GPU_NOW}\\n")
    for name, command in (("pip_freeze.txt", [sys.executable, "-m", "pip", "freeze"]), ("nvidia_smi.txt", ["nvidia-smi"])):
        try:
            (LOG_DIR / name).write_text(subprocess.run(command, capture_output=True, text=True).stdout, encoding="utf-8")
        except OSError:
            pass
    EXIT_CODES, EARLIER = {}, {}
    ARCHIVE = RESULTS / f"agent_sft_{RUN_TAG}_gold_results.zip"
    TRAINED = {"gold_lora": True, "gold_base": False, "gold_v1": True, "dev_lora": True}

    def receipt(root, name):
        p = root / name
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    def gold_run(key, extra=()):
        """한 모델을 시험(gold) 문제로 채점합니다. 다른 버전으로 시작한 폴더는 이어 갈 수 없어서, 그대로 두고 새 폴더를 씁니다."""
        command = [*GOLD_RUNNER, *extra, "--out-dir", GOLD_RUNS[key]]
        if run([*command, "--inputs-only"], check=False) == 3:
            EARLIER[f"{key}_earlier_run"] = GOLD_RUNS[key]
            GOLD_RUNS[key] = GOLD_RUNS[key].with_name(f"{GOLD_RUNS[key].name}_{EXECUTION_COMMIT[:7]}")
            command[-1] = GOLD_RUNS[key]
            print("이 폴더에는 다른 버전으로 시작한 채점이 있어 새 폴더에서 다시 합니다:", GOLD_RUNS[key].name)
        return run(command, check=False)

    def verified_runs():
        """이번 노트북의 소스로 끝까지 돈 단계."""
        found = {}
        for key, root in {**GOLD_RUNS, "dev_lora": DEV_RUN}.items():
            if key not in EXIT_CODES:
                continue
            # If the controls stopped midway, the main step's own receipt is the copy the runner kept before them.
            kept = root / "main_conditions_before_controls"
            now = receipt(root, "run_status.json")
            source = root if now.get("complete") or not (kept / "run_status.json").is_file() else kept
            status, meta = receipt(source, "run_status.json"), receipt(source, "run_metadata.json")
            found[key] = bool(EXIT_CODES[key] == 0 and status.get("complete") and meta.get("execution") == "real_hf"
                              and status.get("run_id") == meta.get("run_id")
                              and meta.get("benchmark_commit") == EXECUTION_COMMIT and meta.get("training") is TRAINED[key])
        if "classifier" in EXIT_CODES:
            status, meta = (receipt(CLASSIFIER_RUN, name) for name in ("run_status.json", "run_metadata.json"))
            found["classifier"] = bool(EXIT_CODES["classifier"] == 0 and status.get("complete")
                                       and status.get("run_id") == meta.get("run_id") and meta.get("execution") == "real_dinov2")
        if "gold_controls" in EXIT_CODES:
            status, meta = (receipt(GOLD_RUNS["gold_lora"], name) for name in ("run_status.json", "run_metadata.json"))
            found["gold_controls"] = bool(EXIT_CODES["gold_controls"] == 0 and status.get("complete")
                                          and status.get("run_id") == meta.get("run_id") and meta.get("controls") is True
                                          and len(status.get("eligible_policies", [])) == 4)
        return found

    def preview_images():
        """시험 사진 246장의 축소본(긴 변 640px)을 한 번 만들어 둡니다. 그림을 그릴 때 원본 ZIP을 다시 받지 않아도 됩니다."""
        from PIL import Image
        folder = GOLD_ROOT / "inputs/preview"
        if (folder / "index.csv").is_file():
            return
        folder.mkdir(parents=True, exist_ok=True)
        cores = {}
        for line in GOLD_MASTER.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cores[json.loads(line)["image"]] = json.loads(line)["core_id"]
        rows = []
        with zipfile.ZipFile(SUP_ZIP) as archive:
            names = {n.replace("\\\\", "/").lstrip("/"): n for n in archive.namelist()}
            for member in sorted(cores):
                data = archive.read(names[member])
                name = hashlib.sha256(data).hexdigest()[:16] + ".jpg"
                with Image.open(io.BytesIO(data)) as picture:
                    width, height = picture.size
                    small = picture.convert("RGB")
                    small.thumbnail((640, 640))
                    small.save(folder / name, "JPEG", quality=80)
                rows.append(dict(core_id=cores[member], image=member, file=name, width=width, height=height))
        with (folder / "index.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def save_zip():
        """Drive에 지금 있는 결과로 ZIP을 다시 씁니다. 실패해도 채점은 계속합니다."""
        try:
            try:
                preview_images()
            except Exception as exc:
                print("시험 사진 축소본은 만들지 못했습니다(결과에는 영향 없음):", exc)
            summary = RESULTS / f"agent_sft_{RUN_TAG}_gold_execution.json"
            summary.write_text(json.dumps(dict(
                exit_codes=EXIT_CODES, verified=verified_runs(), execution_commit=EXECUTION_COMMIT, gpu=GPU_NOW,
                written=time.strftime("%Y-%m-%d %H:%M:%S"), gold_refs=str(GOLD_REFS),
                train_qa=[str(p) for p in globals().get("TRAIN_QAS", [])], sft_v1_adapter_found=HAS_V1,
                training_source_commit=receipt(TRAIN_DIR, "train_metadata.json").get("benchmark_commit"),
                run_folders={k: p.name for k, p in {**GOLD_RUNS, "dev_lora": DEV_RUN, "classifier": CLASSIFIER_RUN}.items()},
                earlier_runs={k: p.name for k, p in EARLIER.items()},
                earlier_dev240_run_found=SFT_RUN.is_dir()), indent=2, ensure_ascii=False), encoding="utf-8")
            partial = ARCHIVE.with_name(ARCHIVE.name + ".partial")
            with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as z:
                z.write(summary, "notebook_execution.json")
                z.write(GOLD_REFS, "gold_inputs/retrieved_refs.csv")
                for folder in LABEL_DIRS:                      # the exact label files this run trained on
                    for name in ("manifest.json", "annotations.json"):
                        z.write(folder / name, f"labels/{folder.name}/{name}")
                # SFT_RUN: the four-condition dev240 run of 2026-11-17 that stopped midway; its saved answers are kept.
                # EARLIER: gold folders started by another notebook version, kept as they are.
                for label, root in (("gold_inputs", GOLD_ROOT / "inputs"), ("logs", LOG_DIR), ("data", DATA_DIR),
                                    ("train", TRAIN_DIR), *GOLD_RUNS.items(), ("classifier", CLASSIFIER_RUN),
                                    ("dev_lora", DEV_RUN),
                                    ("dev_lora_earlier_four_condition_run", SFT_RUN), *EARLIER.items()):
                    if not root.is_dir():
                        continue
                    for p in sorted(root.rglob("*")):
                        rel = p.relative_to(root)
                        private = any(part.startswith("_") for part in rel.parts) or rel.parts[0] == "checkpoint"
                        if p.is_file() and not private and p.suffix != ".safetensors":
                            z.write(p, f"{label}/{rel.as_posix()}")
            partial.replace(ARCHIVE)
            print("결과 ZIP 갱신:", ARCHIVE.name, f"{ARCHIVE.stat().st_size / 1e6:.1f} MB")
        except Exception as exc:
            print("결과 ZIP을 갱신하지 못했습니다(채점은 계속하고, 결과 폴더는 Drive에 그대로 있습니다):", exc)
    '''

GOLD_CHECK = '''
    # 학습을 시작하기 전에 시험 채점 입력을 끝까지 점검합니다(모델 호출 없음). 여기서 멈추면 학습 시간을 쓰지 않습니다.
    TRAIN_QAS = [DATA_DIR / "sft_train_qa.jsonl"] + ([V1_TRAIN_QA] if V1_TRAIN_QA.is_file() else [])
    GOLD_COMMON = ["--qa-dir", GOLD_QA_DIR, "--loc-qa", LOC_QA, "--refs-csv", GOLD_REFS, "--sup-zip", SUP_ZIP,
                   "--unsup-zip", UNSUP_ZIP, "--train-qa", *TRAIN_QAS, "--max-calls", 3]
    GOLD_RUNNER = [sys.executable, REPO_DIR / "scripts/run_agent_sft_gold_eval.py", *GOLD_COMMON]
    # A folder name of its own: this check writes nothing and must not be compared with a saved run.
    run([*GOLD_RUNNER, "--out-dir", GOLD_ROOT / "agent_sft_gold_eval_input_check", "--inputs-only"])
    '''

GOLD_MAIN = '''
    #@title 4. 시험(gold) 채점 — 2차 모델 (주 결과)
    verify_code()
    EXIT_CODES["gold_lora"] = gold_run("gold_lora", ["--adapter-dir", TRAIN_DIR / "adapter"])
    print(EXIT_CODES)
    save_zip()
    '''

GOLD_CLASSIFIER = '''
    #@title 5. 작은 분류기 비교 — 같은 라벨 660장, DINOv2 특징 + 선형 분류기
    # LVLM 대신 작은 전용 분류기를 같은 학습 사진으로 학습해 같은 시험 문제의 Q1·Q2를 풉니다(dev240도 같이). 위치는 답하지 못합니다.
    # 뽑은 특징은 결과 ZIP에 들어가므로, 다른 작은 분류기는 나중에 GPU 없이 시험해 볼 수 있습니다.
    verify_code()
    if RUN_CLASSIFIER_BASELINE:
        EXIT_CODES["classifier"] = run([sys.executable, REPO_DIR / "scripts/run_sft_label_classifier_baseline.py",
                                        "--train-qa", DATA_DIR / "sft_train_qa.jsonl", "--overlap-train-qa", *TRAIN_QAS,
                                        "--qa-dir", GOLD_QA_DIR, "--sup-zip", SUP_ZIP, "--dev-qa", DEV_QA,
                                        "--n-images", N_IMAGES, "--device", "cuda", "--out-dir", CLASSIFIER_RUN], check=False)
    else:
        print("작은 분류기 비교는 건너뜁니다.")
    print(EXIT_CODES)
    save_zip()
    '''

GOLD_BASE = '''
    #@title 6. 시험(gold) 채점 — 학습 전 모델
    verify_code()
    if RUN_GOLD_BASE:
        EXIT_CODES["gold_base"] = gold_run("gold_base")
    else:
        print("학습 전 모델 시험 채점은 건너뜁니다.")
    print(EXIT_CODES)
    save_zip()
    '''

GOLD_DEV = '''
    #@title 7. dev240 평가 — 2차 모델 (한 번에 답, 도구 선택)
    verify_code()
    if RUN_DEV240:
        EXIT_CODES["dev_lora"] = run([sys.executable, REPO_DIR / "scripts/run_agent_tool_pilot.py", "--dev-qa", DEV_QA,
                                      "--refs-csv", REFS_CSV, "--sup-zip", SUP_ZIP, "--unsup-zip", UNSUP_ZIP,
                                      "--model", "qwen3_5_4b", "--n-images", N_IMAGES, "--max-calls", 3,
                                      "--policies", "direct_reference", "adaptive_tools",
                                      "--out-dir", DEV_RUN, "--adapter-dir", TRAIN_DIR / "adapter"], check=False)
    else:
        print("dev240 평가는 건너뜁니다.")
    print(EXIT_CODES)
    save_zip()
    '''

GOLD_V1 = '''
    #@title 8. 시험(gold) 채점 — 1차 모델 (Drive에 있을 때만)
    verify_code()
    if RUN_GOLD_SFT_V1 and HAS_V1:
        EXIT_CODES["gold_v1"] = gold_run("gold_v1", ["--adapter-dir", V1_ADAPTER])
    else:
        print("1차 모델 시험 채점은 건너뜁니다.")
    print(EXIT_CODES)
    save_zip()
    '''

GOLD_CONTROLS = '''
    #@title 9. 시험(gold) 채점 — 2차 모델의 대조 조건 (가운데 고정 확대, 같은 횟수 다시 묻기)
    # 4번에서 저장한 답은 그대로 다시 쓰고, 대조 조건 두 개만 새로 묻습니다. 4번의 표는 폴더 안에 따로 보존됩니다.
    verify_code()
    if RUN_GOLD_CONTROLS and EXIT_CODES.get("gold_lora") == 0:
        EXIT_CODES["gold_controls"] = gold_run("gold_lora", ["--adapter-dir", TRAIN_DIR / "adapter", "--controls"])
    else:
        print("대조 조건은 건너뜁니다(스위치를 껐거나 4번 채점이 끝나지 않았습니다).")
    print(EXIT_CODES)
    save_zip()
    '''

GOLD_REPORT = '''
    #@title 10. 표와 결과 ZIP
    save_zip()                                           # the ZIP comes first and never depends on the tables below
    VERIFIED = verified_runs()
    for key in ("gold_lora", "classifier", "gold_base", "dev_lora", "gold_v1", "gold_controls"):
        if key not in EXIT_CODES:
            print(key, "건너뜀")
            continue
        root = {**GOLD_RUNS, "dev_lora": DEV_RUN, "classifier": CLASSIFIER_RUN, "gold_controls": GOLD_RUNS["gold_lora"]}[key]
        print(key, "완료 확인" if VERIFIED.get(key) else "실행 오류 또는 완료 미확인",
              json.dumps(receipt(root, "run_status.json"), ensure_ascii=False))
    try:
        import pandas as pd
        def boards(pairs):
            found = []
            for label, root in pairs:
                board = root / "leaderboard.csv"
                if board.is_file() and board.stat().st_size:
                    found.append(pd.read_csv(board).assign(model=label))
            return pd.concat(found) if found else None
        NAMES = {"gold_lora": f"lora {RUN_TAG} (사람 박스)", "gold_base": "학습 전", "gold_v1": "lora sft_v1 (자동 박스)"}
        table = boards((NAMES[key], root) for key, root in GOLD_RUNS.items())
        if table is not None:
            cols = ["model", "subset", "policy", "images", "q1_anomaly_f1", "q1_balanced_accuracy", "q2_defect_macro_f1",
                    "joint_accuracy", "q3_presence_balanced_accuracy", "q3_position_hit", "q3_anomalous_location_labels",
                    "q3_always_center_hit", "q1_q2_q3_location_joint", "q3_crop_covers_defect", "q3_crops_on_located",
                    "center60_covers_defect", "crop_distinct_boxes", "invalid_episodes", "calls"]
            print("시험(gold) 결과. 주 결과는 subset=frame_disjoint(학습 사진과 같은 드론 사진을 뺀 사진), 2차 모델, direct_reference 행입니다.")
            print("fixed_evidence와 matched_reask는 대조 조건입니다(2차 모델만).")
            display(table[[c for c in cols if c in table]])
        table = boards([("DINOv2 특징 + 선형 분류기", CLASSIFIER_RUN)])
        if table is not None:
            cols = ["model", "eval_set", "subset", "images", "q1_anomaly_f1", "q1_balanced_accuracy", "q2_defect_macro_f1",
                    "joint_accuracy", "false_defect_rate", "q1_tp", "q1_fp", "q1_tn", "q1_fn"]
            print("작은 분류기 결과(같은 학습 사진 660장, Q1·Q2만). gold의 subset=frame_disjoint 행을 위 표의 같은 subset과 비교합니다.")
            display(table[[c for c in cols if c in table]])
        table = boards(((NAMES["gold_lora"], DEV_RUN), *PREVIOUS.items()))
        if table is not None:
            cols = ["model", "policy", "q1_balanced_accuracy", "q2_defect_macro_f1", "q2_invalid_rate", "joint_accuracy",
                    "false_defect_rate", "q3_presence_balanced_accuracy", "q3_normal_false_region_rate", "invalid_episodes", "calls"]
            print("dev240 결과. 지난 실행의 행은 Drive에 남은 결과를 읽은 것입니다(이번에 다시 돌리지 않음).")
            display(table[[c for c in cols if c in table]])
    except Exception as exc:
        print("표를 그리지 못했습니다(결과 ZIP에는 영향 없음):", exc)
    train_meta = receipt(TRAIN_DIR, "train_metadata.json")
    print("학습:", {k: train_meta.get(k) for k in ("optimizer_steps", "val_loss_before", "val_loss_after")})
    if train_meta.get("benchmark_commit") != EXECUTION_COMMIT:
        print("학습은 이전 실행(소스", str(train_meta.get("benchmark_commit"))[:7] + ")이 Drive에 남긴 어댑터를 그대로 썼습니다.")
    print("결과 ZIP은 Drive에 저장됐습니다:", ARCHIVE)
    print("모두 완료" if VERIFIED and all(VERIFIED.values()) else "일부 미완료 또는 건너뜀(위 목록 확인)")
    try:
        from google.colab import files
        local = Path("/content") / ARCHIVE.name             # a local copy, so the download does not depend on Drive
        shutil.copyfile(ARCHIVE, local)
        files.download(str(local))
    except Exception as exc:
        print("브라우저 다운로드는 건너뜁니다. Drive에 있는 ZIP을 쓰세요:", exc)
    try:
        drive.flush_and_unmount()                           # push everything to Drive before the runtime can go away
        print("Drive 저장을 마쳤습니다. 이제 런타임을 끊어도 됩니다.")
    except Exception as exc:
        print("Drive 마무리 저장을 확인하지 못했습니다. 몇 분 뒤에 런타임을 끊으세요:", exc)
    '''


def build(execution_commit, variant="sft_v1"):
    if not re.fullmatch(r"[0-9a-f]{40}", execution_commit):
        raise ValueError("execution commit must be a full immutable SHA")
    if variant not in NOTEBOOKS:
        raise ValueError(f"unknown variant {variant}")
    gold = variant == GOLD
    human = gold or variant == "sft_v2_human"
    run_tag = "sft_v2_human" if gold else variant       # the gold notebook is the same training run, scored further
    bundle = pilot_nb.source_bundle(execution_commit)
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        names = set(json.loads(archive.read(".agent_pilot_source.json"))["files"])
    for needed in ("scripts/build_agent_sft_data.py", "scripts/train_agent_sft_lora.py"):
        if needed not in names:
            raise ValueError(f"source commit lacks {needed}")
    if human and b"--human-boxes" not in subprocess.check_output(
            ["git", "show", f"{execution_commit}:scripts/build_agent_sft_data.py"], cwd=ROOT):
        raise ValueError("source commit lacks the human-box data builder")
    labels = f"LABEL_RUNS = {json.dumps(LABEL_RUNS)}\n        FIRST_LABELS_SHA256 = \"{FIRST_LABELS_SHA256}\"" if human else ""
    if gold:
        if "scripts/run_agent_sft_gold_eval.py" not in names or b'"--policies"' not in subprocess.check_output(
                ["git", "show", f"{execution_commit}:scripts/run_agent_tool_pilot.py"], cwd=ROOT):
            raise ValueError("source commit lacks the gold runner or the pilot runner's --policies option")
        if b'"--controls"' not in subprocess.check_output(
                ["git", "show", f"{execution_commit}:scripts/run_agent_sft_gold_eval.py"], cwd=ROOT):
            raise ValueError("source commit lacks the gold runner's --controls step")
        if "scripts/run_sft_label_classifier_baseline.py" not in names:
            raise ValueError("source commit lacks the gold runner's companion, the small-classifier baseline")
        loc = subprocess.check_output(["git", "show", f"{execution_commit}:{LOC_QA}"], cwd=ROOT)
        gap = "\n        "                              # the settings cell is indented eight spaces before dedent
        labels += gap + gap.join((
            'GOLD_QA_FOLDER = "lvlm_insplad_qa/gold_v2_final"',
            f'LOC_QA_SHA256 = "{hashlib.sha256(loc).hexdigest()}"',
            f'LOC_QA_B64 = "{base64.b64encode(zlib.compress(loc, 9)).decode("ascii")}"',
            "RUN_CLASSIFIER_BASELINE = True  # 같은 라벨로 학습한 작은 분류기(DINOv2 특징 + 선형 분류기) 비교",
            "RUN_GOLD_BASE = True      # 학습 전 모델의 시험 채점",
            "RUN_DEV240 = True         # 2차 모델의 dev240 평가",
            "RUN_GOLD_SFT_V1 = True    # Drive에 1차 어댑터가 있으면 1차 모델의 시험 채점",
            "RUN_GOLD_CONTROLS = True  # 2차 모델 시험 채점의 대조 조건(가운데 고정 확대, 같은 횟수 다시 묻기)"))
    encoded = base64.b64encode(bundle).decode("ascii")
    cells = [cell("markdown", """
        # 도구 선택 에이전트 LoRA SFT — 한 번에 실행
        **A100 또는 L4(20 GiB 이상) GPU**에서 **모두 실행**하세요. 4bit 양자화는 쓰지 않습니다(BF16).

        순서: ① 학습 데이터 만들기 → ② Qwen3.5-4B LoRA 학습 → ③ 학습 전 Qwen3.5-4B로 dev240 평가 → ④ LoRA 모델로 dev240 평가 → ⑤ 비교표와 결과 ZIP 다운로드.

        - 학습 데이터: supervised train에서 dev240 사진, 같은 드론 원본 사진, test와 같은 사진을 모두 뺀 뒤 클래스당 60장을 씁니다. gold는 쓰지 않습니다.
        - 확대 박스: DINOv2로 정상 참조와 비교해 **자동으로** 만듭니다. 사람이 그린 박스가 아니라서 틀릴 수 있습니다.
        - 학습 목표: "정상 사진 요청 → (차이가 크면) 확대 → 답"의 JSON 행동입니다. 근거 문장은 넣지 않습니다(평가 파서가 JSON 밖 문장을 무효로 처리하기 때문).
        - 평가는 지난번과 같은 dev240·4개 조건·채점입니다. dev에는 위치 정답이 없어 Q3는 유무만 채점합니다.

        예상 시간은 측정 전입니다. 학습은 1~2시간, 평가는 두 번에 2시간 안팎으로 예상합니다. 끊기면 같은 노트북을 다시 **모두 실행**하세요. 학습 checkpoint와 평가 호출을 이어서 씁니다.
        """), cell("code", f'''
        #@title 1. 실행 설정 (수정 불필요)
        EXECUTION_COMMIT = "{execution_commit}"
        SOURCE_ZIP_SHA256 = "{hashlib.sha256(bundle).hexdigest()}"
        SOURCE_ZIP_B64 = "{encoded}"
        N_IMAGES = 240
        PER_CLASS = 60
        EPOCHS = 1
        LEARNING_RATE = 1e-4
        RUN_TAG = "{run_tag}"
        DEV_FOLDER = "lvlm_insplad_qa/gold_v2_dev"
        SUP_ZIP_PATH = ""  # 자동 탐색이 모호할 때만 Drive의 절대 경로 입력
        UNSUP_ZIP_PATH = ""
        {labels}
        '''), cell("code", inspect.getsource(pilot_nb.extract_source_bundle) + '\n' + textwrap.dedent('''
        import csv, json, os, re, subprocess, sys, zipfile
        from pathlib import Path
        from google.colab import drive

        def run(command, cwd=None, check=True):
            command = [str(x) for x in command]
            print("+", " ".join(command[:10]), flush=True)
            keep_log("+ " + " ".join(command) + "\\n")
            process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                keep_log(line)
            result = process.wait()
            keep_log(f"[exit {result}]\\n")
            if result and check:
                raise RuntimeError("명령 실패: 위 오류를 확인하세요. 결과를 성공으로 처리하지 않습니다.")
            return result

        def keep_log(text):
            pass                     # the one-run gold notebook replaces this with a log file on Drive

        if N_IMAGES != 240 or not re.fullmatch(r"[A-Za-z0-9_-]+", RUN_TAG):
            raise ValueError("240장 평가 설정을 유지하세요.")
        ''') + textwrap.dedent(GPU_GATE) + textwrap.dedent('''
        drive.mount("/content/drive", force_remount=False)
        DRIVE = Path("/content/drive/MyDrive")
        REPO_DIR = Path("/content/insplad_agent_sft_once")
        def verify_code():
            extract_source_bundle(SOURCE_ZIP_B64, SOURCE_ZIP_SHA256, REPO_DIR, EXECUTION_COMMIT)
        verify_code()
        run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.14.0", "accelerate==1.14.0", "peft>=0.18",
             "safetensors", "sentencepiece==0.2.2", "numpy>=1.26", "pillow>=10", "pandas>=2.2", "tqdm>=4.66", "timm"])
        # Colab ships torchao 0.10; peft refuses to build LoRA layers when an old torchao is importable.
        # This run never uses torchao (BF16, no quantization), so remove it instead of upgrading torch-coupled code.
        run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
        if subprocess.run([sys.executable, "-c", "import torchao"], capture_output=True).returncode == 0:
            raise RuntimeError("torchao를 제거하지 못했습니다. 런타임을 다시 시작한 뒤 모두 실행하세요.")
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3:
            raise RuntimeError(f"torch가 GPU를 쓰지 못합니다(nvidia-smi: {GPU_NOW}, torch {torch.__version__}, "
                               f"CUDA 사용 가능: {torch.cuda.is_available()}). VRAM 20 GiB 이상 L4/A100이 필요하고 "
                               "양자화로 우회하지 않습니다. 런타임을 다시 시작한 뒤 모두 실행하세요.")
        print("GPU:", torch.cuda.get_device_name(0), "source:", EXECUTION_COMMIT)
        DEV_DIR = DRIVE / DEV_FOLDER
        summary = json.loads((DEV_DIR / "build_summary.json").read_text(encoding="utf-8"))
        if summary.get("quality_status") != "FROZEN":
            raise RuntimeError("기존 dev 세트가 FROZEN이 아닙니다.")
        DEV_QA = (DEV_DIR / summary["file"]).resolve()
        if not DEV_QA.is_relative_to(DEV_DIR.resolve()) or not DEV_QA.is_file():
            raise ValueError("dev QA 경로를 확인하세요.")

        def find_zip(name, size, override):
            if override:
                p = Path(override)
                if not p.is_file() or p.stat().st_size != size:
                    raise ValueError(f"ZIP 경로/크기 불일치: {p}")
                return p
            for p in [DRIVE / name, DRIVE / "lvlm_insplad_data" / name]:
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
        RESULTS = DRIVE / "lvlm_insplad_results"
        DATA_DIR = RESULTS / f"agent_sft_data_pc{PER_CLASS}_{RUN_TAG}"
        TRAIN_DIR = RESULTS / f"agent_sft_lora_qwen3_5_4b_{RUN_TAG}"
        BASE_RUN = RESULTS / f"agent_tool_pilot_dev{N_IMAGES}_qwen3_5_4b_base_{RUN_TAG}"
        SFT_RUN = RESULTS / f"agent_tool_pilot_dev{N_IMAGES}_qwen3_5_4b_lora_{RUN_TAG}"
        print("학습 데이터:", DATA_DIR, "\\n어댑터:", TRAIN_DIR, "\\n평가:", BASE_RUN, SFT_RUN)
        ''') + (textwrap.dedent(HUMAN_CHECK) if human else "")
            + (textwrap.dedent(GOLD_PREP) + textwrap.dedent(GOLD_TOOLS) if gold else "")), cell("code", '''
        #@title 2. 학습 데이터 만들기 (자동 박스·궤적)
        verify_code()
        run([sys.executable, REPO_DIR / "scripts/build_agent_sft_data.py", "--dev-qa", DEV_QA,
             "--sup-zip", SUP_ZIP, "--unsup-zip", UNSUP_ZIP, "--out-dir", DATA_DIR,
             "--per-class", PER_CLASS, "--device", "cuda"])
        print(json.dumps(json.loads((DATA_DIR / "build_summary.json").read_text(encoding="utf-8"))["trajectories"]))
        '''), cell("code", '''
        #@title 3. Qwen3.5-4B LoRA 학습 (BF16)
        verify_code()
        run([sys.executable, REPO_DIR / "scripts/train_agent_sft_lora.py", "--data-dir", DATA_DIR,
             "--out-dir", TRAIN_DIR, "--epochs", EPOCHS, "--lr", LEARNING_RATE])
        '''), cell("code", '''
        #@title 4. 학습 전 / 학습 후 dev240 평가 (지난번과 같은 4개 조건)
        verify_code()
        common = ["--dev-qa", DEV_QA, "--refs-csv", REFS_CSV, "--sup-zip", SUP_ZIP, "--unsup-zip", UNSUP_ZIP,
                  "--model", "qwen3_5_4b", "--n-images", N_IMAGES, "--max-calls", 3]
        EXIT_CODES = {}
        EXIT_CODES["base"] = run([sys.executable, REPO_DIR / "scripts/run_agent_tool_pilot.py", *common,
                                  "--out-dir", BASE_RUN], check=False)
        EXIT_CODES["lora"] = run([sys.executable, REPO_DIR / "scripts/run_agent_tool_pilot.py", *common,
                                  "--out-dir", SFT_RUN, "--adapter-dir", TRAIN_DIR / "adapter"], check=False)
        print(EXIT_CODES)
        '''), cell("code", '''
        #@title 5. 비교표와 결과 ZIP
        import pandas as pd
        from google.colab import files
        def receipt(root, name):
            p = root / name
            return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
        verdicts, boards = {}, []
        for label, root in (("base", BASE_RUN), ("lora", SFT_RUN)):
            status, meta = receipt(root, "run_status.json"), receipt(root, "run_metadata.json")
            verdicts[label] = bool(EXIT_CODES.get(label) == 0 and status.get("complete") and meta.get("execution") == "real_hf"
                                   and status.get("run_id") == meta.get("run_id")
                                   and meta.get("benchmark_commit") == EXECUTION_COMMIT
                                   and bool(meta.get("training")) == (label == "lora"))
            board = root / "leaderboard.csv"
            if board.is_file() and board.stat().st_size:
                boards.append(pd.read_csv(board).assign(model=label))
            print(label, "완료 확인" if verdicts[label] else "실행 오류 또는 완료 미확인", json.dumps(status, ensure_ascii=False))
        train_meta = receipt(TRAIN_DIR, "train_metadata.json")
        base_rev = receipt(BASE_RUN, "run_metadata.json").get("resolved_model_revision")
        if train_meta and base_rev and train_meta.get("resolved_model_revision") != base_rev:
            print("경고: 학습과 평가의 모델 revision이 다릅니다.")
        if boards:
            table = pd.concat(boards)
            cols = ["model", "policy", "q1_balanced_accuracy", "q2_defect_macro_f1", "q2_invalid_rate", "joint_accuracy",
                    "false_defect_rate", "q3_presence_balanced_accuracy", "q3_normal_false_region_rate", "invalid_episodes", "calls"]
            display(table[[c for c in cols if c in table]])
        print("학습:", {k: train_meta.get(k) for k in ("optimizer_steps", "val_loss_before", "val_loss_after")})
        print("Q3 위치 정확도는 미평가입니다(dev에 위치 정답 없음). dev240은 여러 번 본 개발 세트입니다.")
        (RESULTS / f"agent_sft_{RUN_TAG}_execution.json").write_text(json.dumps(dict(
            exit_codes=EXIT_CODES, verified=verdicts, execution_commit=EXECUTION_COMMIT), indent=2), encoding="utf-8")
        archive = RESULTS / f"agent_sft_{RUN_TAG}_results.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(RESULTS / f"agent_sft_{RUN_TAG}_execution.json", "notebook_execution.json")
            for label, root in (("data", DATA_DIR), ("train", TRAIN_DIR), ("base", BASE_RUN), ("lora", SFT_RUN)):
                if not root.is_dir():
                    continue
                for p in sorted(root.rglob("*")):
                    rel = p.relative_to(root)
                    private = any(part.startswith("_") for part in rel.parts) or rel.parts[0] == "checkpoint"
                    if p.is_file() and not private and p.suffix != ".safetensors":
                        z.write(p, f"{label}/{rel.as_posix()}")
        files.download(str(archive))
        print("비교 완료" if all(verdicts.values()) else "일부 미완료(진단용)", "ZIP:", archive)
        ''')]
    if human:
        tail = (GOLD_MAIN, GOLD_CLASSIFIER, GOLD_BASE, GOLD_DEV, GOLD_V1, GOLD_CONTROLS, GOLD_REPORT) if gold else (HUMAN_EVAL, HUMAN_REPORT)
        cells = [cell("markdown", GOLD_INTRO if gold else HUMAN_INTRO), cells[1], cells[2],
                 cell("code", textwrap.dedent(HUMAN_DATA) + (textwrap.dedent(GOLD_CHECK) if gold else "")), cells[4],
                 *(cell("code", source) for source in tail)]
    cells[1]["metadata"]["cellView"] = "form"
    return dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", name="python3"),
                                          language_info=dict(name="python"), accelerator="GPU",
                                          colab=dict(provenance=[], gpuType="L4")),
                nbformat=4, nbformat_minor=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--variant", choices=sorted(NOTEBOOKS), default="sft_v1")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    args.output = args.output or ROOT / NOTEBOOKS[args.variant]
    notebook = build(args.execution_commit, args.variant)
    args.output.write_bytes((json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    print(f"Built {args.output}; {sum(c['cell_type'] == 'code' for c in notebook['cells'])} code cells compiled")


if __name__ == "__main__":
    main()

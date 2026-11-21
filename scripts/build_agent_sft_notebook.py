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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_agent_tool_pilot_notebook as pilot_nb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
cell = pilot_nb.cell
NOTEBOOKS = {"sft_v1": "notebooks/lvlm_agent_sft_lora_colab.ipynb",
             "sft_v2_human": "notebooks/lvlm_agent_sft_lora_human_boxes_colab.ipynb"}
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


def build(execution_commit, variant="sft_v1"):
    if not re.fullmatch(r"[0-9a-f]{40}", execution_commit):
        raise ValueError("execution commit must be a full immutable SHA")
    if variant not in NOTEBOOKS:
        raise ValueError(f"unknown variant {variant}")
    human = variant == "sft_v2_human"
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
        RUN_TAG = "{variant}"
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
            process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
            result = process.wait()
            if result and check:
                raise RuntimeError("명령 실패: 위 오류를 확인하세요. 결과를 성공으로 처리하지 않습니다.")
            return result

        if N_IMAGES != 240 or not re.fullmatch(r"[A-Za-z0-9_-]+", RUN_TAG):
            raise ValueError("240장 평가 설정을 유지하세요.")
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
            raise RuntimeError("VRAM 20 GiB 이상 L4/A100 GPU를 선택하세요. 양자화로 우회하지 않습니다.")
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
        ''') + (textwrap.dedent(HUMAN_CHECK) if human else "")), cell("code", '''
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
        cells = [cell("markdown", HUMAN_INTRO), cells[1], cells[2], cell("code", HUMAN_DATA), cells[4],
                 cell("code", HUMAN_EVAL), cell("code", HUMAN_REPORT)]
    cells[1]["metadata"]["cellView"] = "form"
    return dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", name="python3"),
                                          language_info=dict(name="python"), accelerator="GPU", colab=dict(provenance=[])),
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

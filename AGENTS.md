# lvlm_insplad — 에이전트 운영 규칙

InsPLAD 데이터로 LVLM 이상탐지 · 구조화 결함 이해 QA 벤치마크(MMAD 스타일)를 만들고, 여러 VLM 을 같은 QA · 프롬프트 · 채점 계약으로 비교한다. 설계·트랙 정의는 `README.md`, 목표·범위는 `GOAL.md`, 진행 중 작업은 `plan.md`, 실험별 프로토콜은 `docs/`, 데이터 검증 기록은 `reports/*.md`.

## 변경 금지 계약
- Track A: unsupervised 17개 자산, 공식 `test` 6,481장. Track B: supervised 5개 자산, 공식 `val` 6,417장(test split 없음). Joint Accuracy 는 Track B 안에서만 계산
- Aligned Core = unsup test ∩ sup val ∩ SHA256 일치 ∩ binary GT 일치 = **1,378장**. binary GT 가 충돌하는 6장은 트랙 간 비교에 쓰지 않는다
- 라벨: `good`/`normal` → `normal`, 결함 폴더 → `anomalous`. `corrosão`→`corrosao` 는 slug 변환만 하고 `rust` 와 합치지 않는다. `image_list_val.txt` 정수 ID 는 자산별 클래스 인덱스이지 binary 라벨이 아니다
- 1-shot 참조 이미지는 같은 자산의 unsupervised train normal 에서만(`seed=42`). size+CRC32 지문이 test 에 나오는 후보는 제외(`leakage_safe_train_normal_candidates`)
- zero-shot 과 one-shot 은 같은 ID · 이미지 · GT 를 쓴다. 1-shot+(DINOv2 cosine Top-1)는 기존 고정 baseline 산출물을 덮어쓰지 않고 별도 경로에 쓴다
- 무효 출력: GT anomalous 행은 FN, GT normal 행은 FP 와 따로 집계(`metrics.anomaly_classification_summary`). 이 정책은 바꾸지 않는다
- 모든 모델에 같은 QA · 프롬프트 · 출력 스키마 · 평가기. HF 오픈소스 모델을 몰래 양자화하지 않는다. VRAM 20 GB 미만이면 HF 러너가 실행을 거부한다
- smoke/dev 코호트는 방법 선택용이다. 최종 holdout 은 프로토콜 고정 전에는 쓰지 않는다(실험별 holdout 크기는 해당 `docs/` 참고)
- gold v2 세트(246장 / 534문항)는 `FROZEN`. 수정하지 않는다
- 원본 ZIP 과 이미지는 커밋하지 않는다(`.gitignore`). `insplad_google_drive` 는 공식 Mendeley 다운로드 링크 텍스트 파일이다. 개인 Drive 링크를 다시 넣지 않는다
- 비용이 큰 full LVLM 추론(1,378행 이상)은 사용자가 명시적으로 요청할 때만 돌린다

## 보고 규칙
- anomaly: Balanced Accuracy, F1, Precision/Recall, TP/FP/TN/FN, Invalid Output Rate + 자산별 · asset-macro 지표. normal-only 자산(`damper-preformed`, `spacer`, `yoke`)은 class support 를 명시
- structured: defect BAcc, Macro F1, Joint Accuracy, FDR, Consistency Error, Invalid Output Rate
- 실험마다 `run_metadata.json`(모델 ID, backend, 벤치마크 커밋, GPU/VRAM, protocol, 사후 규칙)과 `run_status.json` 을 남긴다. 결과는 `results/latest/<실험명>/` 에 `leaderboard.csv`, `metrics_by_asset.csv`, `github_publish_manifest.json` 형식으로 둔다
- 실제로 돌린 검증과 돌리지 않은 명령을 나눠 보고한다
- `scripts/publish_results_to_github.py` 는 GitHub API 로 `main` 에 바로 커밋한다. 커밋 · push · publish 는 사용자가 요청할 때만

## 빌드·테스트 명령
```powershell
python -m pip install -e ".[dev]"            # retrieval 은 .[retrieval,hf]
python -m pytest -q --basetemp=.pytest-tmp    # fixture 기반이라 원본 ZIP·GPU 없이 돈다. Windows 는 --basetemp 없으면 임시폴더 권한 오류
python scripts/evaluate.py --task anomaly --qa <qa.jsonl> --predictions <pred.jsonl> --regime zero_shot|one_shot|one_shot_plus
```
LVLM 추론은 Colab `notebooks/*_colab.ipynb` 에서 VRAM 20 GB 이상 GPU(L4/A100급)로 돌린다. 로컬 GPU(6 GB)로는 HF 추론을 하지 않는다. 린터 설정은 없다.

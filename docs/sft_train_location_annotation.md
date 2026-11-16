# 학습 사진 결함 위치 표시 (300장 + polymer 녹 60장)

파일: **`lvlm_sft_train_location_annotation_colab.ipynb`**.
[Colab 열기](https://colab.research.google.com/github/arnold6444/lvlm_insplad/blob/main/notebooks/lvlm_sft_train_location_annotation_colab.ipynb)

**GPU 불필요. 위에서부터 실행 → 표시 화면에서 박스 표시 → 마지막 결과 저장 셀 실행.**

## 왜 하는가

LoRA SFT(`sft_v1`)는 확대할 위치를 자동 DINOv2 박스로 배웠다. 결과(2026-11-16, dev240)에서 확대 365번 중 341번이 같은 왼쪽 위 구석이었다. 자동 박스로는 위치가 학습되지 않으므로, 학습에 쓴 결함 사진에 사람이 박스를 표시한다. 자세한 결과는 [agent_sft_lora.md](agent_sft_lora.md).

## 대상

- `sft_v1` 학습 데이터의 결함 사진 300장: 누락 60, 녹 180(자산 3종 × 60), 새 둥지 60. 정상 사진 300장은 표시하지 않는다.
- 목록은 학습 데이터(`sft_samples.jsonl` SHA256 `a12ae433…`)와 원본 supervised ZIP으로 고정했다. 300장 모두 원본 ZIP의 크기·CRC32·SHA256과 대조했다.
- dev240과 gold 사진은 들어 있지 않다. 도구가 `sft-` ID와 supervised `train` 경로만 받는다. dev·gold 표시와 기존 결과는 바꾸지 않는다.

## 표시 기준

- 화면에는 사진과 **결함 종류**만 나온다. 파일명, 모델 답, 자동 박스는 나오지 않는다.
- 결함을 감싸는 박스를 그린다. 결함이 여러 곳이면 각각 그린다. gold 박스와 같은 기준이다.
  - 누락: 원판이 빠진 자리(연결부·드러난 금속 몸통). 절연체 줄 전체를 감싸지 않는다.
  - 녹: 녹슨 부분. 부품 전체가 녹슬었으면 그 부품을 감싸는 박스 하나.
  - 새 둥지: 둥지(나뭇가지·풀 뭉치).
- 위치를 정하기 어려우면 `위치 판단 불가` + 사유, 결함 라벨이 이상하면 `라벨 이견` + 사유. 두 경우 모두 위치 정답에서 빠진다.
- 순서는 같은 종류 12장씩 묶음이다. 60장을 끝낼 때마다 다섯 종류가 12장씩 채워진다.

## 저장·재개

박스를 그리거나 지우면 Drive에 초안이 자동 저장된다. `저장 후 다음(Enter)`으로 확정한다. `초안 저장됨`을 확인하고 나가면, 다음 실행 때 첫 미완료 사진부터 이어진다. 저장 규칙은 [dev_missing_location_annotation.md](dev_missing_location_annotation.md)와 같다.

저장 폴더: `MyDrive/lvlm_insplad_results/sft_train_location_annotation_sft_v1/`. 사진 사본은 Colab 런타임(`/content`)에만 두고 Drive에는 쓰지 않는다.

| 파일 | 내용 |
|---|---|
| annotations.json | 사진 SHA·크기, 정규화 박스, 상태·메모·버전·저장 시각 |
| manifest.json | 고정된 300장 목록 |
| history/ | 수정하기 전의 이전 버전 |
| sft_train_location_gt.json | 확정 박스와 박스 중심의 9칸 정답. 나머지는 null |
| annotation_summary.json | 확정·판단 불가·이견 수, 종류별 박스 수, 9칸 분포 |

마지막 셀이 `<저장 폴더 이름>_results.zip`을 만든다(사진 제외). 중간 백업도 된다.

## 300장 표시 결과 (2026-11-16)

300장 모두 박스 확정, 판단 불가·이견 0, 박스 348개(여러 박스 38장). 대상 목록과 사진 SHA가 학습 데이터와 일치한다.

| 종류 | 가장 흔한 칸 | 박스 면적 중앙값 |
|---|---|---|
| glass-insulator 누락 | 가운데 28/60, 나머지는 퍼짐 | 1% |
| lightning-rod-suspension 녹 | 위-가운데 37/60 | 9% |
| yoke-suspension 녹 | 위-가운데 42/60 | 19% |
| vari-grip 녹 | 가운데 48/60 | 30% |
| vari-grip 새 둥지 | 가운데 48/60 | 73% |

- 기준선: 가운데를 포함한 사진 159/300(53%), 종류별 최빈 칸 203/300(68%). gold 133장에서는 "항상 가운데" 72/133(54%), 학습 라벨의 종류별 최빈 칸 67/133(50%).
- 새 둥지는 박스가 사진 대부분이라 확대의 의미가 작다.

## polymer 녹 60장 추가 (2026-11-16 사용자 결정)

- gold 결함 133장 중 24장이 polymer-insulator-upper-shackle의 녹(`val/rust`, 위-가운데 23/24)이다. `sft_v1` 학습에는 polymer 결함이 없었다. supervised train의 폴더명이 `corrosão`라서 dev240과 `sft_v1`이 이 종류를 쓰지 않았기 때문이다. 양쪽 사진을 직접 보면 같은 고리의 같은 부식이다.
- 다음 SFT 세트는 train `corrosão` 60장을 넣고, Q2 정답은 gold와 같은 "Rust / corrosion."으로 둔다. 평가용 라벨과 gold는 바꾸지 않는다. 학습 쪽에서만 두 이름을 같은 답으로 쓴다.
- 60장은 SFT 선정 코드(`select_train_images` + `POLYMER_CORROSION`)가 같은 제외 규칙으로 뽑는다. 기존 600장 선정은 그대로다(테스트와 실제 데이터로 확인).
- 표시 노트북: **`lvlm_sft_train_location_annotation_polymer_colab.ipynb`**. [Colab 열기](https://colab.research.google.com/github/arnold6444/lvlm_insplad/blob/main/notebooks/lvlm_sft_train_location_annotation_polymer_colab.ipynb)
- 저장 폴더는 `MyDrive/lvlm_insplad_results/sft_train_location_annotation_sft_v2_polymer/`로 300장 폴더와 분리된다. 결과는 `sft_train_location_annotation_sft_v2_polymer_results.zip`.
- 대상 목록: `reports/sft_polymer_location_annotation_preparation_2026-11-16/manifest.json`.

## 표시가 끝난 뒤

1. `annotation_summary.json`의 9칸 분포를 먼저 본다. gold 위치 정답은 "항상 가운데"가 72/133을 맞힌다. 학습용 정답도 가운데에만 몰려 있으면 위치 학습은 의미가 없다.
2. polymer 60장 표시까지 끝나면, 자동 박스 대신 사람 박스 360장으로 SFT 데이터를 다시 만들어 학습한다(별도 작업, 아직 없음).

## 검증

- `tests/test_sft_train_location_annotation.py`: 초안·확정·재개·버전 충돌·이력, 판단 불가/이견, 잘못된 박스, dev·gold·모델 출력이 섞인 목록 거부, 사진 손상 복구, 노트북 내장 코드와 현재 코드 일치, 300장이 60장 단위로 균형인지.
- 로컬 브라우저 확인(2026-11-16): Colab 콜백을 흉내 낸 로컬 서버 + 실제 학습 사진 300장. 박스 그리기, 초안 저장, 확정 후 다음, 사유 없는 판단 불가 거부, 이전 사진 복원, 결과 내보내기를 확인했다. 실제 Google Colab 연결은 확인하지 않았다.

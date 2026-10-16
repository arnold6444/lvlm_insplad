# 부품 누락 판별기 단독 학습·검증

`notebooks/missing_component_probe_colab.ipynb`를 L4/A100에서 모두 실행한다.
기존 Colab Secret `GITHUB_TOKEN` 허용과 Drive 인증 외 수동 라벨·박스·파일 경로 입력은 없다.
기존 Drive의 원본 supervised/unsupervised ZIP과 `lvlm_insplad_qa/gold_v2_dev`를 자동 탐색한다.

기존 train의 glass-insulator `good`/`missing-cap` 라벨을 사용하는 **지도학습 보조 조건**이다.
학습 없는 LVLM·정상 참고 사진 조건과 같은 방법으로 취급하지 않는다. 이번에는 독립 이진 판별기만 검증한다.
Q1/Q2/Q3 답이나 기존 결과를 바꾸지 않으며 gold 추론·agent 결합·위치 검증은 하지 않는다.

## 고정한 방법

- DINOv2-base encoder는 고정한다. `facebook/dinov2-base` revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`.
- 전체 사진을 RGB 변환·정사각 패딩·518×518 리사이즈한다. 중앙 crop으로 끝 부품을 잘라내지 않는다.
- L2 정규화한 CLS 특징 위에 StandardScaler + balanced LogisticRegression을 학습한다.
  C=1, lbfgs, max_iter=2000, seed42. 정규화 통계와 가중치는 **fit만** 사용한다. 수렴 실패는 오류다.
- 임계값은 **calibration만** 사용해 정상 오탐률 5% 이하에서 균형 정확도가 최대인 값을 고른다.
  calibration 점수 사이 중간값을 후보로 쓰며 동률이면 높은 임계값을 택한다. 모두 정상 판정도 허용한다.
  이 5% 조건은 calibration의 조건이며 새 사진의 오탐률을 보장하지 않는다.
- 모델·임계값을 `frozen_detector.json`에 저장한 뒤 내부 평가와 dev 사진을 예측한다.
  dev는 이미 여러 번 분석한 탐색 데이터이며 추가 튜닝에 사용하지 않는다. gold는 평가하지 않는다.

[공식 DINOv2 구현](https://github.com/facebookresearch/dinov2)과
[모델 설명](https://huggingface.co/facebook/dinov2-base)은 고정 특징 위 분류기 학습을 설명한다.
InsPLAD의 실제 효과는 이번 Colab을 돌린 뒤 판단해야 한다. 새 장비·새 결함으로의 범용 성능은 검증하지 않는다.

## 학습 누출 차단

1. 공식 supervised val6417·unsupervised test6481 전부와 dev240·gold246 대상 사진을 보호 목록에 넣는다.
   dev는 기존 QA, gold는 실행 커밋에 고정된 `manual_review_final.csv`를 사용한다.
   QA/사진/보호 목록 누락, 잘못된 공식 이미지 수, 승인되지 않은 gold 목록은 실행을 중단한다.
2. 후보는 supervised train의 glass-insulator 두 클래스만이다. 다른 결함은 자동 변환하지 않는다.
3. 보호 사진과 파일명·크기+CRC32·SHA256·EXIF 방향 보정 후 RGB 픽셀 SHA256이 겹치면 제외한다.
   이름 변경·메타데이터가 다른 재저장도 차단한다. 손실 압축이나 다른 crop 등 **유사 사진 전체를 보증하지는 않는다**.
4. 남은 train 내부의 동일 RGB 사진은 한 장만 유지한다. 동일 픽셀의 라벨 충돌은 중단한다.
5. 촬영일을 단위로 약 60/20/20%의 fit/calibration/internal_test로 나눈다.
   seed42의 첫 유효 분할을 사용하며, 각 분할에 두 클래스가 최소10장씩 있어야 한다.
   유효성 판단에는 train의 날짜·클래스 수만 사용한다. 특징·성능·dev/gold 정답은 사용하지 않는다.
6. `input_lock.json`, `split_manifest.json`에 입력 SHA·코드 SHA·모델 버전·분할을 고정한다.
   입력/코드/설정이 바뀐 상태로 같은 폴더를 재사용하면 중단한다.

2026-10-16 실제 원본 사전 점검: 보호 파일13138개, 후보1381장 중253장 제외,
정상530/누락598 유지. fit 정상343/누락380, calibration 정상158/누락52,
internal_test 정상29/누락166. 내부 촬영일·정확한 바이트/RGB 중복0.
이 점검은 **학습 성능 측정이 아니다**. 클래스 불균형 때문에 정확도 하나로 판단하지 않는다.
dev는 내부 train과 촬영일을 공유할 수 있으므로 내부 촬영일 평가와 별도로 표시한다.

## 보존·재시작·결과

새 실행마다 `MyDrive/lvlm_insplad_results/missing_component_probe_v1/<시간_UUID>/`를 만든다.
같은 런타임에서 실행 셀을 다시 돌리면 성공한 이미지 특징을 재사용한다(20장마다 진행 표시).
새 런타임에서 모두 실행하면 새 결과 폴더가 만들어져 과거 결과를 보존한다.
원본 ZIP은 `/content`로 복사해 읽고 이미지·QA·기존 결과를 수정하지 않는다.

- `leakage_audit.json`, `image_manifest.csv`, `excluded_images.csv`: 제외 이유와 분할.
- `input_lock.json`, `split_manifest.json`, `frozen_detector.json`: 재현 가능한 입력·모델·임계값.
- `predictions.csv`: 사진별 정답·점수·판정. 확률은 모델 점수이며 외부 보정된 신뢰도는 아니다.
- `leaderboard.csv`, `metrics_by_asset.csv`: fit/calibration/internal_test/dev 각각의 TP/FP/TN/FN,
  Precision/Recall/F1, BAcc, 정상 오탐률, 누락 놓침률, ROC-AUC/AP와 항상 같은 답 기준선.
- `run_metadata.json`, `run_status.json`: 실행 환경·완료 여부. 실패 결과를 완료로 표시하지 않는다.

오류가 있으면 결과 확인·다운로드 셀이 중단한다. `frozen_detector.json`이 있더라도
`run_status.complete=true`가 아니면 평가 완료 모델로 사용하지 않는다. 설정 선택·agent 적용은 자동 실행하지 않는다.
마지막 셀이 `missing_component_probe_results.zip`을 다운로드한다. 특징 캐시·원본 이미지는 ZIP에서 제외한다.
실제 GPU 추출·전체 학습·성능은 사용자 Colab에서 확인한다.

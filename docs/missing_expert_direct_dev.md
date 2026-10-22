# 누락 판별기와 위치후보 직접 채택 검증

실행 파일: **`notebooks/lvlm_missing_expert_direct_dev_colab.ipynb`**.
L4/A100 → 기본값 유지 → 모두 실행 → Drive 연결 허용.
결과: **`missing_expert_direct_dev240_results.zip`**. 이전 코랩 재실행/추가 표시/토큰 입력은 불필요하다.

## 목적과 범위

이전 독립 유무 진단은 실제60회 모두 누락 없음이었다. 이번에는 LVLM의 재검토가 동결 누락 판별기 결과를 취소하지 않도록 역할을 바꾼다. 기존 dev240 답을 보존하고 유리 절연체50장에만 전문가를 적용한다. 이전에 탐색한 dev이며 새 holdout/최종 Agent 실험이 아니다. Q1/Q2/최종 gold/재학습/임계값 선택/새 LVLM 호출은0이다.

유무 양성은 동결 판별기가 결정한다. 위치는 기존 3×3칸별 흐리기/지역 평균색 가리기의 양의 logit 감소가 함께 나타나는 후보 중1순위만 선택한다. 두 방식의 L2정규화 감소량 중 작은 값으로 순위를 매기며 동점은 칸 번호 순서다. 후보가 없으면 **누락 있음·위치 보류**로 기록한다. 가림 인공물/배경에 반응할 수 있어 후보는 정답이 아니다.

## 고정 비교

| 조건 | 경로 | 위치 |
|---|---|---|
|cached_lvlm|기존 실제240답|원래 답, 무효도 보존|
|same_route_top1/center/random|기존 답이 빈 목록이며 판별기 양성|후보1순위 / 5번 / 고정 해시 무작위|
|all_positive_top1/center/random|기존 답·무효와 무관하게 판별기 양성|위와 동일|

전문가 음성에서는 원래 LVLM 답을 보존한다. 전체 양성 경로는 기존 양성·무효도 별도 새 조건에서 교체할 수 있다. 대조군은 같은 경로·호출수·후보 보류를 공유하며 무작위는 `sha256('matched-random42/'+core_id) % 9 + 1`로 고정한다. GT로 후보/정책/임계값을 고르지 않는다. 기존 CPU 재조합은 후보 없는 경우 원래 답을 유지했으므로, 이번의 유무 양성·위치 보류 정책과 숫자가 달라질 수 있다.

## 입력과 재사용

- 실제 PR63 ZIP `38b050835bbb…`: 원본49개 성공 특징, 동결 head `d9ff0610d0b7…`.
- 실제 PR64 ZIP `dc60885b0e8a…`: 가림414개 성공 특징과 요청/배열/점수/모델 provenance.
- 사용자 실제 표시 SHA `9a9e71b99517…`: 박스24/판단불가1, 위치 채점에만 사용. 정상은125장, 다른 결함의 공간 정답은 계속 unknown.
- 공식 모델 `facebook/dinov2-base@f9e44c814b77203eaa57a6bdbbd535f21ede1415`의 FP32/eval CLS, 전체 RGB square pad(124,116,104),518 resize/no center crop/L2 정규화. 기존 feature 함수를 재사용한다.
- 원본 supervised ZIP 한 개만 필요하다. 내장 특징은 원본 이미지가 아니며, 원본 SHA와 재구성한 RGB/mask pixel SHA·특징 float32/L2/배열 SHA·head 점수를 검사한다. 이전 PNG encoding SHA와 현재 decoded pixel SHA를 구분한다. 플랫폼별 PNG 재인코딩 바이트 차이를 동일 바이트 재현으로 주장하지 않는다.
- 신규 환경은 GT를 보지 않고 정렬된 기존 원본1/가림1을 먼저 실제 재계산한다. `allclose(atol=1e-5,rtol=1e-4)`를 만족해야 신규 특징과 캐시를 함께 쓴다. 실패하면 새 경로 계산을 중단하고 complete=false/비교 부적격으로 저장한다. 임계값을 느슨하게 바꾸지 않는다.

미계산 원본1개와 새 양성 경로의 가림만 계산한다. 정확한 새 가림 수는 그 원본의 동결 판별 결과에 따라 달라진다. canary2개는 별도 비용으로 기록한다. DINO만 필요할 때 로드하며 InternVL 가중치를 받지 않는다.

## 저장·재개·평가

별도 Drive `lvlm_insplad_results/missing_expert_direct_dev240_v1`에 atomic JSON/특징/도구 점수/가림 후보/예측/metadata/status를 저장한다. 코드·입력·software 버전·정책이 달라지면 같은 폴더로 재개하지 않는다. 성공 특징/성공 canary는 SHA 확인 후 재사용하고 실패 시도는 이력을 남겨 재시도한다. 원본 `_images`는 결과 ZIP에서 제외하며 source capsule과 raw 특징/응답/GT를 보존한다.

`present`와 `regions`는 분리된다. 유무 무효는 기존 계약대로 anomalous FN/normal FP다. 위치 보류는 위치 정확도에서 실패로, GT unknown은 분모에서 제외한다. 기존 `q3_multiregion.score` 위치 보고서에 포함된 presence/invalid 필드는 위치 목록에서 파생된 보조값이므로 검출 성능은 반드시 별도 `presence` 항목으로 읽는다. 원래 invalid와 새 위치 보류는 예측의 source/abstention 필드로 구분한다.

전체/자산별/결함별 유무 TP/FP/TN/FN, 위치 완전일치/칸 TP/FP/FN/적어도1칸 적중, 보류 수 및 paired bootstrap CI를 저장한다. 정상FP와 누락FN을 함께 보고, 중앙 기준과의 위치 차이 CI가0을 포함하면 우위를 확정하지 않는다. 실행 실패 조건은 leaderboard에 eligible=false이며 paired 비교를 내지 않는다. 탐색 중인 dev에서 자동 최종 선택/골드 실행을 하지 않는다.

기능 검증은 CPU fixture와 실제 입력 연결 검산이다. 실제 새 DINO 효과는 사용자의 Colab 실행 뒤에 확인한다.

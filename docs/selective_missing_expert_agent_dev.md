# 3번: 정상 Reference + 동결 누락 판별기 + LVLM 통합 평가

실행 파일: `notebooks/lvlm_selective_missing_expert_agent_dev_colab.ipynb`.
[Colab에서 열기](https://colab.research.google.com/github/arnold6444/lvlm_insplad/blob/main/notebooks/lvlm_selective_missing_expert_agent_dev_colab.ipynb).

기능 제작/CPU 검증과 실제 효과를 구분한다. **새 통합 Agent의 실제 모델 성능은 아직 미검증**이다.
L4/A100 → `N_IMAGES=240` 유지 → 모두 실행 → Drive 연결 허용.
이 파일은 실제 실행용이며 준비만 하는 플래그가 없다. GitHub token/추가 라벨/재학습은 필요 없다.
작은 실행을 원하면 `N_IMAGES=40`과 자동으로 달라지는 별도 출력 폴더를 사용하며, 같은 결과를 독립 반복이라고 하지 않는다.
완료 ZIP은 `selective_missing_expert_agent_dev240_results.zip`이다.

## 앞선 결과를 반영한 선택

Q2 후보 확인(dev40/L4,40호출)은 엄격한 사전 파서가 `A.`4개/`B.`4개를 무효로 판정하여 후보 조건이 제외됐다.
과거 점수를 재해석해 되살리지 않는다. 동일예산 재질문 F1 .622582는 기존 cue .651546보다 낮고 차이 CI는0을 포함한다.
이번에는 그 이진 자기지지 가설을 재실행하지 않는다. Q2는 기존 벤치마크 MCQ 파서(`A.` 허용), Q3는 PR61의 기존 JSON 집합 파서를 사전에 고정한다.

누락 판별기 완료 결과: dev 정상25/누락25에서 TP25/FP1/TN24/FN0; 촬영일 분리 internal-test 정상29/누락166에서 TP164/FP4/TN25/FN2.
동결된 threshold .0306027427376333을 그대로 사용한다. calibration 정상 오탐률 상한5%는 평가/운영 보증이 아니며 internal-test 정상 오탐은13.79%였다.
판별기는 기존 train의 glass-insulator 정상/누락 라벨로 학습한 **지도학습 보조 조건**이다. 다른 자산/결함이나 일반 현장으로의 일반화를 주장하지 않는다.

과거 dev240 cue 답+같은 dev의 판별기 예측을 CPU에서 단순 결합하면 Q2 F1 .692612→.838622, 정답190→208/240,
누락 정답6→25/25, 기존 오답19건 복구/정답1건 퇴행이었다. **저장된 응답의 규칙 결합 결과**이며 새 온라인 Agent나 LVLM 확인의 실측 성능이 아니다.
이번 실험의 단순 결합 대조군은 새 첫 판단을 사용하므로 이 수치를 그대로 재현한다고 가정하지 않는다.

앞선 Q3 유무 gate는 실제 gold에서 FP29→15와 함께 FN31→35, 양성 위치 완전정답12/133동일이었다.
이번에는 Q3 직접 위치 질문부터 실행하고 이미 찾은 위치/누락을 도구 음성으로 취소하지 않는다. gold를 다시 돌리거나 튜닝하지 않는다.

## 질문별 흐름과 대조군

1. native InternVL이 원본 전체 target+동일 자산 train-normal reference로 새 Q2 유형과 새 Q3 복수 위치를 각각 판단한다.
2. glass-insulator이며 **현재 Q2**가 누락 이외의 유효한 답이거나 **현재 Q3**가 유효한 빈 집합일 때만 누락 판별기를 요청한다.
3. 도구가 양성일 때만 현재 질문의 검토를1회 수행한다. 같은 사례에 도구 제안을 숨긴 재질문도1회 수행한다.
4. 도구 음성/미요청이면 첫 답 유지. 도구는 위치를 주지 않으며 Q3 영역을 규칙으로 만들어 넣지 않는다.

| 정책 | Q2 | Q3 |
|---|---|---|
| lvlm_only | 새 cue 첫 답 | 새 직접 복수 위치 첫 답 |
| expert_q2_override | 필요 시 도구 양성을 누락 답으로 결합 | 첫 위치 그대로 |
| selective_agent | 도구 양성 충돌에서 제안을 받은 LVLM 검토 | 첫 빈 위치+도구 양성에서 LVLM 검토 |
| matched_reask | 같은 양성 경로에서 자기 답을 보고 재질문 | 같은 양성 경로에서 자기 답을 보고 재질문 |

각 질문의 라우팅/프롬프트는 다른 Q1/Q2/Q3 답·정답·GT 경로를 읽지 않는다. 단, 같은 사진의 순수 이미지 특징은 내용 해시로 공유해 중복 추출을 줄인다.
Q1은 기존 cache를 고정하고 신규0호출이다. Q1 성능을 향상한 시스템이라고 주장하지 않는다.
tool probability를 보정된 자신감으로 설명하지 않고 모델에는 양성 제안만 전달한다. 제안이 틀릴 수 있음을 명시하고 원본 사진으로 판단하게 한다.
순서는 질문 ID 해시로 agent/control을 번갈아 실행하며 GT로 순서를 정하지 않는다.

## 모델/범위/비용

- InternVL3.5-8B revision `9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79`, native BF16/greedy/출력상한80토큰/이미지별3타일.
- expert: DINOv2-base revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`, 전체RGB 정사각패딩/518리사이즈/중앙crop없음/L2 CLS.
- 동결 head SHA256 `d9ff0610d0b74752a2dfc5b53f1cdc1db5e916e17b4260172f1c7abab8c6e7f1`. fit/calibration만으로 정한 scaler/head/threshold, 새 학습/임계값선택0.
- 240장=기본 Q2/Q3 480회 + 도구 양성 질문마다 agent1/control1. glass50장이므로 최대680 LVLM호출/DINO 최대50장.
- 정책별 논리 호출과 공유된 실제 호출, 신규/재사용/blocked, 도구 요청/특징추출, 오류/무효/지연시간을 구분한다. 같은 호출 수가 같은 토큰/시간은 아니다.
- `matched_reask`도 동일 expert 양성 경로를 사용한다. 비교는 **같은 경로에서 도구 제안 정보의 추가 가치**를 시험하며 라우터 하나만의 순수 효과가 아니다.
- runner wall time은 모델 다운로드/노트북 ZIP 내보내기 시간을 포함하지 않는다.
- 두 모델은 native GPU에서 필요할 때 로드한다. GPU/OOM 오류를 숨기거나 4bit/다른 모델로 대체하지 않는다.

## 평가와 실패 처리

Q2 macro-F1/BAcc/정확도·결함별 정답/false-normal, 초기 대비 정답퇴행/복구, 자산별 지표, Q1 고정·Joint/FDR/모순,
Q3 유무 TP/FP/TN/FN·정상오탐/미탐·무효, Joint Q1+Q2+Q3유무(위치정확도 아님), paired image-bootstrap CI를 저장한다.
all240/design40/additional200은 별도 진단하며 모두 이미 분석한 dev다. 추가200을 새 미사용 holdout으로 부르지 않는다.
Q3는 `{"regions":[1,9]}` / `{"regions":[]}` 계약이며 양성 공간정답은 전부 미상, 양성 위치정확도는null/지원수0이다.
정상 사진의 빈 위치 정답률을 양성 위치 정확도로 바꿔 부르지 않는다.

무효 최종 Q2는 오답, Q3 무효 anomalous는FN/normal은FP로 집계하고 비율·raw를 함께 남긴다. 무효를 성공할 때까지 재질문하지 않는다.
실행/도구 오류가 있는 정책은 leaderboard/CI에서 제외한다. 이전 유효 첫 답을 남기더라도 오류를 성공으로 숨기지 않는다.
새 호출은 매회 저장하며, 같은 입력의 성공·무효를 재사용하고 오류만 재시도한다. 실패한 과거 raw는 attempt_history에 보존한다.
입력/모델/코드/요청/원본 이미지/head/feature cache가 바뀌면 같은 폴더 재개를 거절한다. 기존 QA/결과/원본 ZIP을 덮어쓰지 않는다.

ZIP에는 입력·head·원본 출처 해시·QA·이미지 SHA·모든 프롬프트/raw/분기/점수·작은 도구 특징·지표/CI/metadata/status·소스 묶음을 저장한다.
원본 이미지와 모델 가중치는 포함하지 않는다. `run_status.complete`와 반환 코드/오류/무효를 함께 확인한다. 다운로드 자체는 효과/성공의 증거가 아니다.

# IG 계산 오차 진단 Colab

**파일: `notebooks/lvlm_missing_ig_numerics_colab.ipynb`**

L4/A100 → 기본값 유지 → 모두 실행 → Drive 연결 허용. 기존 supervised 원본 ZIP 하나를 자동 검색한다. 이전 코랩 재실행, 추가 표시, 토큰 입력은 불필요하다. 결과는 **`missing_ig_numerics_results.zip`**. 기능 제작/CPU 검증과 실제 pretrained GPU 진단은 구분하며, 후자는 사용자 실행 전이다.

## 목적과 고정 범위

실제 PR69는 실행 오류0이지만 completeness 경고49/52, 위치 완전일치 기존12/24→IG합의8/24였다. 이 코랩은 위치 정책을 조정하기 전에 수치 오차를 진단한다. 기존 판별기 양성26개의 core ID를 `sha256('ig-numerics-v1/'+core)`로 정렬해 앞의4장을 고른다. **GT/경고/이전 위치 성능으로 고르지 않는다.** 선택된4장에는 정상 오탐1장도 그대로 포함된다. 사람의 위치 라벨은 묶음에 넣지 않는다. 이 작은 표본으로 전체 위치/검출 정확도를 추정하지 않는다.

원본 full-RGB square padding/518resize/PIL, 고정 DINOv2 revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`, FP32/eval/L2-CLS, 기존 head SHA `d9ff0610d0b74752a2dfc5b53f1cdc1db5e916e17b4260172f1c7abab8c6e7f1`와 sigmoid 이전 logit을 유지한다. 모든 가중치 requires_grad=false, optimizer/학습/LVLM/Q1/Q2/gold 호출0. 기존 유무/위치 답은 변경하지 않는다.

## 무엇을 비교하는가

- **적분 해상도:** 같은 blur/mean 기준에서 midpoint32/128/512단계. Gauss-Legendre128단계도 별도 계산한다. [Captum 공식 설명](https://captum.ai/api/integrated_gradients.html)은 이 적분 근사와 completeness 진단을 설명한다. 추가 Captum 설치 없이 NumPy leggauss로 점/가중치를 만든다.
- **누적 정밀도:** 각 forward/backward의 동일 FP32 gradient를 FP32와 FP64로 동시에 누적한다. FP64 조건도 네트워크와 입력은 FP32이며, 네트워크 전체를 double로 바꾼 비교가 아니다. Midpoint FP32는 이전32단계 코드와 같은 연산 순서를 재현한다.
- **기울기:** 경로 alpha .1/.5/.9에서 `gradient · (원본-기준)`과 logit의 centered finite difference를 비교한다. epsilon .01/.001/.0001을 모두 보존한다. 고정 허용치 max(.1, .05×max(|미분|,|차분|)) 이내 여부를 각 값과 함께 저장한다. 유한차분 역시 절단/반올림 오차가 있어 완전한 정답으로 취급하지 않는다.
- **Attention 경로:** 동일 모델에서 기본 경로와 강제 SDPA MATH context를 비교한다. [Hugging Face 공식 설명](https://huggingface.co/docs/transformers/en/attention_interface)의 SDPA context를 사용한다. 실제 설정과 Attention 클래스명을 기록한다. eager 모델이라면 context가 계산을 바꾸지 않을 수 있으며, 이를 독립 backend 검증으로 주장하지 않는다. 모델을 재로드하거나 가중치를 바꾸지 않는다.
- **재현/맵 안정성:** 이전 PR69의 입력 SHA·signed map SHA·양수 패치맵과32단계 재계산을 비교한다. 32→128→512, GL128↔midpoint512 signed map cosine/relative L1도 남긴다. 적분 합만 맞고 맵이 불안정할 수 있으므로 둘을 함께 확인한다.

Completeness는 signed RGB합의 전체 합과 `원본logit-기준logit` 차이로 계산한다. 이전 허용치 max(.1,.1×|차이|)를 보존한다. 단계 수를 결과에 따라 자동 증가하거나 임계값을 바꿔 통과시키지 않는다. 경고를 숨기거나 실패 사례를 빼지 않는다. 현재 구성에서는 단계 부족, 누적 오차, gradient/backend 일치에 관한 증거를 얻을 수 있지만 한 번의 진단으로 원인이 반드시 분리된다고 보장하지 않는다.

## 입력·체크포인트·출력

실제 ZIP SHA `d9170b124d97f6c7cf2bc5d626bf397cb4167b69d9a296d3e63c4ca7d8248d7d`와 source66e7…의 모든 파일/선택 이미지/특징/이전맵 해시를 builder에서 검사한다. Runtime은 source manifest/설정/환경을 잠그고 기존 원본·가림 canary2개와 각 조건의 원본 특징을 확인한다. 고정 환경은 torch2.11.0/transformers5.14.0/numpy2.1.3/Pillow11.3.0/timm1.0.30.

최대 **6,810forward/6,448backward**, 모델1개/batch1. 4장×2기준×(적분4조건+미분검사1조건)=**40개 체크포인트**다. 최초 canary2개 외, baseline별 적분808forward/800backward와 미분43forward/6backward를 계산한다. 비용을 한정하기 위한 고정 프로토콜이며 전체26장 sweep은 하지 않는다.

Drive `lvlm_insplad_results/missing_ig_numerics_v1`에 저장한다. 같은 설정의 완료 조건은 다시 계산하지 않는다. 완료된 전체 실행은 재개 시 모델 로드/새 추론0이다. 중단된 한 조건은 다시 계산할 수 있다. source/환경/입력/캐시가 달라지면 거부하고 실패 이력을 보존한다.

- `records.json`: 40조건의 입력 SHA, geometry, logit, signed map·feature SHA, completeness, 미분검사와 backend 정보.
- `maps/`, `target_features/`, `canary_features/`, `canary.json`: raw 결과와 재현 증거.
- `integration_summary.csv/json`, `map_stability.json`, `derivative_summary.json`: 비교용 요약.
- `run_metadata.json`, `run_status.json`, source capsule/manifest/console/return code: 실행 증거.

원본 `_images`는 ZIP에서 제외한다. `complete=true`는 계산/체크포인트가 끝났다는 뜻이며 수치 오차 통과나 위치 개선의 뜻이 아니다. 오류/OOM이면 마지막 저장 셀을 실행해 ZIP을 그대로 전달한다. 결과를 분석하기 전 전체 IG 위치 도구를 재실행하거나 채택하지 않는다.

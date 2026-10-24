# PR70 IG 계산 오차 원인 분석

2026-10-23 실제 Colab 결과를 회수하고 원본·코드·맵·계산식을 CPU로 검산했다. **가장 강하게 확인된 원인은 기존 midpoint32단계 적분의 해상도 부족이다. FP32 누적을 FP64로 바꾸는 것만으로는 해결되지 않는다.** 다만 평균색 기준의 잔차와 일부 미분 불일치가 남아 있어 원인을 하나로 확정하거나 전체 IG 위치 도구를 채택할 단계는 아니다.

## 실제 실행 확인

- ZIP `missing_ig_numerics_results.zip`, SHA256 `b80756a9c8f02a50c1b11628e225127db293e3b171a8a471e6945923c2e8224d`.
- 실제 HF/NVIDIA L4, fixture=false, return_code=0, complete=true, condition_errors=0, 40/40조건 완료. 6,810forward/6,448backward, 모델 로드1회,747.7412초.
- Source SHA `c7befa6f6be7c10045e74c06d5f221201f295284ff2ee9705311bce678f08974`,52파일 모두 게시 준비물과 일치. torch2.11.0+cu130/transformers5.14.0/numpy2.1.3/Pillow11.3.0/timm1.0.30.
- 고정 해시로 고른4장×blur/mean2기준=8개 조합이다. 정답이나 오차 크기로 선택하지 않았으며 정상 오탐1장도 포함한다. 위치 GT는 소스 묶음에 없다. 전체26장 또는 독립 holdout 평가가 아니다.
- 원본/가림 seed6개 SHA·RGB, 특징40개, raw signed map64개 SHA/dtype/shape/finite,canary2개,head/입력/설정/요청/payload SHA를 확인했다. Canary와 조건별 원본 특징은 이전 특징과 byte/value가 정확히 같았다.
- 원본 ZIP의 모든 출력은 감사 후에도 byte-identical함을 확인했다. 새 GPU/신경망 추론/학습은0. 저장한 gradient 자체를 새로 계산하지 않고 map합과 finite difference144개 및 backend24쌍의 산식을 재계산했다.

## 1. 적분 단계 부족 — 강한 증거

아래는 **동일4장×2기준의8개 조합**이다. 상대 오차는 `|signed 기여 합 - (원본logit-기준logit)| / |원본logit-기준logit|`. 기존 허용치 max(절대0.1, logit차이의10%)를 유지했다. 이 completeness 성질은 [Captum 공식 설명](https://captum.ai/docs/extension/integrated_gradients)과 같다.

| 적분 조건 | 상대 오차 중앙값 | 경고 | 절대 잔차 평균 |
|---|---:|---:|---:|
| Midpoint32 | 125.0% | 8/8 | 8.096 |
| Midpoint128 | 83.3% | 8/8 | 6.316 |
| Midpoint512 | 13.5% | 4/8 | 0.869 |
| Gauss-Legendre128 | 7.4% | 3/8 | 0.999 |

32→512에서 8개 중7개의 상대 잔차가 줄었다. 기존32단계는 적분 근사로 충분하지 않았다는 증거다. 하지만 중간 단계에서는 잔차 부호가 바뀌거나 오차가 커지는 사례도 있어 단조 수렴이라고 부르면 안 된다. Gauss-Legendre128은 중앙값/경고 수가 작지만 최악 상대 오차146.6%, 절대 잔차 평균도 midpoint512보다 크므로 자동으로 더 좋은 설정이라고 선택하지 않는다.

맵의 변화도 단계 부족을 지지한다. Midpoint32↔128의 signed map 상대 L1 차이는 중앙20.6%(최대43.2%),128↔512는 중앙6.0%(최대12.4%)다. GL128↔midpoint512는 중앙0.93%(최대2.20%), 최소 cosine0.99962로 대체로 비슷해진다. 다만 맵 전체가 비슷하다는 것과 signed 합의 잔차 통과, 실제 결함 위치 정확도는 서로 다른 조건이다.

## 2. FP32 누적 정밀도 — 이번 큰 오차의 주원인에서 제외

동일한 FP32 gradient를 FP32/FP64로 동시에 누적했다. 두 방식의 completeness 잔차 차이는 전체32조건에서 최대 **2.36e-6 logit**, 맵 상대 L1 차이는 최대 **3.20e-7**이다. 실제 문제가 된 잔차는0.1~수십 logit 수준이므로 누적 반올림만으로 설명하기 어렵다.

이 비교는 누적기만 바꿨다. **모델 forward/backward까지 전체 FP64로 검증했다는 뜻은 아니다.** 입력/모델은 FP32이며 GPU convolution 등의 내부 연산 정밀도는 별개다.

## 3. 기준 이미지 의존성 — blur와 mean을 구분해야 함

| 기준 | Midpoint512 경고 | 상대 오차 중앙값 | GL128 경고 | 상대 오차 중앙값 |
|---|---:|---:|---:|---:|
| Blur | **0/4** | 2.7% | **0/4** | 5.2% |
| Mean | **4/4** | 19.7% | **3/4** | 28.4% |

평균색 기준은512단계에서도 모두 경고가 남는다. 평균색 이미지에서 출발하는 경로가 수치적으로 더 어렵다는 결과지만, 특정 alpha 부근의 급격한 변화 때문인지까지는 저장된 자료로 확정하지 못했다. 원본과 기준 logit 차이가 작은 정상 사례의 상대 비율은 커지기 쉽지만, 평균색 기준의 절대 잔차도 평균1.57 logit으로 작지 않아 작은 분모만의 문제는 아니다.

## 4. 미분 불일치와 attention — 일부 미해결

같은24개 경로 지점을 기본 SDPA와 강제 SDPA MATH에서 검사했다. **두 방식 모두 같은4/24지점에서 세 epsilon 중 어느 것도 사전 허용치를 통과하지 못했다.** 두 backend를 합친48개 검사 중8건이다. 모두 `dev-3f1ef393e4238ae8`에 집중된다: blur alpha .1/.5/.9,mean alpha .9.

- 두 backend의 경로 logit 차이는 최대2.71e-5, 방향미분 상대 차이는 최대0.694%다. MATH로 바꿔도 실패 지점이 해소되지 않았다. 따라서 기본 attention backend 하나가 원인이라는 증거는 약하다. 전체 IG를 MATH backend로 적분한 실험은 아니므로 모든 backend 영향을 배제하지는 않는다.
- 유한차분 epsilon을 줄일수록 각 backend의 통과가 **19/24(.01) →13/24(.001) →1/24(.0001)**로 나빠졌다. 작은 epsilon의 반올림/연산 정밀도 민감성과 맞는 패턴이다. 이를 보고 autograd 자체가 틀렸다고 확정하면 안 된다.
- 예: 해당 이미지 blur alpha .1에서 autograd는5.131인데, epsilon .01/.001/.0001의 차분은6.623/0.902/21.225다. 매우 작은 epsilon을 미분 정답으로 취급할 수 없는 상태다.

### 추가로 발견한 precision 제어 공백 — 원인 후보, 아직 미검증

sealed source의 `DinoFeatures`는 `torch.backends.cuda.matmul.allow_tf32=False`만 설정하며 **cuDNN convolution TF32를 명시적으로 끄거나 실행 metadata에 기록하지 않았다.** 같은 transformers5.14.0의 로컬 참조 소스에서 DINO patch projection이 Conv2d임을 확인했다. Attention MATH context는 이 convolution 설정을 대신 통제하지 않는다.

[PyTorch2.11 수치 정확도 문서](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html#tensorfloat-32-tf32-on-nvidia-ampere-and-later-devices)는 matmul과 convolution의 TF32 기본값/정밀도가 다름을 설명하고, [CUDA semantics](https://docs.pytorch.org/docs/2.11/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)는 각각 별도 설정임을 명시한다. 따라서 FP32 파라미터라는 사실만으로 모든 내부 연산이 full IEEE FP32였다고 결론낼 수 없다.

**이번 실제 L4 실행에서 TF32 convolution이 사용됐는지는 로그에 없고, 켰다/껐다 대조도 하지 않았다. 따라서 TF32가 원인이라고 확정할 수 없다.** 로컬 CPU torch2.11의 기본 flag=true를 확인했지만 이는 Colab에서 실제 사용됐다는 증거로 삼지 않는다. 다음 통제 실험의 우선 후보로 남긴다.

## 입력·저장·환경 문제와 구분

8개32단계 재현에서 원본/기준 입력 SHA는 이전 PR69와 전부 일치했다. 양수 patch map cosine은 최소0.999999999994, 상대 L1 차이 최대2.58e-6으로 이전 결과를 사실상 재현했다. NPY 파일8개는 비트 단위로 같지는 않으므로 완전 동일 결과라고 쓰지 않는다. 큰 completeness 경고가 잘못된 이미지나 feature 변경 때문에 생겼다는 증거는 없다.

감사 초기에 NumPy2.4.6의 합계가 native2.1.3과3.55e-15 달라 기존 cache의 exact-float 비교가 실패했다. 실제 맵 SHA는 일치했다. 기존 검사를 약화하지 않고 보고서 폴더 안에 NumPy2.1.3을 별도 설치해 동일 버전으로 다시 감사했고, 원래 `check_record`와 모든 검산을 통과했다. 이 차이는 logit 단위의 실제 큰 오차와 무관하다. 전역 환경/실험 원본은 바꾸지 않았다.

## 결론과 다음 행동

1. **확인됨:** midpoint32는 이4장의 수치 적분에 부족하다. FP64 누적 전환만으로는 해결되지 않는다. Blur는512단계와 GL128에서8검사 모두 기존 허용치를 통과했다.
2. **남음:** mean 경로의 잔차와 특정 이미지의 미분 불일치. 더 작은 epsilon이나 attention MATH로만 바꿔서는 해결되지 않았다. TF32 convolution은 통제/기록이 빠진 유력한 확인 후보이지 확정 원인은 아니다.
3. **권장 다음 실험:** 같은4장·같은가중치/입력에서 cuDNN convolution의 정밀도와 실제 flag를 명시적으로 기록·통제하고, 반복 forward 및 같은 유한차분을 먼저 비교한다. 해당 효과가 없으면 baseline 근처의 경로 변화와 적분 수렴을 추가로 분리한다. 이전 feature와 차이가 나면 canary 허용치를 넓히지 말고 정밀도 대조 조건으로 따로 보고한다.
4. 수치 안정성이 확인된 뒤에만 위치 평가로 돌아간다. 이번에 위치 정답을 채점하지 않았으며, **기존 PR68의 위치12/24를 대체할 근거는 없다.** 전체26장 재실행/새 코랩 제작/merge는 이번 분석에서 하지 않았다.

증거: `audit_receipt.json`, `integration_detail.csv/json`, `finite_difference_detail.json`, `backend_comparison.json`. 감사 코드: 상위 폴더 `audit_actual.py`. 원본 ZIP/출력/소스 모두 보존했다.

# IG 위치 도구 dev 실험

실행: **`notebooks/lvlm_missing_ig_location_dev_colab.ipynb`**.
**L4/A100 → 기본값 유지 → 모두 실행 → Drive 연결 허용.**
결과: **`missing_ig_location_dev240_results.zip`**. 이전 코랩 재실행/추가 표시/토큰 입력 불필요. 기존 supervised 원본 ZIP 한 개만 필요하다.

기능 제작과 CPU 검증은 실제 GPU 효과와 구분한다. 기존 PR68은 누락25/25를 검출했지만 위치 완전일치12/24였고 중앙8/24 대비 차이CI는0을 포함했다. 이번은 그 위치 도구만 바꾸는 탐색적 dev 실험이다. 누락 유무와 정상 유무 오탐은 모든 조건에서 **동결**하므로 이번 방법의 검출 개선으로 보고하지 않는다. Q1/Q2/LVLM/재학습/gold 호출0.

## 방법과 한계

[Integrated Gradients 원 논문](https://proceedings.mlr.press/v70/sundararajan17a)과 [Captum 공식 completeness 설명](https://captum.ai/api/integrated_gradients.html)을 참고했다. IG는 기준 입력에서 실제 입력까지의 경로에 대한 기울기를 적분해 판별 점수의 입력 기여도를 계산한다. 이번 코드는 midpoint32단계를 직접 구현하며 Captum 의존성을 추가하지 않는다.

고정 `facebook/dinov2-base@f9e44c814b77203eaa57a6bdbbd535f21ede1415`의 FP32/eval CLS→L2정규화→기존표준화·로지스틱 head의 **sigmoid 이전 logit**을 설명한다. 가중치는 requires_grad=false이며 optimizer가 없다. 원래 RGB square pad(124,116,104),518resize/no center crop을 유지한다. 입력 픽셀에 대한 역전파는 학습이 아니다.

기준은 원본 전체 Gaussian blur(radius=max(2,min(w,h)/12))와 원본 전체 RGB평균색이다. 둘 다 실제 정상 사진이 아니며, IG가 높다고 그 위치에 결함이 있다는 보장은 없다. 배경·다른 부품·기준의 선택에 민감할 수 있다. 원 논문의 IG와 별도로, 아래 **위치 후보 생성은 새로 시험하는 자체 규칙**이다.

1. RGB기여도를 합친 signed518×518 맵을 저장한다. 양의 픽셀 기여만 선택하되 원본에 없는 square padding은 제거한다.
2. 14×14픽셀 블록으로37×37 점수를 만들고 L1정규화한다. 두 기준 맵의 작은 값을 합의 맵으로 삼는다.
3. peak의50% 이상인 패치의8연결 성분을 찾는다. threshold후 전체질량의15% 이상인 성분 중 최대3개를 질량순으로 고른다. 각각 가중 중심을 원본 좌표로 복원하고 기존 box-center9칸 경계 규칙으로 바꾼다. 같은 칸은 중복 제거한다.
4. 합의 맵이 비거나 성분이 없으면 위치 보류, 유무양성은 보존한다. 임계값·단계수·성분수는 GT 성능을 보고 탐색하지 않고 이번 protocol에 고정했다.

기준 logit과 실제 logit 차이, signed attribution합, 둘의 차이인 completeness residual을 저장한다. 오차가 max(.1,.1×|logit차이|)보다 크면 경고를 낸다. 이 수치는 고정된 수치 진단 규칙이며 허용오차를 바꿔 통과시키지 않는다. 경고 사례를 지표에서 빼거나 숨기지 않고, 실행 완료와 근사 수렴의 품질을 구분한다. NaN/shape/모델 오류는 비교 부적격으로 저장한다.

## 고정 비교

|조건|내용|
|---|---|
|cached_pr68|이전 실제 top1 위치와 유무 그대로|
|ig_blur_top1 / ig_mean_top1|각 기준의 최대 성분 한 개|
|ig_consensus_top1|두 기준 합의의 최대 성분 한 개|
|ig_consensus_multi|합의 성분 최대3개, 칸 중복 제거|
|occlusion_matched_k|multi와 같은 K개를 기존 가림 점수의 전체9칸 순위에서 선택|
|center_matched_k|동일K,고정순서5/2/8/4/6/1/3/7/9|
|random_matched_k|동일K,core/cell 해시로 고정된 순위|

`occlusion_matched_k`는 기존 positive-agreement의 보류 규칙을 쓰는 운영 정책이 아니라, **양수 여부와 무관하게 전체 순위에서 K개를 고르는 강제 개수 대조군**이다. 기존 운영 결과는 cached_pr68로 보존한다. K는 예측 성분 수에서만 정하며 GT는 전달하지 않는다. 단순히 여러칸을 더 찍은 효과와 순위/다중부위 효과를 구분한다.

## 입력·실행·재개

실제 PR68 ZIP SHA `bad35ced5b0b7322630c4630ea28a1638c520d32d79b635520006fffe80e1a38`의 baseline240, 동결 판별기50/양성26, 가림기록26, 원본특징50·가림canary1, head 및 평가전용 GT를 내장한다. source capsule/각파일/특징/이미지 SHA를 검사한다. 원본·가림 canary2를 새로 계산하고, 각26장의 원본특징도 기존것과 비교한다(atol1e-5/rtol1e-4). 실패하면 해당위치를 신뢰 가능한 성공 결과로 올리지 않는다.

26장 모두 신규 수행시 최대1744forward/1664backward(32단계×2기준)다. DINO 하나만 batch1로 필요할 때 로드한다. pretrained 추론과 gradient의 실제 효과는 사용자 Colab에서 확인한다. 원시 signed map52개가 약56MB이며 압축 결과는 수십MB일 수 있다.

별도 Drive `lvlm_insplad_results/missing_ig_location_dev240_v1`에 이미지별 atomic JSON/맵/target특징/진단/예측/평가를 저장한다. 동일source/input/software/config만 재개한다. 성공 이미지와 canary는 SHA/파생결과까지 확인하고 재사용하며, 실패는 이력을 보존하고 재시도한다. 중간에 끊긴 한 이미지의 적분은 다시 계산할 수 있지만 이전 성공 이미지는 재계산하지 않는다. 실패·OOM·적분경고를 로그와ZIP에 남긴다.

출력은 `attributions.json`, `signed_maps/`, `target_features/`, `canary.json`, `predictions.json`, 전체·그룹·macro metrics, `leaderboard.csv`, `paired.json`,metadata/status/source이다. 원본 `_images`는 ZIP에서 제외한다. 위치 unknown1을 정상으로 바꾸지 않으며 정상125장의 오탐칸/보류, 위치 완전일치·적어도1칸·칸TP/FP/FN와 다중부위를 함께 본다. 기존 무효의 유무 FP/FN 계약은 보존한다.

이미 반복 검토한 dev이므로 최종 holdout/골드/전체 Agent 효과가 아니다. GT24개는 채점만, 아직 실제 위치 성능은 미검증이다. 실패해도 마지막 저장 셀의 ZIP을 전달하고, 경고를 없애려고 임계값/단계수/허용오차를 임의 변경하지 않는다.

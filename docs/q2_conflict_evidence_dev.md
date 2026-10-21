# Q2 충돌 후보별 확인 + 동일 호출 수 대조

실행 파일: `notebooks/lvlm_q2_conflict_evidence_dev_colab.ipynb`.
기능 제작 및 CPU 검증 단계이며 **새 구조의 실제 효과는 아직 미검증**이다.

## 실행

[Colab에서 열기](https://colab.research.google.com/github/arnold6444/lvlm_insplad/blob/main/notebooks/lvlm_q2_conflict_evidence_dev_colab.ipynb) → L4 또는 A100 선택 → 모두 실행 → Drive 연결 허용.
`N_IMAGES=40`을 유지한다. 실행을 끄는 기본 플래그는 없으며 실제 GPU 모델을 실행한다.
내장 소스와 입력을 사용하므로 GitHub 토큰과 수동 라벨링은 필요 없다.
Drive의 원본 `supervised_fault_classification.zip` / `unsupervised_anomaly_detection.zip`을 자동 검색한다.
자동 검색이 실패할 때만 `SUP_ZIP_PATH` / `UNSUP_ZIP_PATH`에 실제 경로를 넣는다.
자체 실행 코드를 포함하며 기존 응답과 이미지 해시는 내부 입력 묶음으로 고정한다.

- 출력 폴더: `MyDrive/lvlm_insplad_results/q2_conflict_evidence_dev40_v1`.
- 결과 ZIP: `MyDrive/q2_conflict_evidence_dev40_results.zip` (자동 다운로드).
- 정상 완료: `run/run_status.json`의 `complete=true`, 모든 정책 `eligible_for_comparison=true`, Q1 고정, 신규40회/오류0/무효0.
- 오류/무효 시에도 ZIP을 내보낸다. 다운로드 자체를 성공 판정으로 쓰지 않는다.
- 끊기면 같은 파일/설정으로 재실행한다. 호출마다 저장하며 성공·무효는 재사용하고 오류만 재시도한다. 바뀐 소스/입력/이미지/모델을 같은 폴더로 재개할 수 없다.

## 분석 근거와 범위

보존된 refined-dev 실제 결과의 초기 답과 `q2_define` 원시 출력/프롬프트/이미지 순서를 검산했다.
전체240장에서 초기 정답152→정의 재검토176이며, 77개 충돌은 정답→오답18, 오답→정답42, 오답→다른 오답17이다.
녹70장에서 정답59→48이며 정답→오답13건이다. 내부 모델 원인은 원시 한 글자 응답으로 알 수 없다.
기존 cue 조건은190/240으로 더 강하므로 반드시 비교에 포함한다. 예전 초기 답보다만 좋아졌다고 채택하지 않는다.
노트북의 첫40장은 결과와 무관하게 고정된 기존 design40이다. 충돌10건에서 새40호출을 수행한다.
추가200장도 이미 분석한 개발자료다. 이번40장 결과 확인 후 별도 파일로 준비하며 자동 확대하지 않는다.

## 고정 정책

| 정책 | 동작 | 새 호출/충돌 |
|---|---|---:|
| initial | 기존 초기 답 | 0 |
| definition_review | 기존 자기 답을 본 정의 재검토 | 0 |
| cached_cues | 기존 fresh cue 답, 더 강한 기존 대조 조건 | 0 |
| retain_initial | 충돌에도 초기 답 그대로 유지 | 0 |
| matched_reask | 정의 재검토 답을 보고 재질문, 그 답을 보고 다시 재질문 | 2 |
| candidate_support | 초기 후보와 재검토 후보를 각각 별도 이진 사진 지지 확인 | 2 |

초기 답과 정의 재검토가 다를 때만 두 새 정책을 같은 사례에 실행한다. 라우팅은 정답/다른 질문을 읽지 않는다.
후보 확인은 다른 후보/이전 답/정답 없이 **현재 Q2의 후보 조건 하나와 시각 기준**만 본다.
한 후보만 지지되면 그 후보로 결정한다. 둘 다 지지/둘 다 미지지는 초기 답을 유지한다.
후보 중 하나라도 무효면 최종 답도 무효이며 해당 정책을 비교에서 제외한다. 재질문 첫 응답 무효도 숨기지 않는다.
두 후보가 모두 틀리면 후보 확인은 새로운 정답을 만들어낼 수 없다. ties 유지도 잘못된 초기 답을 보존할 수 있다.
단순 재질문 대조군은 네 선택지를 모두 유지하므로 후보 밖 답으로 교정될 수 있다.

## 통제·해석

새 호출은 모두 같은 native BF16 InternVL3.5-8B snapshot
`9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79`, 정상 reference→target 순서,
원본 이미지/각3타일/출력상한80토큰/greedy로 고정한다. 조용한 양자화나 로컬6GB GPU 대체는 없다.
후보/정책 실행 순서와 이진 선택지 순서는 문항 ID/의미 해시로 정하고 GT는 사용하지 않는다.
Q1은 캐시 그대로이며 신규 호출0, Q3/gold/학습/자동 선택/설정 freeze는 없다.

동일 호출 수는 추가 질문 횟수만으로 설명되는지 보는 대조다. 프롬프트 내용/길이, 이진 대 네 선택지,
고정 결정 규칙도 함께 달라지므로 **컨트롤러 하나만의 순수 인과 효과**로 해석하지 않는다.
실제 토큰 수/시간은 같다고 가정하지 않는다. 정책별 latency와 원시 응답을 보존한다.
과거 초기/정의/cue 호출의 모델 revision은 기록되지 않았다. 신규 두 정책만 같은 snapshot이다.
따라서 이 결과는 캐시 후보를 이용한 탐색이며 새 모델로 처음부터 실행한 전체 Agent의 성능은 아니다.
이미 실행한 define/reask/cue/semantic cross-model 프롬프트를 신규 가설이라고 재실행하지 않는다.

## 평가·보존

Q2 macro-F1/BAcc/정확도, 결함별 정답·false-normal, 정상 false-defect,
자산별 지표, Q1 고정·Joint/FDR/모순, 초기 대비 정답→오답/오답→정답,
cohort별 진단, image-paired CI, raw/신규/재사용/blocked 호출·오류·무효·시간을 저장한다.
오류/무효 정책은 leaderboard/CI에서 제외하되 `cohort_metrics.csv`에는 제외 표시와 진단 수치를 남긴다.
동일 모델의 이진 지지는 외부 증거 모델이나 검증된 사실이 아니다.
누락 결함 미탐·녹 퇴행·정상 오탐을 함께 보고 채택 여부를 결정한다. 이40장만으로 일반화를 주장하지 않는다.

결과 ZIP에는 소스 묶음/해시, 입력 사본/원본 출처 SHA, QA·참조/이미지 SHA, 전체 프롬프트·raw,
결정 분기·지표·CI·metadata/status·console log를 포함한다. 원본 이미지 ZIP은 포함하지 않는다.

## CPU 재현

```powershell
python scripts/prepare_q2_conflict_evidence_dev.py --source-results reports/q2_q3_refined_evaluation_2026-10-14/d348a43771e9/raw --out-dir reports/q2_conflict_evidence_preparation_2026-10-21/v1
python scripts/build_q2_conflict_evidence_notebook.py --dataset reports/q2_conflict_evidence_preparation_2026-10-21/v1/dataset.json --output notebooks/lvlm_q2_conflict_evidence_dev_colab.ipynb --evidence-dir reports/q2_conflict_evidence_preparation_2026-10-21/v1
python -m pytest -q tests/test_q2_conflict_evidence_dev.py --basetemp=.pytest-tmp-q2-conflict
```

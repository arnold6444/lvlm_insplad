# dev 독립 검증 agent 성능 저하 점검 — 2026-10-12

## 결론

InternVL `reask@all` Q1 F1 0.08은 저장된 모델 답 자체에서 재현된다. 파서·F1 계산 오류의 증거는 없다. 현재 코드에서 기준선과 reask는 같은 이미지 순서, 참조 선택 규칙, adapter, 타일 예산, greedy decoding을 사용한다. 주요 차이는 이전 답을 인용하는 재검토 프롬프트다. 프롬프트가 유력한 원인이지만 개별 문구의 인과 효과는 GPU 대조 실행 없이는 확정할 수 없다.

## 직접 읽은 증거

- GitHub 결과 게시: `675ee42587f7251202173ed92b5770dc1f590627`.
- 실행 metadata: `5c903ce229fec663859ad6b71d8b55369610f933`, prompt version 2, NVIDIA L4.
- GitHub `answers.csv`, dev QA 480행, leaderboard, per_question, selection, run_status, run_metadata.
- Drive InternVL `recheck_reask.jsonl` 480행: 원문 응답·파싱 결과·이전 답·프롬프트 길이·오류.
- 원본 ZIP 이미지와 실행 당시 실제 pixel tensor는 이번에 비교하지 않았다. 다른 도구와 Qwen의 원문 응답은 이번에 받지 않았다.

## 수치 재현

Q1 정답: 이상 115장, 정상 125장.

| InternVL | TP | FP | FN | TN | F1 |
|---|---:|---:|---:|---:|---:|
| stage1 | 91 | 15 | 24 | 110 | 0.823529 |
| reask@all | 6 | 29 | 109 | 96 | 0.080000 |

`2*TP/(2*TP+FP+FN) = 12/150 = 0.08`.

- 최종 정상 205/240, 이상 35/240. 이상 recall 6/115 = 5.2%.
- 이전 의미가 이상인 106개 중 104개를 정상으로 변경, 2개 유지.
- 이전 의미가 정상인 134개 중 101개 유지, 33개를 이상으로 변경.
- 137개 답 변경: 정답으로 변경 19개, 오답으로 변경 118개.
- 5개 자산 모두 Q1 정확도 하락. 특정 자산만의 문제는 아니다.
- Q1 보기 순서 두 종류 각각 120개. 고정 A/B 의미를 가정하지 않고 각 문항 options로 의미를 해석했다.
- 이전 이상 답이 A인 46개는 전부 정상으로, B인 60개 중 58개도 정상으로 변경. 단순 A/B 고정 선택만으로 설명되지 않는다.
- Qwen reask는 반대로 최종 이상 173/240. 같은 프롬프트에서 모델별 반응이 다르다.

## 파서·코드 점검

- 원문 480행 재파싱과 저장 predicted_choice 불일치 0, 추론 error 0.
- 모든 원문의 prompt_chars가 현재 version 2 build_prompt로 재구성한 길이와 일치.
- 예: missing-cap 대상의 기존 정답 `A. Anomalous.` → 원문 응답 `B. Normal.`. 모델 응답이 직접 바뀌었다.
- 기준선과 reask 모두 `[diffdate normal reference, target]`, 동일 `infer_hf`, InternVL image prefix 및 num_patches_list 사용.
- InternVL adapter는 두 이미지 각각 동일 타일 한도를 쓰고 `do_sample=False`; reask는 zoom 전처리를 쓰지 않는다.
- `final_answer`는 유효한 재확인 답을 무조건 채택한다. 의심스럽거나 성능이 낮은 답도 유효한 보기이면 기존 정답을 덮어쓴다. 현재 설계에 명시된 동작이지 예외 처리 실패가 아니다.
- 선택 기준은 agent 후보 중 최고를 고르므로 평균 개선이 모두 음수여도 1위를 확정한다. 선택 파일은 개선 효과의 증명이 아니다.

## 프롬프트 차이

기준선에는 `Do not use hidden metadata or external knowledge.`가 있다. agent에는 이 문장이 빠지고 다음 문단이 들어간다:

> This is a second look at this question. In a first look, the answer to this question was "A. Anomalous.". That answer may be right or wrong. Re-examine the TARGET carefully before answering.

이전 답 인용, 재검토 지시, 기존 문장 제거가 동시에 바뀌므로 지금 결과만으로 어느 요소가 원인인지 분리할 수 없다. 정상 참조를 대상으로 오인했을 가능성도 프롬프트 행동 가설이며 이미지 순서 오류로 확인된 것은 아니다.

## 다음 진단

gold는 추가 실행하지 않고 dev에서만 소규모 paired 대조를 해야 한다. 같은 QA/정상 참조/대상/모델/전처리/greedy 설정을 고정한다.

1. 기준선 프롬프트 그대로: 현재 실행 환경에서도 기준선 답 재현되는지 확인.
2. 기준선 + 중립적 재검토 문장, 이전 답 없음.
3. 기준선 + 자기 이전 답만 표시, 재검토 문장 없음.
4. 현재 version 2 프롬프트.

자산·정상/이상·보기 순서를 나눠 고른 dev subset을 모든 조건에서 똑같이 쓴다. 원문 응답과 이미지 경로·해시·타일 수를 저장한다. 기존 결과와 frozen 설정은 보존한다. gold 결과를 보고 프롬프트를 바꾸지 않는다.

## 실제 검증

- Python 3.11: `tests/test_gold_v2_verify_agent.py` + `tests/test_hf_adapters.py`: 29 passed.
- GPU 추론·프롬프트 대조 실험은 이번 점검에서 실행하지 않았다.
- 프롬프트 변경은 아직 적용하지 않았다. 앞선 실패 설정 선택 차단 수정은 별도 로컬 변경으로 유지된다.

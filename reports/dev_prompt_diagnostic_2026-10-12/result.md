# InternVL dev 40장 프롬프트 진단 결과 — 2026-10-12

## 실행 증거

Drive `gold_v2_prompt_diagnostic_v1`에서 status, metadata, leaderboard, answers, significance를 직접 읽음. 4조건 × 80문항 = 320호출, 모두 complete, row_errors 0, invalid_answers 0. GPU 추론은 사용자의 Colab에서 수행됨. metadata의 gpu는 null이다: runner가 torch adapter를 로드하기 전에 gpu_info를 호출하는 기록 시점 문제이므로 이번 metadata만으로 실제 GPU 종류는 증명되지 않는다.

## 비교

| 조건 | Q1 F1 | Q2 macro-F1 | Joint |
|---|---:|---:|---:|
| 기준선 그대로 | .740741 | .488636 | .650 |
| 중립적 재검토 지시만 | .521739 | .500000 | .625 |
| 자기 이전 답만 표시 | .666667 | .488636 | .625 |
| 기존 agent reask 그대로 | .240000 | .300339 | .400 |

같은 40장(이상16, 정상24)에서 Q1 TP/FP/FN/TN: baseline=10/1/6/23, review=6/1/10/23, prior=8/0/8/24, current=3/6/13/18.

## 재현 및 해석

- baseline 새 추론은 기존 stage1 답 80/80 일치.
- current_agent는 이전 전체 dev 실험의 reask 답과 동일 subset에서 80/80 일치. 자기 이전 답도 80/80 일치.
- 즉 0.08 전체 실험의 현상이 같은 입력 subset에서도 재현된다. 여기의 .24는 다른 조건이 아니라 40장 subset 수치다.
- 중립적 재검토 지시만 넣어도 Q1 F1 하락: delta -.219002, paired-bootstrap CI [-.460358,-.047619].
- 이전 답만 넣은 Q1 변화 -.074074, CI [-.352297,.190476]: 이 작은 subset에서 유의한 하락을 입증하지 못함.
- 현재 agent vs baseline: Q1 delta -.500741 CI [-.823529,-.117821], Q2 delta -.188297 CI [-.354932,-.049765], Joint delta -.25 CI [-.375,-.125].
- 환경이나 무작위 추론보다는 프롬프트 변경으로 성능이 떨어지는 현상을 지지한다. 이전 답 표시만이 단독 원인이라고 주장하면 안 된다.
- current_agent는 기존 metadata 금지문장도 제거하고, review_only와 달리 second-look/may-be-wrong 표현을 쓴다. 네 조건은 순수 factorial이 아니므로 어느 단어/상호작용이 원인인지는 완전히 분리하지 못한다.
- CI는 작은 dev 탐색용이고 반복 비교/다중검정 조정은 하지 않았다. gold 결과를 의미하지 않는다.

## 다음 판단

현재 reask 문구를 개선 방법으로 채택하지 않는다. 단순히 모델에게 다시 생각하라고 지시하는 것만으로 성능 향상을 기대할 근거가 없다. 프롬프트 수정 후보는 dev에서 검증해야 하며 gold는 아직 실행하지 않는다. 기존 실험 결과와 frozen config는 보존한다.

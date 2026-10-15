# Q1 유지 · 자동 확대 / 프롬프트 분리 실험

실행 파일: [lvlm_gold_v2_q23_zoom_prompt_dev_colab.ipynb](../notebooks/lvlm_gold_v2_q23_zoom_prompt_dev_colab.ipynb).
새 **L4/A100 런타임**에서 기존 `GITHUB_TOKEN` Secret을 허용하고 **모두 실행**한다.
Drive 인증 외 수동 box·위치 정답·추가 라벨 입력은 없다. 로컬 GPU에서는 실행하지 않는다.

## 목적과 범위

완료된 gold에서는 missing-cap Q2 정답이 최초7→이전5→수정2/27,
Q3 결함 놓침이27/139였다(누락18·새집5·녹4). 누락27장 중25장은 큰 원본을448 한 장으로
줄여 입력했다. Q3 놓침26장은 첫 presence에서 없다고 판단해 위치 단계를 건너뛰었다.
이것은 사후 진단이며 입력이나 특정 프롬프트가 원인이라는 확증은 아니다.
이미448 입력에서 결손/잔가지가 보이는데도 놓친 사례가 있다.

이 실험은 **기존에 분석한 dev240 전체에서 원인을 나누는 탐색 실험**이다.
새 holdout이나 gold 성능 확증으로 부르지 않는다. Q1·원래 QA·gold·과거 결과를 유지한다.
기존 이미지 정답으로 Q2 종류/Q3 유무를 자동 평가한다. 양성 위치 정답이 없으므로
9구역의 실제 위치 정확도는 항상 `null`이다.

## 고정된 비교

| 정책 | normal ref 예산 | target 예산 | Q2 / Q3 문구 |
|---|---:|---:|---|
| cached_baseline | 3 | 3 | 저장된 직접 Q2 / 신규 직접 Q3 |
| control | 3 | 3 | 기존 fresh 수정 정의 / legacy 유무·위치 |
| zoom_only | 3 | 6 | control과 정확히 같은 문구 |
| prompt_only | 3 | 3 | 짧은 결함 증거 정의·질문 |
| zoom_prompt | 3 | 6 | prompt_only와 정확히 같은 문구 |

Q1은 다섯 정책에서 **기존 retrieved-diffdate baseline 답 그대로**, 신규 추론0회다.
Q2는 두 문구 조건 모두 자기 이전 답 없는 fresh 질문이다. 짧은 정의는 금속 cap이 남아도
유리가 빠질 수 있다는 점과 target 내부 반복 구조를 확인하고, 참고 사진과의 단순 개수 차이를
결손으로 보지 않도록 설명한다. 현재 정의에도 유리 결손 설명은 이미 있었으므로,
새 단서를 처음 제공한 실험이라고 해석하지 않는다.

Q3의 선택지·글자 순서·분기 규칙은 네 조건 모두 동일하다.
유무를 먼저 묻고 긍정일 때만 원래 전체 프레임9구역의 위치를 묻는다.
문구 조건은 결손의 틈/노출 구조도 결함 영역임을 명시한다.
어떤 문항도 다른 문항 답이나 정답 라벨을 받지 않는다.
다른 유무 응답 때문에 실행되는 위치 호출 수가 달라질 수 있다. 경로 알고리즘은 같고,
실제 호출 수·시간을 정책별로 보고한다. 무조건 양성 채택/OR 결합은 하지 않는다.

`zoom_only` 대 `control`은 입력 효과, `prompt_only` 대 `control`은 문구 효과다.
둘 다 바꾼 조건도 각 단독 조건과 짝지어 비교한다. 이 실험은 정식 factorial interaction
검정이나 다중비교 보정을 수행하지 않으므로 bootstrap CI는 탐색 비교로 해석한다.

## 확대의 실제 동작

정상 ref·target 두 파일의 순서와 파일 바이트는 동일하다. 기존처럼448로 줄인 후 자르지 않고,
InternVL의 원본 동적 격자 전처리에서 target의 최대 셀 수만3→6으로 늘린다.
기존3으로는2×2를 선택할 수 없고 큰 정사각형이1장이 되지만,6이면2×2+전체 thumbnail을
선택할 수 있다. 작은 사진은 그대로일 수 있다. ref는 모든 조건에서3이다.
확대했다는 별도 문장을 zoom 조건에 넣지 않는다.

최대 셀 수는 정확한 타일 수가 아니다. 종횡비·원본 크기로 격자를 고르고, 여러 셀이면
전체 thumbnail이 추가된다(target 최대7, ref 최대4 가능). `image_manifest.csv`에 폭·높이·
실제 전처리 타일 수·원본 SHA256을 저장한다. 모델에는 기존대로 두 image marker와
각 이미지의 `num_patches_list`를 넘긴다. greedy/unquantized 모델과 생성 설정은 유지한다.

## 입력·출력·재개

자동 입력 경로:

- `MyDrive/lvlm_insplad_qa/gold_v2_dev/`의 FROZEN QA, `refs/retrieved_refs.csv`.
- `MyDrive/lvlm_insplad_results/gold_v2_dev_baselines/predictions/internvl3_5_8b/normal_ref_retrieved_diffdate.jsonl`.
- 두 원본 ZIP: 이름·공식 파일 크기로 Drive 기본 경로를 확인하고 없으면 검색.

QA가 dev가 아니거나 기존 baseline에 오류/무효/중복/잘못된 모델·조건이 있으면 추론 전에 멈춘다.
기존 dev baseline 형식에는 참고 파일의 해시가 없으므로 과거 참고 바이트까지 소급 증명할 수는
없다. 이 한계를 metadata에 기록하고, 현재 네 조건은 같은 검증된 참고 파일을 사용한다.

출력은 새 `MyDrive/lvlm_insplad_results/gold_v2_q23_zoom_prompt_dev_v1/`뿐이다.
다른 실험 폴더·원본 QA·baseline·참고 파일은 읽기만 한다. 신규 폴더에는 baseline 원문과
ref mapping 사본, QA, 프롬프트, 원본 이미지 해시, 타일 수, raw steps, 환경·코드 SHA,
정책별 결과/호출 비용/CI/실패 상태를 저장한다. 기존 폴더가 인식 불가 상태면 덮어쓰지 않는다.

20호출마다 checkpoint한다. 추론 셀을 재실행하면 같은 코드·QA·baseline·이미지·설정에
묶인 성공 응답만 재사용하고 실패/무효만 재시도한다. 중단 시 방문하지 않은 cache도 보존한다.
모델 로딩 실패는 한 번만 시도하고 나머지를 실패로 기록한다. 특정 입력 조건에서 메모리 부족이면
해당 view의 나머지를 실패로 표시한다. 예산을 자동으로 바꿔 비교 조건을 훼손하지 않는다.
재실행할 때 실패 호출을 다시 시도하며 코드/입력이 바뀌면 새 `RUN_FOLDER`가 필요하다.

실패/무효/누락이 있는 정책은 leaderboard·지표·CI 비교에서 제외하고 raw 답과 status만 남긴다.
모든 정책이 성공해야 `complete=true`, CLI 성공 종료다. 성공/frozen 설정 파일을 만들거나
gold 정책을 자동 선택하지 않는다. 노트북은 전체 성공 확인 후 표와 ZIP을 표시한다.

신규 호출은240장에 **2,160~3,120회**. 확대는 토큰·GPU 메모리·시간을 추가 사용한다.
실제 전체 소요 시간과 모델 성능은 Colab 실행 전에는 알 수 없다.

## 읽을 결과

- `leaderboard.csv`: Q1 지표 동일 확인, Q2 macro-F1/BAcc/Joint/FDR, Q3 유무BAcc·정상 오탐·결함 놓침, 호출 수/시간.
- `q2_by_defect.csv`: 특히 missing-cap 정답 수·분모.
- `q3_by_defect.csv`: 부품 누락/녹/새집별 놓친 수와 정상 오탐 수.
- `metrics_by_asset.csv`, `metrics_asset_macro.csv`: 자산별·자산 평균 지표와 지원 수.
- `significance.csv`: 사전 고정한 paired bootstrap 비교. 정상 오탐과 결함 놓침의 trade-off를 같이 본다.
- `image_manifest.csv`: 실제 확대가 적용됐는지 확인. 항상없음/항상영역 대조군도 제공.

마지막 셀의 ZIP: **`dev_q23_zoom_prompt_results.zip`**.
기능 제작·CPU 검증과 실제 GPU 효과 검증을 구분한다. 후자는 사용자가 이 노트북을 실행해야 한다.

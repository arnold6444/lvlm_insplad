# 이전 답 없는 누락 유무 진단

파일: **lvlm_missing_presence_independent_dev_colab.ipynb**.
[Colab 열기](https://colab.research.google.com/github/arnold6444/lvlm_insplad/blob/main/notebooks/lvlm_missing_presence_independent_dev_colab.ipynb)

**L4/A100 → 기본값 유지 → 모두 실행 → Drive 연결 허용.** 실제 표시와 확대 후보를 내장했다. 추가 표시·이전 코랩 재실행·새 토큰 입력은 불필요하다. 원본 ZIP 두 개를 자동 검색한다. 결과는 **missing_presence_independent_dev20_results.zip**, 별도 폴더 MyDrive/lvlm_insplad_results/missing_presence_independent_dev20_v1에 저장한다.

## 경고 조사 결과와 실행 전 검사

기존 모델은 공식 config상 internvl_chat/Qwen3이며 tokenizer_class는 Qwen2Tokenizer다. Transformers5.14의 Mistral 검사 경로는 로컬 config에 transformers_version이 없을 때 이 모델에서도 경고를 냈다. **경고 자체가 원본 Qwen 토큰화가 손상됐다는 증거는 아니었다.**

실제 CPU에서 이전과 같은 AutoTokenizer(...,trust_remote_code=True,use_fast=False)를 실행해 경고를 재현했다. 모델 revision과 공식 토크나이저 보조 파일까지7개 SHA를 확인하고, 공식 tokenizer.json을 직접 연 backend와 비교했다. 전체 backend·79개 문장×특수토큰 추가 여부2가지=158개 인코딩·이미지/대화 특수토큰이 모두 일치했다. Mistral 수정을 강제로 켠 CPU 대조에서는1개 문장의 ID가 바뀌었고 검사가 거부했다. 따라서 원본 tokenizer를 유지하며 fix_mistral_regex=True를 적용하지 않는다.

이 조사는 모델 추론 없이 토크나이저만 로드한 CPU 검사다. 과거 GPU 결과가 토크나이저 경고 때문에 실패했다고 판단하지 않는다. 검사 문장에는 이번 실제 질문과 JSON·이미지 토큰·숫자·대소문자·다국어가 포함된다.

코랩도 모델 가중치 다운로드 전에 같은 검사를 수행한다. TOKENIZER GATE PASS가 있어야 진행하며, 실제 모델이 로드한 tokenizer도 다시 검사한다. 원본 경고는 화면과 기록에 남는다. **경고가 다시 보여도 원본 일치 검사가 통과하면 진행한다.** 불일치면 중단하고 진단 ZIP을 내보낸다. 경고 필터나 로그 억제는 넣지 않았다. FlashAttention 미설치·torch_dtype deprecation은 원래 설정의 경고로 보존하며 연산/정밀도를 바꾸지 않는다.

근거: [고정 revision 공식 tokenizer 설정](https://huggingface.co/OpenGVLab/InternVL3_5-8B/blob/9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79/tokenizer_config.json), [공식 모델 config](https://huggingface.co/OpenGVLab/InternVL3_5-8B/blob/9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79/config.json), [Transformers5.14 검사 소스](https://github.com/huggingface/transformers/blob/v5.14.0/src/transformers/tokenization_utils_tokenizers.py). 실제 CPU 수치·해시·원문 경고는 준비 보고서 verification.json에 보존했다.

## 이번 질문과 비교

이전 PR66과 같은 위치 정답20장·세 확대 조건을 쓴다. 기존 후보 칸/정답 칸/정답 박스+25% 여백, 정상 참조+원본+확대판3이미지,3/3/2타일·greedy80토큰,조건당1회,최대60회다. 모델도 이전 pinned native InternVL3.5-8B이며 양자화하지 않는다.

질문은 필요한 유리 원판/스커트/절연체 부품이 빠지거나 깨져 반복 구조가 끊기고 금속 몸체가 드러나는지만 판단하게 한다. 기존 정의를 재사용하며, 정상 시점 차이나 다른 부품에 가려진 것만으로 누락을 판단하지 않게 한다. **이전 답·대화 history·위치 질문·정답 칸 번호·조건 이름을 넣지 않는다.** 세 조건에 완전히 같은 질문을 쓰며 출력은 present=true/false JSON 하나다.

Q1/Q2/gold/재학습/DINO는 실행하지 않는다. 원래 양성dev20에서 누락 유무 질문만 독립시킨 진단이다. 정답 crop을 쓰는 조건이 있고 정상 사진 지원0이므로 위치 정확도·정상 FPR·최종 Agent·독립 holdout 성능이 아니다.

이전 joint Q3 결과와 비교할 수 있지만, 질문 목적과 이전 답 문구를 함께 바꿨으므로 어떤 변경이 효과의 원인인지는 분리할 수 없다. 이번에는 누락 유무 회복/빈도/무효를 진단한다.

## 출력·재개

calls.json,predictions.json,crop_manifest.json,crops/,dataset_snapshot.json,metrics_*.json,leaderboard.csv,paired_diagnostic.csv,run_metadata.json,run_status.json을 저장한다. 모델 ID/코드·입력 hash·GPU·소프트웨어·새 호출/재사용 수를 남긴다. 원래 Q3 답은 previous_joint_q3_descriptive.json에만 저장하며 질문에는 넣지 않는다.

tokenizer_audit.json은 실행 전 검사,tokenizer_loaded_audit.json은 실제 모델 tokenizer 검사다. ZIP에는 사전 검사 결과·소스·raw/crop이 포함되고 원본 데이터 사진은 제외된다.

같은 입력/코드/모델/폴더의 성공·무효 답은 재사용한다. 무효는 FN이고 정답이 나올 때까지 재질문하지 않는다. 실행 오류만 이력을 보존해 재시도한다. 오류가 있는 조건은 비교 제외한다. 원본·요청·파싱·crop 조작은 중단한다.

기능·CPU 검증은 완료했지만 새 실제 GPU 효과는 실행 전이다. 성공 시 RETURN_CODE0/complete=true/60개 기록을 확인하고 ZIP으로 실제 결과를 감사한다.

# 누락 유무 독립 질문 코랩 준비와 토크나이저 경고 조사

요청대로 경고를 조사하고, 이전 답 없이 누락 유무만 묻는 코랩을 만들었다. 실제 모델 효과는 아직 실행 전이다.

파일: **lvlm_missing_presence_independent_dev_colab.ipynb**. L4/A100/기본값/모두 실행. 결과 **missing_presence_independent_dev20_results.zip**.

## 조사에서 확인한 것

- 공식 고정 모델은 internvl_chat/Qwen3,tokenizer_class Qwen2Tokenizer다. config의 transformers_version이 없고,Transformers5.14 로컬 Mistral 검사 분기로 경고가 재현됐다.
- CPU 전용 격리 venv(Transformers5.14.0/tokenizers0.22.2,PyTorch 모델 없음)에서 공식7파일SHA·전체 backend·79문장/158인코딩·이미지 및 대화 특수토큰 일치를 실제 확인했다. 질문은 이번 실제 프롬프트다. 경고 원문을 보존했고 필터를 적용하지 않았다.
- fix_mistral_regex=True를 강제한 별도 CPU tokenizer는1개 검사문장 ID를 바꿨고 gate가 pre_tokenizer 불일치로 거부했다. 모델의 원래 토큰화를 유지한다. 이것이 과거 결과를 개선한다는 주장은 하지 않는다.
- 근거와 공식 URL은 docs/missing_presence_independent_dev.md, 실제 해시/CPU 원문 경고는 verification.json에 있다. 원본 tokenizer 다운로드와 소스 조사 자료는 작업 clone의 tokenizer_evidence에 보존하되 모델 파일은 커밋하지 않는다.

## 새 실험의 차이

PR66 실제 ZIP e5baeefd1020의 같은20장과 세 crop 조건을 동결했다. JSON present bool만 요구하며 이전 답/위치 질문/history/GT 번호를 모델 입력에서 제거했다. 누락 정의는 기존 정의를 재사용한다. 원본+정상 참조+확대판,3/3/2타일·greedy80·최대60회는 유지한다.

같은 양성dev20·oracle조건이며 정상 FPR/위치 정확도/실제 Agent/새holdout 점수가 아니다. task와 이전 답 문구의 동시 변경이라 원인 하나만의 효과로 보고하지 않는다. Q1Q2goldtrainingDINO0이며 로컬 모델 추론0이다.

## 실제 검증

- 관련47개 테스트 통과. strict bool parser,이전답·위치질문 없음,동일프롬프트,무효FN 유지/재질문 없음,실행오류만 재시도/이력,중단/재개,요청·파싱·crop·입력·코드 조작 거부,tokenizer거부 시 모델 로드0 확인.
- 정확한 내장 소스를 추출해 새 테스트22개 통과. 내장33파일과 코드4셀 검사.
- 실제20target+16reference SHA/크기/EXIF,60crop/84panel 좌표·확대 픽셀 재구성 검증. 모델 대신 false를 반환하는 fixture60회와 재개0추가호출/0로드. 실제 모델 효과로 혼동하지 않는다.
- 마지막ZIP셀도 가짜다운로드로 실행해 raw60/source 보존·원본사진 제외 확인. 사전 gate 실패를 주입해 모델가중치 다운로드/runner제출0,ret98·실패receipt ZIP 보존을 확인했다.
- 원본 모델 weight나 실제 GPU forward는 수행하지 않았다. 실제 모델 tokenizer 재검사는 사용자 GPU 실행 시 수행한다.

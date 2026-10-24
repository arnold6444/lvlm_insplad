# InsPLAD-QA 논문 초안 (2026-10-23)

| 파일 | 내용 |
|---|---|
| `main_en.tex` | 영어판 본문 |
| `main_ko.tex` | 한국어판 본문 (kotex) |
| `tables/*.tex` | 두 판이 같이 쓰는 표 숫자. 각 파일 첫 줄 주석에 출처 결과 파일이 있음 |
| `refs.bib` | 참고문헌. `TODO-VERIFY` 항목은 제출 전 저자·학회·연도 확인 필요 |

## Word 파일 (docx)
- `InsPLAD_QA_paper_en.docx`(10쪽), `InsPLAD_QA_paper_ko.docx`(12쪽): 위 LaTeX 원고를 그대로 변환한 Word판이다. 표·인용 번호·참고문헌을 포함한다.
- 원고를 고친 뒤 다시 만들려면 `tex2docx.js`를 쓴다(Node.js + npm `docx@9` 필요).
  ```
  node tex2docx.js main_en.tex InsPLAD_QA_paper_en.docx en
  node tex2docx.js main_ko.tex InsPLAD_QA_paper_ko.docx ko
  ```
- 2026-10-24 Word로 PDF를 만들어 쪽별로 확인했고, OOXML 형식 검사도 통과했다. 원본은 LaTeX이므로 내용은 LaTeX에서 고치고 docx를 다시 만든다.

## 컴파일 (Overleaf 권장)
1. 이 폴더 전체를 ZIP으로 묶어 Overleaf → New Project → Upload Project.
2. Menu → Main document를 `main_en.tex` 또는 `main_ko.tex`로 지정.
3. Compiler는 **pdfLaTeX** 그대로 둠. 한국어판은 Overleaf TeX Live에 포함된 ko.TeX(kotex)를 사용.

로컬에는 LaTeX가 없어 컴파일 확인은 하지 못했다. 대신 괄호·환경 짝, `\ref`/`\label`, `\cite`/`refs.bib`, `\input` 파일, 표 열 개수를 정적 검사했다.

## 채움 상태
- 5.5절 / 표 `tab:matched`는 실행 `902a9f8a1752`의 결과로 채웠다(감사: `reports/gold_matched_call_evaluation_2026-10-23/`). 미리 정한 규칙에 따라 "CI가 0을 제외 → 구조의 효과" 문장을 남겼다.
- 남은 것: 저자·소속·제목 확정, Overleaf 컴파일 확인, InsPLAD 논문의 권·쪽 확인(DOI는 확인됨).
- 참고문헌은 2026-10-23 웹에서 확인했다. InsPLAD(DOI), Dinomaly(저자 6명·쪽수), InternVL3.5(arXiv 2508.18265)를 고쳤고, Qwen3.5는 기술보고서가 없어 모델 카드로 인용한다.

## 결과 파일 (GitHub)
- 호출 수 대조 결과: `results/latest/gold_v2_matched_call/` (리더보드, 유의성, 호출 수, 문항별 답)
- 감사 보고서: `reports/gold_matched_call_evaluation_2026-10-23/` (`audit_actual.py`, `902a9f8a1752/analysis.md`, `audit_summary.json`)
- "에이전트 − 직접" CI 재계산 스크립트: `compute_agent_vs_direct_ci.py` (인자: 코드 폴더, gold 실행 원시 폴더)

## 수치 출처
- 기준선·조건 비교: GitHub `results/latest/gold_v2_baseline`, `gold_v2_consistency_metrics`. 행은 CSV에서 생성했고 손으로 옮기지 않았다.
- 에이전트 본표·클래스별: gold 실행 ZIP `0c577ffe3bdd` (`reports/q23_gold_evaluation_2026-10-15`).
- "에이전트 − 직접" CI: 저장된 `answers.csv`로 같은 bootstrap 함수를 CPU에서 다시 계산했다(새 추론 없음). 이미 발표된 "후보 − 직접" 행이 소수점까지 그대로 재현됨을 확인했다.
- dev·후속 실험: `reports/paper_tables/latest_results.md`, `plan.md`의 감사 기록.

## 주의
- MiniCPM-V-4.5의 기준선과 참조 3조건은 greedy로 다시 돌린 결과(실행 `1661680ed362`, 감사: `reports/gold_minicpm_greedy_evaluation_2026-10-23/`)로 바꿨다. vision expert 20구성 평균 행만 샘플링 결과이고 표에 †로 표시했다. 비교는 부록 표 `tab:minicpm`에 있다.
- 한국어판 표의 행 이름(지표·종류 이름)은 공유 표 파일이라 영어로 남아 있다.

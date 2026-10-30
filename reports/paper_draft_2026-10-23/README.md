# InsPLAD-QA 논문 초안 (최종, 2026-10-27 팀 피드백 반영)

| 파일 | 내용 |
|---|---|
| `main_en.tex` | 영어판 본문 |
| `main_ko.tex` | 한국어판 본문 (kotex) |
| `tables/*.tex` | 두 판이 같이 쓰는 표 숫자. 각 파일 첫 줄 주석에 출처 결과 파일이 있음 |
| `figures/fig_*_{en,ko}.png` | 그림 4개(벤치마크 흐름, QA 예시, 결함 참조 예시, 에이전트 구조)의 영·한판 |
| `make_figures.py` | 위 그림을 결과 CSV와 원본 이미지에서 다시 그리는 스크립트 |
| `refs.bib` | 참고문헌. 2026-10-24 웹·독립 검증으로 확인, 2026-10-27 추가 4건(ReConPatch, 송전선 점검 리뷰, CPLID, STN PLAD)은 Crossref·arXiv로 확인 |
| `InsPLAD_QA_paper_en.docx`, `InsPLAD_QA_paper_ko.docx` | 위 LaTeX 원고를 그대로 변환한 Word판(그림·표·인용 번호·참고문헌 포함) |
| `VERIFICATION_2026-10-24.md` | 독립 검증 4개로 논문 주장 약 186개를 원본 결과와 대조한 보고서 |
| `tex2docx.js` | LaTeX → docx 변환기 |
| `compute_agent_vs_direct_ci.py` | "에이전트 − 직접" CI 재계산 스크립트(인자: 코드 폴더, gold 실행 원시 폴더) |

## 그림 다시 그리기
답과 참조는 `results/latest/` 결과 파일에서 읽고, 이미지는 원본 ZIP에서 꺼낸다(Python + matplotlib, Pillow). InsPLAD 이미지는 CC BY-NC 3.0이라 그림 캡션에 출처를 적었다.
```
python make_figures.py --sup-zip <supervised_fault_classification.zip>
```

## Word 파일 다시 만들기
원고는 LaTeX가 원본이다. 내용은 LaTeX에서 고치고 docx를 다시 만든다(Node.js + npm `docx@9` 필요). 그림은 PNG만 지원한다.
```
node tex2docx.js main_en.tex InsPLAD_QA_paper_en.docx en
node tex2docx.js main_ko.tex InsPLAD_QA_paper_ko.docx ko
```

## 컴파일 (Overleaf 권장)
1. 이 폴더 전체를 ZIP으로 묶어 Overleaf → New Project → Upload Project.
2. Menu → Main document를 `main_en.tex` 또는 `main_ko.tex`로 지정.
3. Compiler는 **pdfLaTeX** 그대로 둔다. 한국어판은 Overleaf TeX Live에 포함된 ko.TeX(kotex)를 쓴다.

로컬에는 LaTeX가 없어 컴파일 확인은 하지 못했다. 대신 괄호·환경 짝, `\ref`/`\label`, `\cite`/`refs.bib`, `\input` 파일, 표 열 개수를 정적 검사했다.

## 남은 일
- Overleaf 컴파일 확인
- 제목·저자·소속 확정
- InsPLAD 논문의 권·쪽 확인(DOI는 확인됨)

## 수치 출처
- 기준선·조건 비교: `results/latest/gold_v2_baseline`, `gold_v2_consistency_metrics`. 행은 CSV에서 생성했고 손으로 옮기지 않았다.
- MiniCPM-V-4.5: 기준선과 참조 3조건은 greedy 재실행 결과다(`results/latest/gold_v2_minicpm_greedy`, 감사 `reports/gold_minicpm_greedy_evaluation_2026-10-23/`). vision expert 20구성 평균 행만 샘플링 결과이고, 표에 †로 표시했다. 비교는 부록 표 `tab:minicpm`에 있다.
- 에이전트 본표·클래스별: gold 실행 `0c577ffe3bdd`(요약 `reports/q23_gold_evaluation_2026-10-15/0c577ffe3bdd/analysis.md`). 같은 세 정책은 `results/latest/gold_v2_matched_call`에도 들어 있다.
- 호출 수 대조: `results/latest/gold_v2_matched_call`(감사 `reports/gold_matched_call_evaluation_2026-10-23/`).
- Q3 위치 적중(에이전트 표·CI 표의 위치 행): `results/latest/gold_v2_q3_position`. 위 저장 답을 위치 박스 133장과 대조한 사후 채점이고 새 추론은 없다(`scripts/score_gold_v2_q3_position.py`).
- 조건 표의 Q1 BAcc: 각 리더보드의 정확도·재현율에서 계산했다(계산식은 `tables/conditions_rows.tex` 머리 주석).
- dev·후속 실험: `reports/paper_tables/latest_results.md`와 각 실험 폴더의 `analysis.md`.
- 원시 호출 기록과 Colab 결과 ZIP은 git에 없다. SHA-256과 함께 보관하며 요청하면 제공한다.

## 참고
- 한국어판 표의 행 이름(지표·종류 이름)은 공유 표 파일이라 영어로 남아 있다.

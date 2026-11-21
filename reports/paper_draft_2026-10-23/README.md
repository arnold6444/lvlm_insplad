# InsPLAD-QA 논문 초안 (최종, 2026-10-27 팀 피드백 반영)

| 파일 | 내용 |
|---|---|
| `main_en.tex` | 영어판 본문 |
| `main_ko.tex` | 한국어판 본문 (kotex) |
| `tables/*.tex` | 두 판이 같이 쓰는 표 숫자. 각 파일 첫 줄 주석에 출처 결과 파일이 있음 |
| `figures/fig_*_{en,ko}.png` | 그림 4개(벤치마크 흐름, QA 예시, 결함 참조 예시, 에이전트 구조)의 영·한판 |
| `make_figures.py` | 위 그림을 결과 CSV와 원본 이미지에서 다시 그리는 스크립트 |
| `refs.bib` | 참고문헌. 2026-10-24 웹·독립 검증으로 확인, 2026-10-27 추가 4건(ReConPatch, 송전선 점검 리뷰, CPLID, STN PLAD)은 Crossref·arXiv로 확인, 2026-11-03 추가 7건(Power-LLaVA, PowerGPT, ElecVQA-Bench, MVTec LOCO, POPE, VLMs are blind, MLLMs know where to look)은 arXiv·Crossref·OpenAlex·학회 페이지로 확인. 6절의 ElecVQA 캡 누락 재현율 69.85%는 그 논문 arXiv v1 표 17 값 |
| `InsPLAD_QA_paper_en.docx`, `InsPLAD_QA_paper_ko.docx` | 위 LaTeX 원고를 그대로 변환한 Word판(그림·표·인용 번호·참고문헌 포함) |
| `VERIFICATION_2026-10-24.md` | 독립 검증 4개로 논문 주장 약 186개를 원본 결과와 대조한 보고서 |
| `tex2docx.js` | LaTeX → docx 변환기 |
| `compute_agent_vs_direct_ci.py` | "에이전트 − 직접" CI 재계산 스크립트(인자: 코드 폴더, gold 실행 원시 폴더) |
| `compute_sft_gold_differences.py` | 소량 미세조정 절의 모델 간 차이 CI 계산 스크립트(입력: `results/latest/agent_sft_gold_eval`의 이미지별 결과) |

## 2026-11-21: 소량 미세조정 절에 gold 결과 반영
아래 11/17 초안의 **[작성 예정]** 부분을 채웠다. 11/17 항목 중 "gold는 쓰지 않았다", "초록·서론·논의·결론은 고치지 않았다", "Word 파일은 반영하지 않았다"는 이제 맞지 않는다.
- 넣은 것: 사람 박스 2차 학습(660장)의 dev240 결과(표 `tab:sft` 아래 두 행), gold 채점 절차, gold 표 `tab:sftgold`(`tables/sft_gold_rows.tex`), 결함 유무·종류, 작은 분류기 비교, 결함 위치, 확대 영역, 말할 수 있는 것과 없는 것.
- 같이 고친 곳: 초록(후속 실험 두 문장), 서론(기여 넷째 항목), 6절의 "라벨이 달린 예시" 항목, 논의의 "부재형 결함" 항목, 결론 끝, 재현성 부록.
- 수치 출처: `results/latest/agent_sft_gold_eval/`(Colab 결과 ZIP `agent_sft_sft_v2_human_gold_results.zip`, sha256 `507ee879bbe1…`, L4, 소스 `e548776`). 주 범위는 학습 사진과 같은 드론 사진 20장을 뺀 gold 226장이다. 채점 절차는 실행 전에 고정했다(`docs/agent_sft_gold_eval.md`, 브랜치 `feat/sft-gold-eval`, 아직 main에 없음).
- 모델 사이의 차이(2차 대 학습 전·1차·분류기)의 CI: `compute_sft_gold_differences.py`가 이미지별 결과에서 계산해 `results/latest/agent_sft_gold_eval/model_differences.csv`에 쓴다. 모델 호출은 없다.
- 검사: 표 두 개의 새 행과 본문 숫자 38개를 결과 CSV에서 다시 계산해 대조했고 모두 일치했다. 괄호·환경 짝, `\ref`/`\label`, `\cite`, `\input`, 표 열 개수 정적 검사도 통과했다. 컴파일은 하지 않았다.
- Word 파일 두 개를 다시 만들었다(표 14개, 그림 4개, 인용 28개).

## 2026-11-17 추가: 소량 미세조정 절 (초안, 위 11/21 항목으로 갱신됨)
사용자 결정으로 LoRA 미세조정 결과를 이 논문에 넣기로 했다. `main_ko.tex`, `main_en.tex`의 "부품 누락" 절 뒤에 새 절(`sec:sft`, 표 `tab:sft`)을 넣었다.
- 지금 들어 있는 것: Qwen3.5-4B LoRA 1차 학습(600장, 자동 박스)의 dev240 결과뿐이다. gold는 쓰지 않았다.
- 표 숫자: `tables/sft_dev_rows.tex`. 출처는 Colab 결과 ZIP `agent_sft_sft_v1_results.zip`(sha256 `bf2ddefc0352…`)의 두 실행 `leaderboard.csv`, `q2_by_defect.csv`, `significance.csv`이고 요약은 main의 `docs/agent_sft_lora.md`다. 이 결과 파일은 아직 `results/latest/`에 없다.
- 절 끝의 **[작성 예정]** 표시가 남은 일이다: 사람 박스 2차 학습(660장)의 dev240 결과와 gold 채점. gold 246장 중 2차 학습 사진과 같은 드론 사진(촬영일 + DJI 번호)에서 나온 20장(정상 6, 결함 14)을 빼면 226장이 남는다(위치 박스가 있는 133장 중 119장).
- 초록, 서론, 논의, 결론은 고치지 않았다. gold 결과가 나온 뒤 함께 고친다.
- Word 파일 두 개는 이 절을 반영하지 않았다(다시 만들지 않음).
- 참고문헌 2건 추가: LoRA, AgentIAD. arXiv API로 제목과 저자를 확인했다. LoRA의 학회판은 확인하지 못해 arXiv로 인용했다.
- 검사: 괄호·환경 짝, `\ref`/`\label`, `\cite`/`refs.bib`, `\input` 파일, 표 열 개수를 정적 검사했다. 컴파일은 하지 않았다.

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

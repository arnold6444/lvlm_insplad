# 공개 부품 검출기 가능성 검사 — 실제 Colab 결과

**계산은 정상 완료했지만, 이 설정을 부품 개수 증거로 쓰는 최소 조건을 통과하지 못했다. 확대나 LVLM 연결을 하지 않는다.**

고정된 기존 design40 중 glass-insulator 전부8장과 각 기존 정상 참고사진을 사용했다. 정상 참고 하나가 공유되어 고유 이미지는15장(대상8·정상 ref7)이다. 정답을 보고 잘되는 대상을 고르지 않았다. 대상의 정상/누락 라벨은 로컬 평가 파일에만 남겼으며, 배포된 입력에는 이미지 ID·역할·원본 member·SHA256과 pair 연결만 있다.

## 직접 실행·회수·검산

- 기존 Dot cloud computer 안 브라우저의 같은 Colab NVIDIA L4. Grounding DINO tiny 고정 revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`, float32, torch2.11.0+cu130/Transformers5.14.0. 학습0, 새 로그인/권한0, 신규 LVLM/Q1 호출0, 기존 QA 답 변경0, gold0.
- 고정 개념 `glass insulator disc.`와 `insulator.` 각각 한 번, **30/30 detector forward**, 오류0·차단0·cache재사용0. 원시900 query/call의 box·score, 합27,000개 제안을 남겼다. 이 개수는 검출 객체 수가 아니다.
- 원시 latency 합25.635초. 모델 로딩을 포함하며 설치/인증/파일 전달까지 포함한 전체 왕복 시간은 아니다.
- 소스 ZIP SHA256 `bcd9db6c638d5f1f3796424a8fa38d37e421490220ca91fc478f48ee93b98e5b`. 원격·로컬 결과 ZIP SHA256 `273b655e8d9efeebe1fbd311c5443cf61efd564844002f9a8b72ffed5b8d5597` 일치.
- 독립 CPU audit에서 소스/원본 manifest 바이트·실행 identity·원본15이미지 SHA·30 raw ID/개념·900-query 유효 범위·4개 threshold×15이미지의 NMS/개수/geometry gate를 재계산해 저장 결과와 일치했다. manifest 출력 사본의 줄바꿈 재직렬화는 구조 동등성으로, 원본 입력 바이트는 별도 소스 ZIP의 해시로 확인했다.
- 브라우저 화면 캡처가 간헐적으로 timeout 났지만 동일 뷰어 연결을 다시 잡아30/30 저장을 확인했다. 재실행하지 않았다. 이후 ZIP 원격 SHA와 로컬 회수 파일을 대조했다. 런타임 오류와 화면 캡처 오류를 구분한다.

## 사전 기준과 관측

confidence .3/NMS IoU .5에서, 큰 조립체 box 안에 중심이 있고 면적이 조립체의 .6 이하인 원판 후보가2개 이상인 정상 ref가 **7장 중6장 이상**이어야 최소 geometry 검사를 통과하도록 실행 전에 고정했다. 이는 실제 원판 위치/개수 정확도가 아니라 후속 실험을 진행할 최소 조건이다.

| 정상 ref ID | disc 제안 수 | assembly 제안 수 | 작은 subpart 후보 수 |
|---|---:|---:|---:|
| pair00-normal |7|3|7|
| pair01-normal |1|1|0|
| pair02-normal |1|2|0|
| pair04-normal |1|1|0|
| pair05-normal |3|1|3|
| pair06-normal |2|1|1|
| pair07-normal |4|1|3|

기준을 통과한 정상 사진은 **3/7**로 요구6/7에 못 미친다. `complete:true`는30회 계산이 종료됐다는 뜻이며, `geometry_feasibility_gate:false`가 과학적 판단이다. 이 둘을 혼동하지 않는다.

contact sheet를 직접 확인했다. 빨강은 disc 문구의 예측 box, 청록은 insulator 문구의 예측 box다. 어떤 사진에서는 원판 부근에 여러 box가 생기지만, 다른 사진에서는 문자열 전체가 하나의 box가 되거나 금속 몸체·원판 부위가 섞여 잡힌다. 이는 정성 관찰이며 새 box/count GT를 만들거나 정확도 수치로 바꾸지 않았다. 작은 후보가2개 이상 있다는 사실만으로도 올바른 원판 수를 셌다는 뜻은 아니다.

## 결정과 한계

1. 이 고정 설정의 확대/동일 프롬프트 재시도/threshold 선택/LVLM 결합을 하지 않는다. 부품 후보 자체가 신뢰할 수준인지 먼저 보려던 검사이며 통과하지 못했다.
2. Q2 누락 recall이나 Q3 false-none 개선을 측정한 실험이 아니다. F1/CI/위치 정확도·count accuracy를 만들지 않는다. 기존 Q1/Q2/Q3 결과는 그대로다.
3. 다른 세그멘터나 부품 대응 방법 전체가 실패했다는 결론은 아니다. SAM3는 별도 접근·연락처 공유 동의가 필요해 이번에 쓰지 않았다. 무단 승인을 하거나 다른 출처에서 제한을 우회하지 않았다.
4. 근거: [Calibrated Fusion](https://arxiv.org/html/2609.05091v1)은 정상 자료로 보정한 counting 신호를 별도로 다룬다. [SAM-LAD Discussion](https://arxiv.org/html/2406.00625v4#S5)은 반복 부품의 분할/대응 한계를 기술한다. 여기서는 공개 [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-tiny)로 primitive 가능성만 검사했으며 두 논문 방법 재현이 아니다.
5. 다음 연구는 기존 정상 참조와 대상 사이의 시점·배치 차이를 정렬/대응할 수 있는지에 대한 별도 근거 검토부터 시작한다. 이전 image-only/자동 타일 확대/같은 답 재질문/두 제안 전달/이번 grounding 설정을 반복하지 않는다. 아직 새 가설·후속 GPU 실행을 고정하지 않았다. 전체 Q2/Q3 미탐 감소 연구 Goal은 미완료다.

## 보존 파일

- `component_grounding_probe_design40_glass_v1.zip`, `raw/`: 실제 Colab 원시결과.
- `component_grounding_probe_source.zip`, `input_manifest.json`: 실행 소스/입력 신원. 평가용 정답 파일은 이 배포 ZIP에 없다.
- `audit_probe.py`, `audit.json`, `independent_counts.csv`: 독립 검산과 모든 고정 임계값의 proposal 개수.
- `raw/proposals_contact.jpg`: 모델 예측만 표시한15이미지 contact sheet.
- `../../scripts/run_component_grounding_probe.py`, `../../tests/test_component_grounding_probe.py`: 단독 runner와3개 표적 테스트(모두 통과).
- `../../notebooks/lvlm_component_grounding_probe_colab.ipynb`: 정확한 소스 ZIP/입력을 담은 재현용 노트북. 포함 소스·SHA·셀 AST·추출 소스 별도 프로세스 fixture 검증 통과. 실제 GPU 호출은 동일 소스를 기존 Colab 터미널에서 실행했다.

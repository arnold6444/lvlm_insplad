# dev25 누락 위치 표시 코랩 준비·검증 — 2026-10-21

`lvlm_dev_missing_location_annotation_colab.ipynb`는 CPU에서 실행하는 사람 박스 표시 도구다. **사용자 표시 및 실제 위치 평가 결과는 아직 없다.**

기존 fixed dev240 중 missing-cap25장 전체를 원본 SHA·크기·EXIF와 함께 고정했다. PR64 오답/후보/점수로 사진을 선택하지 않으며 모델 답·후보 위치·파일명은 표시 payload에 포함하지 않는다. 반복 검토한 dev의 위치 라벨 추가이며 새 독립 holdout이라고 부르지 않는다. Gold/Q1/Q2/모델/기존 결과 변경·추론0.

원본 해상도 확대·이동, 여러 실제 누락 박스, 초안 자동 저장과 명시적 확정, 재개, 판단 불가/라벨 이견·사유, 이전 버전 이력 및 stale revision 보호를 구현했다. 확정 박스의 각 중심을 독립적으로9칸 집합으로 변환한다. 초안/미완료/판단 불가/라벨 이견은null이며 정상/빈 정답으로 변환하지 않는다. 마지막 셀은 부분·완료 상태를 구분해 ZIP을 만든다.

## 검증

- 관련51 tests 통과. 실제 내장 source capsule을 추출한 별도18 tests 통과;4code cells AST, 소스/HTML/manifest3파일 해시 검증.
- 실제 원본25장 SHA/해상도/EXIF, 25개 원본 전체 이미지data URL SHA 확인. 모델/후보가 없는 입력 manifest·UI payload 확인.
- Aside 브라우저에서 localhost mock Colab bridge와 실제 Python Session/실제25사진으로 마우스 박스, 자동 저장, reload 복원, Enter 확정/다음, 사유 없는 판단 불가 차단, 사유+unknown 확정,140%확대/선택 안내선, 두 박스, 삭제 자동 저장 및 재개를 확인했다.
- 로컬 합성 박스만 사용한 fixture에서 확정2건(박스1/판단불가1),초안1건을 저장하고 서버 Session 재생성과 partial export를 확인했다. 위치 support1이며 초안은 GT에 포함하지 않았다. **사용자 정답 라벨이 아니다.**
- 브라우저 screenshot은 CDP 캡처 timeout으로 확보하지 못했다. DOM 상태/실제 포인터 이벤트/서버가 수락한 저장 기록으로 검증했다. 실제 Google Colab bridge 연결은 사용자 실행 전이며 기존 표준 invokeFunction/JSON callback을 사용한다.
- source capsule SHA `43aabe80aa43e14f95d48f7a2640be8db97eab45f31e18fb49969431266999e2`.

사용자는 표시 화면까지 실행 후25장을 확정하고 마지막 셀로 `dev_missing_location_annotation25_results.zip`을 내보낸다. 중간에는 ‘초안 저장됨’을 확인하고 나가면 된다. 미완성 드래그/응답을 못 받은 변경까지 저장했다고 약속하지 않는다. 이후 사용자 라벨을 검산하고, 저장된 후보 위치와 사람 위치를 비교한다.

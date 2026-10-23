# 다음 위치 도구: IG 기반 부위 후보 코랩 준비

**기능 제작·CPU 검증 완료, 실제 pretrained GPU 위치 효과는 아직 실행 전이다.**

파일 **lvlm_missing_ig_location_dev_colab.ipynb**, L4/A100/기본값/모두실행.
결과 **missing_ig_location_dev240_results.zip**. 이전 코랩 재실행·추가 표시·토큰 입력은 불필요하다.

## 변경과 실험의 의미

이전 PR68 실제 결과는 누락25/25 검출·위치12/24 완전일치였다. 1순위만 출력하면 다중 정답2장을 완전히 맞힐 수 없고, 경로를 늘린3장도 위치정답을 추가하지 못했다. 이번은 **유무를 PR68 결과로 완전히 고정하고 위치 도구만 시험**한다. 유무성능·정상FPR가 같다는 것은 설계의 결과이지 새 도구의 검출 개선 증거가 아니다.

같은 동결 DINO와분류기 logit에 대해 [IG 원 논문](https://proceedings.mlr.press/v70/sundararajan17a)의 input attribution을 midpoint32단계로 계산한다. [공식 completeness 설명](https://captum.ai/api/integrated_gradients.html)을 따라 logit차이와기여도합의잔차를기록한다. Captum 추가설치 없이 직접구현했다. 가중치requires_grad=false, optimizer/재학습0이며입력에대한역전파만수행한다.

흐림/평균색두기준과양의기여도·padding제거·37×37패치·정규화최소값합의·성분peak50%/mass15%/최대3개를고정한다. 이 위치후보생성은 **새로운 자체 가설**이며 논문의 위치정확도 보장이 아니다. 성분중심을 원본9칸으로복원한다. 위치보류는누락유무를바꾸지않는다.

고정8조건은기존PR68,blur단일,mean단일,합의top1,합의multi,동일K의가림순위/중앙/해시무작위다. 전체9칸가림순위에서K개를강제로고르는대조는예전positive-agreement운영정책과구분한다. 칸을많이찍는효과와위치순위효과를분리하며GT로K/임계값을고르지않는다.

## 실제로 검증한 것

- **새21개 테스트 통과(36.51초)**: 선형함수의정확기여/완전성,3차함수적분수렴,실제HF `Dinov2Model`의작은랜덤모델로역전파·기울기유한성·가중치불변·진단수식확인. pretrained가중치다운로드나실제DINOforward는하지않았다.
- 좌표padding제거/원본복원/정확third경계/분리된복수부위/빈합의/같은칸중복/동일K대조/GT-blind/기존정상·무효보존/소스·요청·맵·특징·진단·canary조작거부/중단재개·실패재시도/적분경고비제외검증. Windows에서외부Torch모듈까지경로resolve하던과정은동일scripts/src인벤토리를유지하는빠른검사로구현하고기존hash집합과동일함을테스트했다. 기존공통모듈은수정하지않았다.
- **기존연결15개 테스트 통과(41.35초)**. 새정확내장소스에서도 **21개 테스트 통과(33.99초)**.
- 실제PR68 ZIP SHA bad35ced5b0b…/source2ddb37fc4235…/동결head확인. 원본50SHA/전체518원본·가림특징SHA·L2·점수검산후 원본50+canary가림1개만내장. 새로운GPU해석을기존특징으로대체하지않는다.
- 실제26장/240답입력연결에 **고정가짜기여도맵52개**를반환하는명시적fixture로8조건유무동결/동일K/좌표/원시맵저장확인. 이fixture의위치예측은효과근거가아니다. 실제새IG지도는없다.
- 성공fixture와canary실패fixture에서 **정확한마지막ZIP셀을실행**,원본사진제외·raw/소스/진단/오류보존확인. 성공재개는새계산0/모델로드0이다.
- Source capsule `66e7e6f5209af548428653e5c0c1a70c1eb4aa4a883f3e07704265bca1d34834`,90파일/5코드셀. `verification.json`, `notebook_build.json`, `verify_actual_inputs.py`에해시와검증범위를보존했다.

## 사용자 GPU 실행에서 확인할 것

원본/가림canary2와26장targetfeature가동결PR68특징과일치해야한다. 양성26장각각2기준×32적분으로최대1744forward/1664backward,새LVLM/Q1Q2gold/학습0이다. signed518×518맵52개·target특징·기준별logit·적분잔차·성분·좌표·예측·평가를저장한다. 잔차경고를숨기거나해당사례를지표에서빼지않는다. 실행실패조건은부적격이며paired출력을중단한다.

실제위치성능/전체518입력에대한nativeGPU역전파및메모리/기준민감도/적분수렴은아직검증전이다. IG는판별기설명이며배경을선택할수있다. 이미탐색한dev24위치GT의효과로최종gold/전체Agent/새holdout을주장하지않는다.

## 완료 조건 대조

|요구|구현|증거|
|---|---|---|
|위치만새도구|고정PR68presence+IG8조건|모든240유무불변테스트|
|학습/GT입력없음|frozen logit·입력기울기/선택기GT없음|가중치불변·라벨변경예측동일|
|다중부위와개수효과분리|성분중심최대3/동일K대조|toy다중부위·실제26입력동일K|
|정확수학·좌표·수치진단|IGmidpoint/원본좌표/completeness|선형·다항·tiny실모델/PILgeometry|
|복구·원시결과보존|per-image원자저장/캐시lock/ZIP|실패·중단·재개0·정확export셀|
|실행가능한파일/merge|self-contained Colab|source추출21·원본연결fixture,Git게시후별도merge검증|

기존dirty4/원본ZIP/연구결과를보존하며별도clone에서새파일만게시한다. 전체연구는미완료이고사용자가실제ZIP을반환하면위치효과와경고부터감사한다.

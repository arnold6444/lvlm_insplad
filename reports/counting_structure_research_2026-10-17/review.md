# 다음 구조: CountGD++ 개수 전용 사전학습 모델 검토

현재 연구 checkpoint이며 실행 완료 보고가 아니다. 이전 구성비율/Q3 출력 계약 실패와 SAM 마스크 전제 실패를 반복하지 않는다. CountGD++는 [공식 CVPR2026 코드](https://github.com/niki-amini-naieni/CountGDPlusPlus), [논문](https://arxiv.org/abs/2512.23351)에 공개된 개수 전용 가중치/자동 pseudo-exemplar 방식으로 검토한다. 일반 GroundingDINO tiny 박스 모델을 이름만 바꾸어 다시 쓰는 실험은 금지한다. 사전학습은 generic counting 데이터에서 이루어졌으며 InsPLAD 추가 학습이나 범용성 입증과 혼동하지 않는다.

읽기 검토한 공식 코드 commit: `8489f836d75eb407ae778c56cff1f72cc6f8baae`. 복사 파일과 SHA256은 source_review_identity.json에 보존. 로컬에서 해당 외부 코드를 실행하거나 설치하지 않았다.

확인한 차이/조건:

- text-only와 시각 예제 입력을 모두 지원. test_shanghai_tech.py의 get_pseudo_exemplars는 모델 box 점수에서 상위 후보를 고르는 두 단계 추론이다. InsPLAD GT box/QA답을 pseudo-exemplar로 사용하지 않는다. 단일 고정 개념과 해당 토큰 범위만 입력해 category 선택을 정답에서 읽는 평가 코드를 제거해야 한다.
- 공식 FSCD 설정은 별도 synthetic exemplars와 유료 서비스로 만든 초해상도 crop을 포함한다. 이 부분을 자동 도입하지 않는다. 수작업 box 없이 가능한 text-only→자동예제 경로를 별도로 고정할 때 원논문 완전 재현이라고 부르지 않는다.
- requirements의 torch<2.6과 현 Colab torch2.11/Transformers5.14가 다르다. 현재 로그인·마운트·라이브러리를 리셋하지 않는다. 별도 remote 폴더에 필요한 구버전 BERT/tokenizer만 격리하거나 독립 환경을 검증한다. bertwarper.py는 구 API get_extended_attention_mask/get_head_mask를 호출하므로 그대로 현재 버전 호환이라고 추측하면 안 된다.
- groundingdino_app.py는 torchvision.ops.roi_align을 사용한다. 논문 repo 전체 설치의 detectron2가 실제 최소 inference import 경로에서도 필수인지는 아직 확인하지 않았다. requirements 전체를 무작정 설치하지 않는다.
- ms_deform_attn.py에는 공식 pure-PyTorch 구현이 있지만, CUDA tensor에서는 custom C++ op를 무조건 선택한다. custom op가 없는데 GPU fallback이 된다고 주장하면 안 된다. 필요하면 명시적·기록된 backend 선택 변경과 작은 수치 동등성 fixture를 선행하고, 시간/결과에 backend 차이를 보고한다. 임의 경고 숨기기/torch.cuda false 우회는 금지한다.
- 공식 예제 load_state_dict(strict=False)를 그대로 신뢰하지 않는다. checkpoint와 모델의 missing/unexpected keys를 감사해 학습되지 않은 신경망 파라미터가 새로 들어가지 않았음을 확인한다.

다음 실행 순서: 최소 inference 의존성/가중치 출처·정확한 신원 확인→현재 Dot Colab을 보존하는 격리 호환성 검증→기존 원시 사진과 누락 형태를 대조해 검증 가능한 가설 하나 고정→입력·대조군·중단 기준·출력 계약 fixture 고정→작은 실제 Dot Colab 실험→원시 결과 회수/독립 검산. 정확한 부품 GT가 없으므로 count/point 정확도는 unknown이며, 정상사진별 부품 수가 같다고 가정하거나 감지 점을 검증된 좌표로 넘기지 않는다. 후속 Q2/Q3 판단이 유효한지 새 실험에서 확인해야 한다. Q3에 다른 문항 답을 전달하지 않는다.

아직 새 실험 가설/입력/성능은 고정·실행되지 않았다. 현재 활성 GPU 작업은 없고 Native Goal/heartbeat는 ACTIVE이며 다음 유용한 작업을 계속하도록 유지한다.

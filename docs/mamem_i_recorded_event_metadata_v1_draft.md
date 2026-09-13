# MAMEM I recorded-event metadata 후속 초안

> 후속 갱신: [scalar 관문 결과](mamem_i_scalar_adapter_v1_results.md)에서 저장250Hz를 확인했다. DIN descriptor 정의는 미확정이라 읽지 않았다. Row257의 물리적 의미를 추측하지 않으면서 공통 앞256행만 사용하는 별도 정책은 [새 후속 초안](mamem_i_256row_adapter_v1_draft.md)에서 제안한다. 아래의 ‘임의 삭제 금지’는 이런 사전 공통정책 자체를 영구 금지한다는 뜻으로 확대하지 않는다. 아래는 당시 초안이다.

2026-09-13. DRAFT_NOT_EXECUTED. [식별 관문 결과](mamem_i_timing_identifiability_v1_results.md)
뒤의 제안이며 새 EEG fit이나 만료된 probe의 재실행 명령이 아니다.

## 다음 질문과 가장 작은 확인

1. 정확한 EGI/NetStation MATLAB export 및 HydroCel256 reference/channel 설명을
   공식 자료 최대3targets/30분으로 확인한다. 지금 읽은 Cedrus 공지에 직접 연결된
   원래 ST100–EGI 설정 안내를 포함할 수 있다. 과거 실패 DOI/비공개 경로는 우회하지 않는다.
2. 실제 파일값이 필요하면 **이미 개발용으로 지정된 S001a 한 기록**에서
   samplingRate1scalar와 첫DIN event의 non-EEG descriptor 필드만 읽는 별도
   memory-bounded manifest를 먼저 만든다. 첫 row/third row의 의미를 추측해서
   추출하지 않고, field schema/type 정의와 공개 설명을 먼저 연결한다. 추가 참가자,
   전체DIN재요약/EEG파형/효능은0으로 둔다.
3. S011을S013으로 바꾸지 않는다. Adapter는 실제 배포 파일명을 사용하고 역사적인
   code alias는 별도 unresolved로 유지한다. Row257의 의미를 알아내지 못하면
   전체257행을 EEG라고 가정하거나 마지막 행만 임의 삭제하지 않는다.

## 남길 수 있는 연구 가설

`기록된 event 규칙성 M`은 물리적 자극 jitter와 동의어가 아니다. 그래도 외부에서
측정된 marker context가 support EEG 추정의 불확실성을 Q 이상으로 예측한다면,
source와 few-shot support를 결합하는 데 도움이 될 수 있다. 이 가설은 marker의
물리적 원인 식별과 별개의 예측 질문이며 현재 검증되지 않았다.

학습 후보로 진행하기 전 다음을 수식/입력권한으로 먼저 고정해야 한다.

- 알려진 class/nominal schedule에 따른 차이는 공통 정보로 분리한다. 고정순서,
  절대 session 시간, queryDIN frequency를 shortcut 입력으로 사용하지 않는다.
- M은 support에서만 관측하고 k=1에서도 실제로 template/uncertainty/source연산을
  바꾸어야 한다. 정규화 trial scalar의 one-shot 상쇄 경로는 재사용하지 않는다.
- 공통 deterministic correction+Q 대비 추가 M을 비교하며 Q2/SHAM을 동일 capacity/
  같은 support/query로 맞춘다. LearnedQ는 source-only 선택, 새 참가자 결과로튜닝금지.
- S001은 모든 후속 독립 효능 증거에서 제외한다. Acquisition 전 휴식/적응/대기·버린
  trial·센서setup을 비용에 포함한다. 문서80/100초 discrepancy는 실제timeline을
  확인하기 전 임의로 유리한 쪽을 선택하지 않는다.
- 위 전제와 channel/eventadapter가 해결되면 후보1개와 source/human fit예산·중단
  기준을 따로 동결한다. 이 문서는 그 숫자를 사후로 메우는 실행 허가증이 아니다.

기록된M을 정의할 수 없거나 공통보정/Q로 충분하면 이 후보는 종료할 수 있다.
조건부 물리해석의 실패를 metadata 전체의 무용성으로 확대하지도, 목적 달성을
위해 새 feature/threshold를 끝없이 추가하지도 않는다. Held60/외부요청/유료는별도승인.

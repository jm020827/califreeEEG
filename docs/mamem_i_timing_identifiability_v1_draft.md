# MAMEM I timing-identifiability 다음 관문 초안

2026-09-13. **DRAFT_NOT_EXECUTED**. [획득 결과](mamem_i_acquisition_v1_results.md) 뒤의
다음 루프이며 이번 single-record 예산을 늘리는 실행 기록이 아니다.

목표는 바꾸지 않는다. 다운로드 완료·첫 prefix의 국소 clock 일관성을 확인했으므로, 이제 실제 저보정
학습 전에 ‘공통 보정 후에도 외부 timing M이 추가 정보를 줄 수 있는가’를 판별한다.
현재 첫 묶음의5.93ms spread만으로 후보를 승격하지 않는다.

## 먼저 결정할 것

- 공개 배포/저자 code의 S011↔S013 불일치와257번째 row 의미를 확인한다.
  새 source 조회는 정확한 공식 설명/code의 최대4targets,30분,요청당2MiB/30초.
  PDF가 필요하면 별도 질문·읽기 범위를 먼저 등록한다. 접근 실패는 우회하지 않는다.
  실제 추가 MAT 값이나 신원 정보는 이 단계에서 읽지 않는다.
- 확인 불가하면 영향받는 ID/채널은 미확정으로 남긴다. 임의 동일인 매핑·row 삭제나
  효능 실험은 하지 않는다. filename namespace와 device-channel namespace를 구분한다.
- DIN은 support에서만 사용한다. Stimulus label을 만드는 query DIN은 predictor에서
  완전히 격리한다. 기본 clock/schedule correction은 모든 arm에 공통 제공한다.

## 후보보다 먼저 만드는 생성자료 반증 검사

최대3개 사전 고정 시나리오·각1seed·fit0으로 구현 가능한 연산 경로를 검사한다.

1. **격자만 있는 음성:** 정수 frame schedule+250Hz 기록 격자만으로 간격 spread가
   생겨도 이를 독립 physical jitter나 M 효과로 표시하지 않아야 한다.
2. **공통 보정으로 제거되는 변동:** 알려진 support stimulus timing으로 template을
   정렬했을 때 문제가 사라지면 그 개선은 공통 보정의 결과다. QM만 이 정보를
   받도록 baseline을 약하게 만들지 않는다. 이 경우 learned-M 후보는 종료할 수 있다.
3. **기록 오차 음성:** 실제 광출력/EEG는 그대로 두고 timestamp 기록만 바꾼다.
   이 조건과 실제 광학 event 변화가 관측 DIN에서 구별되지 않으면 physical jitter로
   명명하지 않는다. Timestamp 분해능/clock/event polarity의 provenance가 필요하다.

이후에도 남는 불확실성이 실제 DIN 필드로 측정된다는 근거가 있을 때만 시간/
고조파별 template uncertainty에 연결하는 조건부 경로를 정의한다. 없으면 인공적인
양성 M을 만들어 사람 후보로 주장하지 않는다. Part2의 flat member 경로는 basename+
archive 출처로 연결하되, S011↔S013 identity alias와는 별개로 다룬다.

양의 trial scalar를 정규화해 one-shot template에 곱하면 상쇄되는 구조를 다시
사용하지 않는다. 새로운 학습 후보가 필요하다면 M이 실제로 바꾸는 템플릿/우도/
source 경로와 matched Q2 capacity를 식으로 먼저 명시한다. Neural ODE를 바로 추가하거나
이전 gyro optimizer를 새 데이터에 무가설 적용하지 않는다.

## 실제 값이 더 필요해질 때의 별도 실행 계약

이번에 개발용으로 고정한 S001 외에 결과를 미리 보지 않는다. 먼저 S001의 추가
지원구간을 열 필요와 최대범위, nominal schedule 제거식, clock 오차 처리, 실패
판정을 문서로 고정한다. Query/held participant 정보는 사용하지 않는다. 이 초안은
추가 DIN 전부를 여는 명령이나 이미 승인된 학습 manifest가 아니다.

잔여 M의 기작과 identity/channel mapping이 해결되면 참가자 분리 source 학습,
동일 support/query의 Q/Q2/QM/SHAM, 동일 공통 보정, 실제 acquired-prefix cost를
가진 한 후보의 유한 fit 예산을 정한다. 효과가 없으면 종료한다. 유리한 결과가
나올 때까지 후보·threshold·관측 범위를 늘리지 않는다. S001은 독립 효능 증거에서
제외하고, held60/비공개 요청/유료 자원은 계속 별도 승인이다.

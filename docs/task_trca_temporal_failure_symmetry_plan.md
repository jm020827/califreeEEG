# C1 종료 후 제한된 S/C 대칭성 원인 진단

2026-09-09. C1 failure SHA4b57bc9bad8cd53eda24f49cbee063e32eba3729ce72d10b5d3b94de2f167c7e.
이는 [프로그램](metadata_learning_program_v1.md)의 validity 원인 감사이며 후보 재실행/새 efficacy 실험이 아니다.
24,000updates/274query-bearing요청 뒤 실패했고 전체 결과가 없어 C1 효능은 NOT_EVALUATED다.

## 실행 전에 고정하는 범위

- 새 rawEEG/nativearchive/원numericM/projection/query/옛실패 입력을 열지 않는다.
- 이번 C1의 실패/start/access/events/freeze/model JSON hash와 **source1/2.npz의 keys,s,c만** 읽는다.
  두 source 파일은 실패 전에 이미 해독한 prefix 통계의 저장본이며 새로운 사람 dataattempt가 아니다.
- Access의 마지막 위치는 pid63/interface1/N125이며 full_k3 요청 전 evaluate가 중단됐다.
  평가 loop의 첫 budget k3, 첫 positive arm ISO라는 추론을 실제 ISO 연산으로 한 번 확인한다.
- 단일 prefix key(63,1,125,3)의5bands×12classes S/C가 두 저장본에서 같은지 검산한다.
  C1 ISO의 B=C+tau I와 C2 uniform의 B=C를 각각 원래 CPU float64 두 triangular solve로 계산한다.
  두 operator 각60개, 최대120 whitened matrices만 검사한다. 원 S/C 대칭오차, C고유값/조건수,
  H의 Frobenius 대칭오차와 고정1e−12×max(norm,1)비율을 기록한다.
- Existing strict leading_projector 호출의 성공/예외만 확인한다. 새 class점수/argmax/정확도/fit/lambda 탐색0,
  symmetrization 추가·threshold 완화·학습/평가 backend 전환·실패person제외0이다.
- CPU1, 최대60초, 새출력1MiB, 생성검사 후 한 번 진단. 다른GPUprocess/held60/outreach/paid 사용0.

## 사전 해석 분기

- ISO에서 같은 예외가 재현되면 이 실패는 learnedM을 요구하지 않는 baseline 계산에서도 발생한다.
- C2 uniform의 B=C에서도 같은 guard가 실패하면 C2가 공유 미해결 결함을 가진 것으로 기록하고
  사전 eligibility 규칙으로 C2를 NOT_EVALUATED 종료한다. 실패를 M음성 증거로 만들지 않는다.
- C2 uniform이 통과하더라도 전체C2 안정성/효능을 추론하지 않는다. 공유 결함의 범위와 정확한
  guard 재현 여부를 별도 검토한 뒤 사전 명세 내 C2 자체 generatedpreflight만 진행할 수 있다.
- ISO가 재현되지 않으면 실패 arm을 아직 확정하지 않는다. 새로운 query/refit 없이 필요한 저장된
  frozen prior 재구성 범위를 별도 기록하며 자동으로 넓히지 않는다.

Query 이후 scorer의 수치 정책을 고쳐 C1을 다시 평가하는 것은 허용된 infrastructure recovery가 아니다.
이번진단에서도 old/new runtime25pins·실패산출물은 변경하지 않는다.

# C1 결과: 수치 유효성 실패, metadata 효능 미평가

> 최종 프로그램 업데이트: 이후 미리 정한 C2도 자체 수치 관문에서 실패했고 독립 감사·전체회귀를
> 마쳤다. [두 후보 종합](metadata_learning_program_v1_results.md)에 따라 이번 유한 루프는 종료한다.
> 아래는 C1 종료·진단 당시의 기록이며, C1 재실행이나 부분 효능 해석은 하지 않았다.

2026-09-09. **C1은 종료한다. 성공도, metadata 효과가 없다는 음성 효능 결론도 아니다.**
세 외부 분할의24,000학습updates와 학습 검산은 완료했으나, 최종 평가 중 필수 대조군의 수치
검사가 실패했다. 전체 비교·보정량 판정을 만들 수 없으므로 부분 정확도를 선택해 보고하지 않는다.

## 실행과 실패

- 실행commit10a05222fab8be52a36830540a3543f9dfa55783; 새 manifest SHA
  `05e6251de0463a0cf49d8ac434135d8df18544b3a640d9a6128432dd2eebdd33`.
- 생성 완료경로/cold와 전체2578tests를 통과한 뒤 UTC2026-09-08T18:13:36.279605에 시작했다.
- 실제39명/2interfaces/4windows/k3,5;30pipelines/120heads/24,000updates, 세outer의 Q-only선택 lambda=.001.
- 587.195초 뒤 `h must be symmetric within the frozen tolerance`로 종료했다.
  출력 `/home/whwovy/task-trca-temporal-source39-v1-primary1-20260909`는 전부 보존한다.
  Failure SHA `4b57bc9bad8cd53eda24f49cbee063e32eba3729ce72d10b5d3b94de2f167c7e`.
- 모델3개 동결 후274query-bearing요청(69query+69A0+68FULL3+68FULL5)이 있었다.
  사람dataattempt1회·최종query모델개봉1회를 모두 소비한 것으로 센다. 부분실패라 예산을 되돌리지 않는다.
- 평가파일8명(S4,11,22,29,32,41,44,55)까지만 저장됐다. S63은 부분계산만 있었다.
  최종result/완료cold/평가fold감사는 없다. 실험결과를 `COMPLETE`로 올리지 않는다.

## 원인 위치를 좁힌 제한 진단

[진단 전 계획](task_trca_temporal_failure_symmetry_plan.md)을 두 단계에 걸쳐 기록하고,
이번에 이미 저장한 source1/2 통계와 고정model0만 사용했다. 원rawEEG/Mprojection/nativearchive,
query/Grams/label 읽기·점수/정확도 계산·학습·대칭화추가·허용오차변경은 하지 않았다.

1. 같은 prefix S063/wet/N125(0.5초)/k3의5bands×12classes S/C가 두 저장본에서 정확히 일치했다.
   C1 ISO와 C2 uniform의 원계산은 둘 다 통과했다. 최대오차/허용오차비 .762435/.532553.
2. 같은13명 donor partition의 저장 Q/M/packet과 model0의 고정계수를 재구성했다.
   첫 실패는 **PERMUTED**, band3/class7(0부터 셈), donorS84다. Q/QM/SHAM 등은 이 지점에서 통과했다.
   이를 근거로 PERMUTED를 빼거나 QM만 성공이라고 보고하지 않는다.

| 고정 arm | H 대칭오차/허용오차 최대비 | 이 prefix의 strict projector |
|---|---:|---|
| ISO | 0.762435 | 통과 |
| Q | 0.545063 | 통과 |
| Q2 | 0.795801 | 통과 |
| QM | 0.976698 | 통과 |
| SHAM_REFIT | 0.759601 | 통과 |
| PERMUTED | **1.056530** | **거절** |
| STALE | 0.573912 | 통과 |
| MISSING | 0.545063 | 통과 |
| C2 UNIFORM, B=C | 0.532553 | 필요조건만 통과 |

원 S/C의 대칭오차는0이다. 실패행렬의 C조건수는108,654.56, 최소고유값.02301414다.
두 triangular solve 뒤 H의 비대칭 Frobenius norm은2.2176203e−12, H norm은2.0989664이므로
고정1e−12×max(norm,1) 허용치를 넘는다. 이는 **finite-precision whitening의 수치 민감성**과
일치한다. 높은 조건수가 원인에 기여한다는 해석은 수치적 추론이며 EEG 손상이나 metadata 부정효과가 아니다.
실패시 local H/arm은 별도 저장되지 않았고, 위 위치는 같은 저장prefix·계수·순서를 이용한 재현이다.

진단1 SHA`57cbb4838b836d3bdf3351f01b17b069028a7a9e18df9310417ad017519b6703`, .503초;
진단2 SHA`2f90f56236d5c27a0897dc9aec6b5bf674c4d0dcf3b34765d7bbe36c69df8ebf`, .529초.
새 generated-only7tests PASS .85초/Ruff PASS. 진단도 같은수치정책을썼고사람optimizer/query추가0이다.

## 독립 확인과 예산

읽기전용 reviewer가 failure/start/models/freeze/access/events와 원manifest/program/C1design/25codepins를
확인했다. 세model의각200steptrace, 정확fit접근2,184개와동결후query순서, model→source descriptor연결을 확인했다.
이는 **실패 provenance 감사**로서 미완료 평가에 대한 cold PASS가 아니다. Reviewer는 NPZ내용을 열지 않았다.
Root진단은해시검증된source1/2의허용필드만읽고중복일치를검산했다.
Phase2는 저장된 numeric M와 packet5도 사용했다. NPZ에서는 허용 멤버 전체를 물질화한 뒤
13개 donor 행을 선택했으므로 ‘13행만 디코딩’ 또는 ‘S/C만 읽음’으로 표현하지 않는다.

실제실험출력1,082,188KiB, 진단출력100KiB이며시도예산내다. GPUpeak는실패receipt에없어확정값으로보고하지않는다.
프로그램현재소비: 후보C1 primary1회/24,000updates/587.195초/query모델개봉1회, infrastructure recovery0.
Held60·retiredS1–S3·외부요청·유료사용0, 모든과거실패/부정결과를보존한다.

## 다음 판단

C1은닫고수치정책수리후재평가하지않는다. 이실패는metadata가학습에연결되지않았다는증거도,
도움이안된다는증거도아니다. 아직유효한전체성능·보정비용판정이없다.

C2의같은기본prefix경로는통과해 **공유실패가입증된것은아니다**. 그러나동일whitening검사에수치위험이남아있다.
독립원인검토를마친뒤고정C2자체generatedengineering만시작할수있으며, 사람C2는자체preflight와별도manifest가필요하다.
H를검사전에대칭화하거나허용오차·backend를바꿔회피하지않는다. C2자체유효성관문이실패하면NOT_EVALUATED로종료한다.
프로그램goal은여전히ACTIVE이며C1문서화만으로완료하지않는다.

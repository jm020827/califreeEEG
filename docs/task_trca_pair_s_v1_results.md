# C2 결과 — 사람 실험 전 수치 관문 실패

2026-09-09. **Metadata 효과 없음이 아니라, 효능 미평가다.** 미리 정한 C2의 자체 수치 관문이
실패했으므로 사람 학습·최종 query를 시작하지 않는다. 독립 terminal 감사는 현재 진행 중이다.

## 구현한 것과 아직 하지 않은 것

[사전 설계](task_trca_pair_s_v1_design.md)는 C1 사람 실험 전에 고정했다. C1은 채널별 분모 규제,
C2는 반복 쌍별 numerator S 기여도라는 서로 다른 삽입 위치다. C2는 균일 템플릿과 C를 유지한다.

- 새 Q15/M2, k3의3쌍/k5의10쌍, 정확한 prefix와 STALE mask, 가중 scaler를 구현했다.
- 양의 쌍 가중치→S→C-normalized projector의 미분 연산을 구현했다. 균일 logits에서 S0는
  bit-exact이며 contrast gradient가 살아 있다. C/B와 고정 수치 정책은 바꾸지 않았다.
- 신규190개와 관련 기존127개, 총317tests PASS2.27초. 생성 gate CLI의 toy7tests PASS .90초.
  Unit test는 정확한 수식/미분/경계 증거이지 실제39명 전체 학습이나 안정성 증거가 아니다.
- 아직 실제4head/nested learner·C2 role reader/효능 cold·C2 사람 manifest는 없다. 선행 수치
  관문에서 멈췄으므로 이 미구현 항목을 완료했다고 하지 않는다. C2 optimizer updates0이다.

## 미리 정한 첫 수치 관문

[실행 전 계약](task_trca_pair_s_v1_build.md)에 seed20260909, 고정 Gaussian 공유신호+반복별noise,
12classes/5bands/8channels/N125/k3,5를 명세했다. 원 C의 방향을 유지하는 congruence로 조건수
1/1000/100000/120000의 전체 스펙트럼·trace·draw순서를 고정했다. 이는 앞서 관찰한 높은
조건수 범위를 시험하는 수치 검사이며, 생물학적 모델·새 사람 표본·성능 탐색이 아니다.
합동변환은 정확한 산술에서 generalized spectrum을 유지하므로 별도의 top-gap 범위를 넓히지도 않는다.

실행commit48407d77cb385dd56055d7ceb49a4f3f81379c80, 출력
`/home/whwovy/task-trca-pair-s-conditioning-Vqli4o/primary1`:

- Start SHA `6cc55bddc2f512e7ea7f83eea7052a096f223b30022ee5ad94b937b6388cd70b`.
- Result SHA `98ea5607b72d47174c2f0e3c1569bf9a872bca7230c160f29554b8b9cd841ee2`.
- Generated arrays SHA `01ad4233bba861f32f0a4ca5f68e1f1af883ffc5d51c55fa1b04cbb423421211`.
- 16개 고정 행을 모두 기록, CPU1에서 .639207초, 종료코드2. 전체22,068KiB; GPU·사람·query·학습0.

다음 값은 H 대칭오차/허용오차의 최대 비율이며, **1을 넘으면 거절**이다. Scalar/batch가 같다.

| 목표 C 조건수 | k3 균일 | k3 고정 비균일 | k5 균일 | k5 고정 비균일 |
|---|---:|---:|---:|---:|
| 1 | .000061 | .000053 | .000046 | .000043 |
| 1,000 | .015518 | .011975 | .008756 | .009120 |
| 100,000 | .661519 | .772107 | .856372 | .687926 |
| 120,000 | .839486 | .593762 | **1.038800 실패** | .604251 |

유일한 실패는 row14, **k5/조건수120000/UNIFORM_C2**, band0/class11(0부터 셈)이다.
S=S0가 정확한 균일 대조군이며 metadata나 학습된 가중치조차 필요하지 않았다. H의 비대칭 norm은
6.0869876e−12, norm 기반 허용치는5.8596333e−12다. SPD/top-gap 검사와는 별개인 대칭 검사다.
Scalar와 batch 모두 거절했다. 비균일 행이 통과했다고 실패한 필수 대조군을 제외하지 않는다.

## 판단과 한계

현재 고정된 수치 경로가 C2에서도 거절될 수 있음을 생성 입력으로 확인했다. 이는 C1과 공유하는
검증 실패 양식의 증거다. **C2의 수학적 아이디어가 작동할 수 없다는 증명도, metadata가 쓸모없다는
증거도 아니다.** 특정 실제 참가자의 C2 실패나 실제 정확도를 관측한 것도 아니다.

사전 규칙에 따라 C2를 `NOT_EVALUATED`로 닫는다. 남은 사람 실행 예산은 세 번째 후보나
허용오차 완화 권한이 아니다. 현재 두 슬롯의 종합 보고·후속 검증 계획을 작성한다.
별도 고정 [독립 감사 계약](task_trca_pair_s_conditioning_audit_contract.md)에 따라 원 생성 배열,
raw H, 고정 코드·수치 결과를 확인 중이며 이 문서에 최종 상태를 추가한다. 완료 감사의 PASS는
실패 기록이 정확하다는 뜻이지 C2가 실험 관문을 통과했다는 뜻이 아니다.

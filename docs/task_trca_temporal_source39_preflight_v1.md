# C1 실제 실행 경로 생성 사전검증

2026-09-09. [유한 연구 프로그램](metadata_learning_program_v1.md)의 C1 준비 결과다.
**생성 자료의 전체 학습·동결·평가·독립 cold 검산이 통과했다. 사람 metadata 효과는 아직 미평가다.**

## 실제로 실행한 것

- 통합 commit `5cfea671be09eff9f9cd9bf824c29954045ca2e9`의 같은 실행 경로를 사용했다.
- 고정 seed20260909, 가상9명×2interfaces×N17×k3/5. 모든10 EEG blocks에 서로 다른 noise를 생성했다.
  48 query행은12행을 네 번 복제한 것이 아니다. Metadata780packets는39routing ID에 생성했지만 EEG는9명이다.
- 세 외부 분할, 각9inner+1final pipeline, 총30pipelines/120heads/**24,000task updates**.
  생성 학습에서 고른 lambda는 사람 실험에 재사용하지 않는다.
- 모델3개를 먼저 동결한 뒤 query-bearing 요청72회. Access325행(324요청+barrier), 진행event92행을 저장했다.
- 실행112.244초, CUDA peak allocated59,799,552bytes, process peakRSS1,331,764KiB.
  생성 입력과 결과 전체186,520KiB. 기존 다른 GPUprocess·과거 산출물은 보존했다.

## 독립 검산

별도 cold CLI가5.263초에 `GENERATED_COLD_INDEPENDENT_AUDIT_PASS`를 반환했다.

- 108개 source validation CE의 최대 절대차6.939e−17.
- 36평가 cells의 열 가지 arm 점수/argmax와 총396개 정수count 비교가 일치했다.
  Positive-arm score 최대차2.727e−15, native cache차0, Q/MISSING exact.
- Manifest·program·25localcodepins·source/model/freeze/evaluation 파일을 직접 연결했다.
  정확한 역할별 access multiset과 event/access barrier, failure 우선권도 검증했다.
- 원래 native필터의 FULL_NATIVE와 같은 필터에 새 점수만 적용한 FULL_CENTERED를 구분했다.

## 고정 근거

출력 parent: `/home/whwovy/task-trca-temporal-source39-preflight-F2RGlt/`

| 산출물 | SHA256 |
|---|---|
| generated manifest | `99a20ce834786f0e705aaf31b8091a092e8a1b42ac039c1a67ef8e533d3202e7` |
| fixture origin receipt | `6de6fca3ef38f568b040b793c0d3242011dec12475449f39164cb9bd1fb25b62` |
| generated result | `7ab06a555c592b174e24cb0b894771a81c4daa84756f35641636b493ab534a20` |
| independent cold receipt | `ba36e82f2a82291b39b440b59af994a7b30ef199a21efaf7d7d653a122b3d42f` |
| prospective C1 human manifest | `05e6251de0463a0cf49d8ac434135d8df18544b3a640d9a6128432dd2eebdd33` |

사람 [실행 명세](../configs/analysis/task_trca_temporal_source39_execution_v1.json)는 위 cold receipt와
같은25codepins, 기존 native/source39 provenance, 앞서 고정한 C1/C2 설계와 프로그램을 연결한다.
양쪽 manifest validator 및 실행 전 preflight-chain 검사를 통과했다. 이 검사는 실제 사람 입력을 열지 않았다.
최신 전체회귀 **2,578tests PASS299.16초**, 기존68Torch warnings다. JUnit SHA
`77813bfdd74985aa86c168e78531a3d958d1dd0ed6e4e396126c8be86c2f8a64`.
생성 실행/전체검사 이후25runtimecodepins를 바꾸지 않았다. 이 사전검증 기록 시점 사람 입력·fit0이다.

## 해석 한계와 다음 단계

생성 A0는 고정 random correlations로, 실제 zero-calibration decoder 재현이 아니다.
새9명N17 경로는 사람39명/4window의 수치 안정성을 보증하지 않는다. 독립 감사의 수치 시작점은 저장된
Q/S/C와 Grams이며 raw EEG→Q15/native fit/Adam 자체의 독립 재현이나 OS sandbox 증명이 아니다.
Metadata 효능을 넣어 만든 fixture가 아니며 결과를 인간 성능·보정량 절감·학술 신규성으로 해석하지 않는다.

전체회귀도 통과했으므로 새 명세에 따라 C1 source391회만 실행하고 독립 감사/전체 사전 gate로 판단한다.
원래 실패와 부정 결과는 폐기하지 않는다. C2는 사전 수식만 고정된 조건부 후보이며 아직 활성화되지 않았다.
Held60·외부 요청·유료 자원은 이번 루프에서 자동 허용되지 않는다.

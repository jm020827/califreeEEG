# Metadata 학습 연구 루프 — 후보2개·유한 예산

2026-09-09. 새 user goal에 따른 전망적 연구 프로그램이다.
[기계 계약](../configs/analysis/metadata_learning_program_v1.json).
목표는 **metadata로 저보정 SSVEP 학습을 개선하고, Q+공통 정보 이상의 숫자 M 기여와 실제
보정량 감소를 분리 검증**하는 것이다. 긍정 결과를 강제하지 않으며 단위 테스트만으로 완료하지 않는다.

## 후보와 순서

| 순서 | 바꾸는 학습 결정 | 바꾸지 않는 것 | 왜 시험하나 |
|---|---|---|---|
| C1 temporal-R | 채널별 denominator 규제 모양 | 기존 Q/M·학습예산·새 공통 scorer 및 대조군 | 실제 learner는 생성 검증됐지만 사람 효과는 아직 미평가 |
| C2 pair-S | 보정 block 쌍의 재현성이 numerator S에 기여하는 양 | unweighted C·균일 template·공통 temporal scorer | 평균/SD로 축약한 M 대신 block 대응을 사용해 다른 공간필터 학습 지점을 시험 |

C2는 이전 context-template의 가중 파형 평균 반복이 아니다. 이전 방법도 이미 M을 학습에 넣었으므로
‘최초 metadata 학습’이라고 하지 않는다. 새 삽입 위치가 임피던스 효과의 증거인 것도 아니다.
C2의 수식·Q/M 특징·결측·pair 총량·bound·분모·학습·반증 기준을 **C1 새 최종 query 전에 고정**한다.
현재 C1 reader 구현과 병행할 수 있지만 그 명세 없이 사람 query 평가로 넘어가지 않는다.

## 허용과 경계

새 goal은 다음을 한 범위로 승인한다: 후보별 manifest·입력/완료 감사 구현 → 생성 archive 사전검증 →
정확히 허용된 source39 보정 EEG/M와 source 감독 자료 → 모든 모델 동결 후 최종 평가 → 독립 감사/해석.
과거 engineering-only 문서는 당시 이력으로 보존하고 이번 권한은 새 계약으로 연결한다.
옛 source39 VALIDITY_FAILURE나 negative 후보를 재개하지 않는다. Held60·retiredS1–S3,
외부 자료 요청 발송·유료 자원·제3 후보·사후 threshold 변경은 범위 밖이다.

## 실행 예산

- 과학 후보 최대2개, 주 실행 각1회. 동일 과학계약의 운송/저장 등 인프라 복구만 프로그램 전체1회 추가.
  총 사람 입력 시도≤3, 주 실행각≤24000updates, 부분실패/복구포함 총≤72000updates.
- 모델이 다른 최종 query 공개≤2회. Query 이후 복구는 같은 저장 동결모델/예측만 사용하고 재학습하지 않는다.
- 사람 실행각≤2시간·12GiB, 합계≤6시간·36GiB. Engineering compute≤12시간·출력4GiB.
  CPU1thread, 기존GPU fraction≤.16/allocated≤4GiB. 다른 GPUprocess를 종료하지 않는다.
- 생성seed20260909, 새로운 seed/window/feature/subgroup sweep0. 구현 버그는 생성 검증에서 고치되
  사람 수치실패의 bound·topgap·분산검사를 완화하는 것을 인프라 복구로 부르지 않는다.
- 추가 writingworktree2개까지만, 각checkout64MiB/test512MiB. 기존38trees와 실패 산출물은 보존한다.

이는 토큰 예산 설정이 아니라 연구·계산 예산이다. 새 goal에 사용자가 지정하지 않은 token budget은 넣지 않았다.

## 계속/중단 결정

1. C1의 **모든 calibration gate** 통과 + 독립 감사면 유망 후보로 가족 탐색을 멈추고 후속 독립 확인 계획을 쓴다.
2. 유효한 음성 또는 분류 이득만 있으면 그 후보를 종료하고 이미 고정된 C2가 적격일 때만 진행한다.
   결과에서 유리한 사람·조건을 골라 C2의 수식이나 문턱을 바꾸지 않는다.
3. 수치/감사 유효성 실패는 효능 미평가다. 원인·실패 경로를 감사하고 공유 결함이 해결되지 않으면
   다음 후보도 실행하지 않는다. C1의 실패는 C2의 긍정 근거가 아니다.
4. C2 종료/기각 또는 예산소진이면 전체 표·긍정/부정/미평가·한계·추가 검증 계획으로 프로그램을 닫는다.
   유망 후보를 얻기 위해 제3 후보를 자동 추가하지 않는다. 필수 권한/자료 부족은 완료가 아닌 병목이다.

## 근거와 검증의 수준

기존 [삽입 위치 검토](metadata_learning_covariance_design_review.md), [prior 실제 결과](metadata_prior_source39_v1_results.md),
[template 실제 결과](context_template_source39_v1_results.md), [temporal 검증](task_trca_temporal_v1_engineering.md)을 재사용했다.
Read-only history/skeptic/builder가 비중복 후보·개발 표본 한계·실행 경계를 분리 검토했다.
현재39명은 반복 노출 개발자료다. Nested CV와 독립 재계산 감사를 독립 확증 표본으로 바꾸어 부르지 않는다.
학술 신규성과 외부 일반화를 입증한 것이 아니며, 새 광범위 문헌검색보다 고정 후보의 실제 검증이 우선이다.

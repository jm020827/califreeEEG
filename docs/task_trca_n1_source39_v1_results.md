# N1 source39 v1 — 실행 상태와 결과

2026-09-09. **단일 실자료 실행은 출력 연결 오류로 종료했고, 독립 실패 감사를 완료했다. 성능·보정량은 미평가다.**

## 무엇을 검증하는가

연구목표는 적은 labeled calibration으로 SSVEP를 판별하는 것이다. Metadata 자체가 목표는 아니다.
이번 후보는 EEG에서 얻은 품질 정보 Q와 공통 acquisition 정보를 먼저 학습한 뒤 고정하고,
query 전에 관측한 채널별 impedance의 숫자 M을 작은 추가 학습 경로에 넣는다.
그 숫자가 채널 정규화를 바꾸고, 그 변화가 정확도와 실제 관측 보정량에 도움이 되는지를 묻는다.

원 가설·N1 수학·대조군·중단 기준을 바꾸지 않았다. 기존 C1/C2 실패 및 유효한 부정 결과도 보존한다.
이번 새 실행 ID는 `task-trca-n1-source39-v1`; 재사용한 수학 포맷은 `task-trca-n1-integration-v1`이다.
옛 모델·토큰을 새 실험의 권한으로 사용하지 않는다.

## 고정 설계

- 반복 노출된 개발39명, dry/wet, 4개 window, k3/k5: 312기본조건/624 budget cells.
- 3개 outer 분할, 각 inner3분할·lambda3개; Q-only 선택 후 Q/Q2/QM/SHAM 학습.
- 총30 pipelines/120 heads/24,000 updates. 모든 모델을 고정한 뒤에만 evaluation query를 읽는다.
- Q/Q2/SHAM/잘못 짝지은 M/stale M/missing M, 원 FULL_NATIVE·FULL_CENTERED·실제 A0와 비교한다.
- Metadata 증분, 관측0·36·60 labels에서의 조건별39/48정답(81.25%) 최초 도달, 비용·악화를 고정 기준으로 판단한다. 전체 평균 정확도≥80%는 별도 관문이다.
- CPU1·최대6시간·RSS16GiB·출력12GiB, human primary1회/모델세트 개봉1회/terminal audit1회.
- 첫 등록 실패 후 재시도·N2전환·문턱 완화0. Held60·외부요청·유료·GPU는 이번 범위 밖이다.

세부 기준은 [동결 설계](../configs/analysis/task_trca_n1_source39_v1.json)와
[실행 계약](task_trca_n1_source39_v1_build.md)이 우선한다.
Q2는 EEG에 들어 있는 모든 정보를 제거한 완전한 충분성 대조군이 아니다.
Source39 CI는 기술통계이며, 독립 확인이나 causal acquisition 효과를 증명하지 않는다.
관측 label 절감은 이 세 지점의 오프라인 최초 도달 비교다. 실제 적응형 중단 정책이나 impedance
측정·장비 준비를 포함한 전체 소요시간 절감은 검증하지 않는다. 절감 평균은 양쪽 모두 도달한
조건에 한정하므로 신규 도달·도달 상실·양쪽 미도달 수도 함께 보고한다. 미도달은 null로 유지한다.

## 완료한 선행 검증

| 단계 | 실제 결과 | 해석 범위 |
|---|---|---|
| 구현·통합 검사 | 새275tests, 전체3575tests PASS | 실행 경계와 기존 회귀검사 |
| 전체크기 생성 자원1 | 416cases/800updates, 525.71초, peak2.181GiB | 고정 screening 통과 |
| 새 생성 primary1 | 24,000updates/10arms/36cells, 594.08초 | 전체 학습·평가 경로 완료 |
| 새 생성 cold1 | 독립 검산 PASS, 4.49초 | 저장 통계 이후 점수·선택·접근 계보 확인 |

자원식 `36*fit + 3*prep + 600`은19,234.29초로21,600초 아래였다.
4×peakRSS는8.723GiB였다. 실제 human 시간·메모리를 보증하는 값은 아니다.
작은 generated 완료경로와416case 부하검사를39명 전체 nested 인공 재현이라고 부르지 않는다.

새 생성자료는 네 head family가 모두 작동했고, 모든10arms의 argmax/count가 독립 재계산과 같았다.
이는 개별120head가 모두 비영이거나 유용하다는 뜻은 아니다. Generated A0는 임의 correlation이며,
인공 M의 성능 이득·보정량 절감을 평가한 것도 아니다. Cold는 독립 raw 전처리/Q15/Adam 재현이 아니다.

첫 전체 테스트는 임시 parent 이름의 source39 문자열 때문에 기존 V2 보호규칙에 거부됐다.
5FAIL/14ERROR/604PASS 뒤 중단하고 모든 실패 JUnit을 보존했다. 보호규칙·코드·테스트는 바꾸지 않고
중립적인 새 parent에서 같은3575개를 통과했다. [모든 검사·실패·예산 기록](reports/task_trca_n1_source39_v1_preflight.json).

## 실자료 결과 — 실패 종료·독립 감사 완료

2026-09-09T09:51:53Z에 승인된 단일 primary를 시작했고, 10:54:25Z에 종료했다.
실패 기록의 경과시간은3,751.929초(약62.5분)다. 첫 outer의9개 inner+1개 final과
두 번째 outer의 첫 inner까지11개 pipeline/44개 head/8,800updates를 마쳤다.
전체30개 pipeline/24,000updates는 완료하지 못했다. 세 모델 중 첫 모델만 저장했고,
전역 모델 동결·평가 query·A0/FULL 읽기·성능 결과 공개는 모두0이다.

| 판단 | 결과 |
|---|---|
| 수치·접근·완료 유효성 | 전체 실행은 VALIDITY_FAILURE; 실패 계보 감사는 PASS. 전체 수치 재현 PASS는 아님 |
| QM3−Q3 및 Q2/SHAM 대비 추가 이득 | NOT_EVALUATED / null |
| 실제 관측 calibration labels 감소 | NOT_EVALUATED / null |
| 도움·악화 및 미도달 사례 | NOT_EVALUATED / null |
| 후보 유지·종료 | 이번 실행 슬롯 종료. 재시도0, 가설의 성공·반증 모두 아님 |

### 직접 원인과 아직 모르는 것

`failure.json`은 `BrokenPipeError: [Errno 32] Broken pipe`를 기록했다.
Traceback은 `scripts/run_task_trca_n1_source39.py:244`의 진행상황 stdout 출력이다.
`EventJournal`은 그 전에 파일에 이벤트를 쓰고 flush/fsync했으므로 마지막 `inner_complete`
seq73도 보존돼 있다. 학습 완료 뒤 진행 알림이 전체 실험을 중단시키는 실행 경로 문제다.
출력의 수신측이 왜 닫혔는지는 이 기록만으로 확정하지 못한다. 사용자 메시지·수치 불안정·
EEG 품질·시간/메모리 예산 초과 때문이라고 단정하지 않는다. 이 실행은 stdout 연결 종료에
견디지 못했으며, 선행 생성 검증 및3575tests도 이 실제 수명/출력 단절 상황을 보장하지 못했다.

### 단일 terminal 감사

`--failure-only`를1회 실행해 `FIRST_FAILURE_PROVENANCE_AUDIT_PASS`를 받았다.
감사4.606초/process5.00초, 36codepins·manifest·6개 partial 파일의 hash/0400/계보와
1,456개 연속 fit-only access(지원EEG832/학습 supervision416/M208),74events를 확인했다.
사람 NPZ는 hash만 읽었고 배열 decode·Adam·실패 수치 재현은 하지 않았다.
`partial_events.all_models_frozen_access_seq=1457`은 전역 동결 이벤트가 없는 경우의 내부
sentinel이며 실제 동결 행이 아니다. `publication_barrier_verified=false`, `models_frozen=false`다.

실패 원본·부분 모델·기존 모든 음성 결과를 보존했다. 부분 모델로 효능을 계산하거나
완성되지 않은 비교를 부정 결과로 바꾸지 않는다. Human primary/terminal audit는 각1회,
source39 model-set reveal0/held60·rawMAT·원전체M·GPU·외부요청·유료·추가실험0이다.

### 예산과 자원

Resource800+generated24,000+human8,800=등록 학습33,600updates다. Human의 완료/예산/차감
카운터는 모두8,800이며 남은15,200을 자동 재시도 권한으로 전용하지 않는다.
Human 내부 경과3,751.929초<21,600초; 생성/자원/두 감사는 각각 고정 시간 범위 안이다.
Human `primary.time`은0bytes로 끝나 외부 process elapsed와 전체 peakRSS는 미계측이다.
실행 중 RSS 관측은 있었지만 전체 peak의 증거로 올리지 않는다. 세 실험 parent의 최종 할당량
합계596,344KiB(약582.37MiB)는16GiB보다 작다. 전체 test 예산은 사전 기록에 별도 보존했다.

## 후속 계획 — 추가 승인 전 실행하지 않음

원 저보정 목표·M 가설·Q/Q2/SHAM/FULL/A0 대조와 문턱은 유지한다. 지금 필요한 차이는
모델 변경이 아니라 **대화 stdout과 장시간 작업 수명을 분리하는 실행 복구**다.

1. 파일을 원본 로그로 삼고 stdout/stderr를 일반 파일로 연결한 bounded supervisor/launcher를
   검증한다. 수신측 종료를 의도적으로 발생시키는 순수 생성 시험으로 작업·journal·종료 상태
   보존을 확인한다. 디스크 쓰기 실패까지 무시하는 예외 처리는 하지 않는다.
2. 승인 후에도 먼저 새 복구 envelope/입력·실행 pin/새 attempt ID와 예산을 고정한다.
   현재 실패 attempt를 재개·덮어쓰거나 부분 모델을 재사용하지 않는다. 예상 차이는 출력 연결이
   사라져도 동일 학습 일정이 유지되는 것뿐이며, metadata 정확도 향상을 예측한 수정이 아니다.
3. 제안 한도는 실행수명 생성 검증·관련tests 합300초/128MiB, 새 human primary1회/24,000updates/
   CPU1/최대6시간/RSS16GiB/출력12GiB와 terminal audit1회/30분, 총 새 출력16GiB다.
   과학 코드의 동일성·입력 계보를 다시 확인하고, 수학 변경이 필요하면 이 복구 범위를 중단한다.
   첫 실패 후 추가 복구0, 성공 뒤에도 원 고정 효능·보정량 판정과 독립 감사가 필요하다.
4. Source39는 계속 노출 개발자료다. 유망 결과가 생겨도 독립 확인이 아니며 held60/독립 paired-M
   자료 접근·외부 요청·유료 사용은 각각 별도 승인 사항이다.

이는 후속 제안이며 **현재 새 복구/추가 사람 실행 승인은 없고 착수하지 않았다**.
출력 연결 실패는 이번 구현을 종료시키지만 acquisition metadata 전체의 유용성을 반증하지 않는다.
원 연구 goal은 미완료로 유지한다.

최종 기계 상태는 [state](reports/task_trca_n1_source39_v1_state.json)에 있고,
불변 원본 failure/terminal audit와 journals가 기준이다. 이전 진행 snapshot은 당시 상태다.

## 원본 계보

- Resource: `/home/whwovy/task-trca-n1-source39-Nyp4o0/resource1/resource.json`, SHA `526ef01b8b5b3422b0dfbe3d7bbfec0f04d4625193703648111ce2088c6320e8`.
- Generated: `/home/whwovy/task-trca-n1-source39-Pborkn/task-trca-n1-source39-generated1/cold_audit.json`, SHA `25c62abe79166ff33779a2ac805846779b8dae60875b96901bad6f37f682cad1`.
- Human manifest: `/home/whwovy/task-trca-n1-source39-K1qYuE/human_manifest.json`, SHA `c1780ad0f3edcccef5b32cbf1a097d21255d54b11a7c0154f3cfc0f2cb5a380d`.
- Human failure: `/home/whwovy/task-trca-n1-source39-K1qYuE/task-trca-n1-source39-primary1/failure.json`, SHA `e4ebf797526580fd3a163038832e2f0d09dd3fd639140929ee8144bf96f2a107`.
- Human terminal: `/home/whwovy/task-trca-n1-source39-K1qYuE/task-trca-n1-source39-primary1/terminal_audit.json`, SHA `3e92cd6b2328c5f20cdba27507e860ee683644a73e8f50b8a0e088c6c76a92a9`.
- Human output의 `events.jsonl`/`access.jsonl`와 모든 실험 파일은 종료 후0400으로 보존했다.
- 36개 고정 실행 핀은 resource·generated·human이 같다. 진행 문서 커밋은 이 핀을 변경하지 않았다.

`academic-research` 원칙으로 구현 검증과 실자료 효능 근거를 분리했고, 기존 열린 gap과 부정 결과를 유지한다.
`coordinate-worktree-changes`에 따라 두 작성 lane을 격리하고 root가 통합·실행·최종 확인을 소유했다.
이번 새 문헌검색/PDF0, 신규 데이터 수집0이다. 기존39명 cache를 재사용하며 rawMAT/전체 원metadata는 읽지 않는다.

연구 workspace에는 새 claim `ffe23763f1e4081f`(QUALIFIED/5evidence)와 technique
`f719d2a9a58d4bfa`를 추가했다. 기존 gap `6a8254f849b36b2f`는OPEN이며 이전 claim을 덮어쓰지
않았다. Render/SQLite quick_check를 완료했다. [연구 근거 맵](/home/whwovy/research-spaces/califree-eeg-experiment-design/writing/claim-evidence.md).

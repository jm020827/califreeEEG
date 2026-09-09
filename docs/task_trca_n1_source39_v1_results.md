# N1 source39 v1 — 실행 상태와 결과

2026-09-09. **실자료 primary1 실행 중. 성능·보정량 결론은 아직 없다.**

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
- Metadata 증분, 관측0·36·60 labels에서의80% 최초 도달, 비용·악화를 모두 고정 기준으로 판단한다.
- CPU1·최대6시간·RSS16GiB·출력12GiB, human primary1회/모델세트 개봉1회/terminal audit1회.
- 첫 등록 실패 후 재시도·N2전환·문턱 완화0. Held60·외부요청·유료·GPU는 이번 범위 밖이다.

세부 기준은 [동결 설계](../configs/analysis/task_trca_n1_source39_v1.json)와
[실행 계약](task_trca_n1_source39_v1_build.md)이 우선한다.
Q2는 EEG에 들어 있는 모든 정보를 제거한 완전한 충분성 대조군이 아니다.
Source39 CI는 기술통계이며, 독립 확인이나 causal acquisition 효과를 증명하지 않는다.

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

## 실자료 결과 — 실행 중

2026-09-09T09:51:53Z에 승인된 단일 primary를 시작했다. 첫26fit ID의416cases 준비 이후 학습 중이다.
진행 중 통계나 선택한 window를 근거로 설정을 수정하지 않는다. 아래는 terminal audit 후 채운다.

| 판단 | 결과 |
|---|---|
| 수치·접근·완료 유효성 | PENDING |
| QM3−Q3 및 Q2/SHAM 대비 추가 이득 | PENDING |
| 실제 관측 calibration labels 감소 | PENDING |
| 도움·악화 및 미도달 사례 | PENDING |
| 후보 유지·종료 | PENDING |

최신 기계 상태는 [state](reports/task_trca_n1_source39_v1_state.json), 진행은 아래 원본 journals에 있다.
진행 snapshot의 완료 카운터는 그 시점의 값이며 미래 결과를 뜻하지 않는다.

## 원본 계보

- Resource: `/home/whwovy/task-trca-n1-source39-Nyp4o0/resource1/resource.json`, SHA `526ef01b8b5b3422b0dfbe3d7bbfec0f04d4625193703648111ce2088c6320e8`.
- Generated: `/home/whwovy/task-trca-n1-source39-Pborkn/task-trca-n1-source39-generated1/cold_audit.json`, SHA `25c62abe79166ff33779a2ac805846779b8dae60875b96901bad6f37f682cad1`.
- Human manifest: `/home/whwovy/task-trca-n1-source39-K1qYuE/human_manifest.json`, SHA `c1780ad0f3edcccef5b32cbf1a097d21255d54b11a7c0154f3cfc0f2cb5a380d`.
- Human output: `/home/whwovy/task-trca-n1-source39-K1qYuE/task-trca-n1-source39-primary1`; `events.jsonl`/`access.jsonl`는 실행 중 추가 기록된다.
- 36개 고정 실행 핀은 resource·generated·human이 같다. 진행 문서 커밋은 이 핀을 변경하지 않았다.

`academic-research` 원칙으로 구현 검증과 실자료 효능 근거를 분리했고, 기존 열린 gap과 부정 결과를 유지한다.
`coordinate-worktree-changes`에 따라 두 작성 lane을 격리하고 root가 통합·실행·최종 확인을 소유했다.
이번 새 문헌검색/PDF0, 신규 데이터 수집0이다. 기존39명 cache를 재사용하며 rawMAT/전체 원metadata는 읽지 않는다.

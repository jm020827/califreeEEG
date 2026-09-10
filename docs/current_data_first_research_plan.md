# 현재 자료 우선 — 연구계획 수정과 단일 선택공간 진단

> 2026-09-10 후속 구현: [시간·비용 경계 결과](acquisition_boundary_generated_v1_results.md). 실제로 수집한 trial과 선택 labels, 사전 context 시간까지 분리해 검증하는 인공 경로를 만들었다. 새34tests PASS와 기존 회귀fixture 배열상한 이탈을 함께 보고한다. 현재 실제 Choi 센서/시간 적격성은 미확정이고 새로운 사람 실험은 없다. 다음은 공개 export·취득 문서의 정확한 근거 확인이며, 더 많은 인공 PASS로 이를 대체하지 않는다.

> 2026-09-10 후속: [이미 보유한 Choi 보조센서 검토](choi_aux_context_feasibility_v1_results.md)를 마쳤다. IMU 기록은 논문에 보고됐지만 raw Gyro 대응·시간/단위와 Temperature 연결은 아직 확정하지 못했다. 새 dataset을 받거나 효능을 평가한 것이 아니다. 현재 데이터 우선 원칙을 유지하면서, 먼저 export 문서와 인공 시간경계/실제 비용 검증을 제안한다. 단위 미상은 모든 움직임 학습의 불가능성을 뜻하지 않으며 기존 부정 결과/완료 goal은 그대로다.

> 2026-09-10 최신: [유한 연구 루프 통합 결과](metadata_research_loop_closeout_20260910.md). 별도 N1-R의 실제 정확도·관측 보정량 비교까지 완료했으나 절감0이고, 후속 채널 진단도 음성이다. 현재 자료로 좁은 기작을 연구할 수 있다는 아래 원칙은 유지한다. 새 독립 자료가 모든 연구의 필수조건은 아니지만, 종료한 후보·단회 권한을 자동 재개하지 않는다. 새로운 표현과 새로운 측정, 로컬 보유와 실행·확증 권한을 분리한 후속 검증 계획은 통합 보고를 따른다.

> 2026-09-09 **고정 2후보 프로그램 종료**: [종합 결과와 후속 검증 계획](metadata_learning_program_v1_results.md). C1은 source39의24,000updates 뒤 최종 수치 검증 실패, C2는 균일 대조군의 사전 인공 수치 검증 실패로 종료했다. 독립 실패 감사·전체2857tests PASS를 완료했지만 **효능·보정량 감소는 미평가이며 유망 후보는 없다.** 원 저보정 목표와 연구 질문은 유지한다. 사람 attempt1/query모델 개봉1/recovery0; held60·외부 요청·유료 자원0. 다음 수치 검증 연구는 별도 계획/승인 대상으로 남기며 세 번째 후보나 실패 재실행을 자동 추가하지 않는다. 아래는 이전 단계 이력이다.

> 2026-09-09 [Temporal Q/QM 학습기 통합 검증 완료](task_trca_temporal_v1_engineering.md): 같은 새 점수의 Q/QM/Q2/SHAM 및 원 FULL_NATIVE/별도 FULL_CENTERED 대조 설계를 고정했다. 실제 Q15·Q 동결·잔차·nested 학습을 생성6명에서 연결했고, 네 head CPU/CUDA 일치·독립 cold 감사·전체2376tests PASS를 확인했다. **사람 EEG 효능·metadata 보정량 절감은 미평가**다. 이번 source archive/실제 M/기존 실패 진단/사람 query/held60 접근0. 다음은 새 역할 제한 reader·완료 경로 감사·실행 manifest의 별도 구현·생성 검증이며, 기존 실패 후보 재개나 source39 자동 실행은 없다. 아래는 이전 단계 이력이다.

> 2026-09-08 [필터 부호 독립 수학·인공 검증 완료](task_trca_shape_signfree_v1_engineering.md): 새 C-normalized projector와 성분별 시간중심 점수의32tests·CPU/CUDA·독립 SciPy 검산을 통과했다. 보존된 실패 행렬 한 지점에서도 계산·1차 미분을 확인했다. **Native 점수와 일반적으로 다른 별도 후보이며 metadata 효과·보정량 절감은 미평가**다. 기존 실제 후보는 종료 유지, 새 사람 학습·query·held60 접근0(이미 노출된 작은 진단 JSON만 재검산). 다음은 동일 새 scorer의 Q/QM 및 native/centered FULL 대조 설계·실제 learner 결합 검증이다. 자동 사람 재실행은 없다. 아래는 이전 단계 이력이다.

> 2026-09-08 [task-shape 실제 실행 종료](task_trca_shape_source39_v1_results.md): `VALIDITY_FAILURE`. 첫 외부 분할10pipelines는 완료·독립 검산했으나 두 번째 분할의 SHAM 학습이 native 기준 정렬 하한을 위반했다. 같은 실패 지점의 CPU/GPU 재현으로 확인했으며 최종 query·held60 접근0이다. **Metadata 효과 없음이 아니라 효능 미평가**다. 연구목표는 유지하고 다음은 별도 필터 방향·부호 규칙의 수학/인공 검증 설계다. 원 후보 재실행·실패 대조군 제외·threshold 완화·held60 자동 개봉은 없다. 아래는 이전 단계 이력이다.

> 2026-09-08 [task-shape 인공 구현 검증](task_trca_shape_engineering_v1_results.md)까지 완료했다. 기존39명에 새 방법을 실행한 것은 아니며 새 data 요청도 없었다. 다음 병목은 실제 입력 역할 경계·독립 검산·실행 계약이다. 기존 자료를 좁은 개발용으로 사용하는 원칙과 독립 paired-acquisition 자료의 확증/일반화 역할을 구분한다.

> 2026-09-08 후속 설계: [task-aligned Q+M](task_aligned_trca_shape_v1_design.md)를 한 후보로 명세했다. 기존39명은 추후 좁은 개발 평가에 쓸 수 있으나 반복 노출을 독립 확증으로 바꾸어 부르지 않는다. 새 데이터 요청은 이번 구현 전제조건이 아니다. **이번에는 설계만 완료**, 새 EEG/M 접근·학습0; 다음은 인공 구현 검증이다. 외부 paired-acquisition 자료의 일반화 보강 가치는 남고 held60은 자동 개봉하지 않는다.

> 2026-09-08 후속 기하 진단 완료: [규제가 실제 support 필터를 얼마나 바꿨나](trca_support_geometry_v1_results.md). Metadata를 넣기 전 고정γ=.1만으로 필터 방향이 중앙값 약44.4°/42.3° 바뀌었다. 큰 연산자 변화는 확인했지만 정확도 손실의 인과 원인이나 새 metadata 이득은 입증하지 않았다. 원 후보 종료·held60 보호는 유지하며 새 gamma 탐색/분류 실행은 없다. 아래는 기존 효능 결과와 이전 단계 기록이다.

> 2026-09-08 실제 실험 종료: [Source39 metadata-prior 결과](metadata_prior_source39_v1_results.md). 39명×8조건의 단일 후보 학습·평가와 독립 검산을 완료했다. QM3−Q3 −0.006677%p, QM5−Q5 0%p이고312개 사람×조건의80% 최초 도달 단계가 모두 같아 보정량 절감은0이다. **이 구현의 metadata 이득 미확립으로 종료**한다. 공통 규제부터 native FULL보다 낮았다는 한계, 실행 복구와 정수 count 보고 정정은 결과 문서에 보존한다. 연구목표는 유지하되 새 후보 튜닝·held60 자동 개봉은 없다. 아래는 이전 단계 기록이다.

> 2026-09-08 검증 설계 수리 완료: [M-blind Q 학습·필터 전달성 검사](metadata_prior_validation_repair_results.md). 참가자 분리 nested Q 선택, proxy 수준/채널 모양 분리, 식으로 만든 배열의 필터→점수→선택 경로를 구현·검증했다. 새 M 효능 실험은 아니며 이전 합성 v1의 효과 미확립 판정은 유지한다. 다음은 M을 보지 않는 별도 난이도 검증 설계이며 실제 EEG/held60은 이번에도 접근하지 않았다. 아래는 이전 단계 기록이다.

> 2026-09-08 합성 단계 완료: [M-prior v1 실제 합성 결과](metadata_trca_prior_synthetic_v1_results.md). Native-compatible operator와 Q/QM·대조군 구현/검산은 완료했지만, 고정 screen은 **효과 미확립**이다. 주요3시나리오 Q 정확도100%로 분류 ceiling이 있었고, 반복 불일치 예측도 Q만 추가 적합한 Q2가 QM보다 좋았다. 실패 start와 같은 suite의 복구 결과를 모두 보존했다. 새 사람 데이터/held60 접근0이며 난이도·Q 적합의 검증 설계가 다음 검토 대상이다. 결과를 보고 설정을 바꾸거나 사람 실험으로 자동 승격하지 않는다. 아래는 이전 단계 기록이다.

> 2026-09-08 설계 검토 완료: [Metadata의 보정학습 삽입 재검토](metadata_learning_covariance_design_review.md). 목표는 그대로이며, native TRCA에 동일 총량의 Q/Q+M 채널 규제를 주는 후보 하나를 제안했다. V1·context-template도 이미 학습단계 M을 사용했으므로 ‘최초 metadata 학습’이 아니다. 관련 공개 PDF2편 선택 정독·인공 대수 검산 완료, 새 사람 데이터 접근·효능 결과는 0이다. 다음은 proxy·대조군 명세와 순수 합성 구현 검증이며, source-only 입력 준비·실제 평가는 별도 단계다. 아래 완료 결과와 종료 경계는 그대로 보존한다.

> 현재: 명시 승인된 [cold-r1 별도 진단과 독립 검산 완료](native_subset_headroom_cold_r1_results.md).
> 사후 상한 k3/k5=46.67/56.06%, 실제 Q=36.99/44.21%다. M 정보나 실제 비용 절감의 증거는 아니다.
> [이전 start-only 실패](native_subset_headroom_source39_results.md)는 그대로 보존했고 동일 attempt 재실행은 없다.
> 아래 과학설계를 바꾸지 않고 새 실행 신원만 분리했다. 이번 진단은 종료하고 새 learner는 자동 추가하지 않는다.

2026-09-08 사용자 ‘응 그렇게하자’ 승인. 연구목표는 새 사용자의 적은 labeled SSVEP 보정이며,
acquisition metadata는 이를 위한 수단이다. **새 독립 paired-M 확보를 모든 후속 연구의 필수 입장 조건으로 두지 않는다.**
현재 Wearable의 EEG와 사전 block별 임피던스로 같은 측정환경의 좁은 가설을 연구할 수 있다.
이 판단은 모든 효과에 충분한 통계적 검정력이 있다는 보장이나 최근 실패의 원인이 표본수라는 주장이 아니다.

## 무엇을 수정하고, 무엇을 보존하나

- [Known-zero 결과](native_subset_known_zero_source39_results.md)의 ‘다음 연구에는 독립 자료가 먼저’라는
  운영상 요구를 앞으로의 일반 필수조건에서 내린다. 외부 자료는 장비/기관/획득환경 일반화 및 별도 재현용 보강이다.
- [2026-09-07 재검토](research_decision_reaudit_20260907.md)의 ‘현재 oracle 실행 미포함’을
  **아래 단일 저장 예측 진단에 한해서** 변경한다. 옛 실행계획/JSON/종료 결과는 수정하거나 재실행하지 않는다.
- 직접 M 방식1회＋기작 수정1회의 종료와 no-go 해석은 그대로다. 이번은 제3의 효능 후보가 아니다.
- 개발39명은 반복노출된 자료다. 보존60명은 최종 성능 미개봉 유지하며 새 개발용으로 전환하지 않는다.
  같은 코호트 새 참가자 확인은 외부 획득환경 재현과 다르다. 향후 공개 native 전체-cohort preset 상속 문제와
  표본 정밀도/실용효과/최종 프로토콜을 검토한 뒤 별도 권한이 필요하다.
- S1–S3, raw EEG, 전체 manifest/임피던스, 새 metadata packet, 외부 메일/다운로드는 이번 범위 밖이다.

## 한 번만 할 진단: 후보들 안에 정답이 있었나?

현재 정책은 36/60 labels를 이미 모은 뒤 FULL 또는 한 block을 제외한 모델을 고른다.
먼저 이 **고정된 선택지 자체에** 얼마나 개선 여지가 있는지 확인한다.

각 query에서 후보들이 A,A,B를 예측하고 정답이 B라면, 정답을 아는 이상적 선택은 맞힐 수 있다.
정답이 C라면 어떤 선택기도 이 후보들만으로 맞힐 수 없다. 이 이상적 선택의 정확도를 상한으로 계산한다.
실제 배포 시 정답은 없으므로 **구현 가능한 모델도, metadata로 달성한 정확도도 아니다.**

고정 계약: `configs/analysis/native_subset_headroom_source39_v1.json`.
입력은 완료 envelope-r1 `features.npz`와 known-zero `result.json` 두 SHA고정 파일만이다.
**Known-zero 수정 후** Q/QM 최종 actions의 BA를 해당 결과와 먼저 맞춘다. 원 envelope-r1의
수정 전 actions와 혼동하지 않으며, 새 선택 gain/학습/metadata변환은 계산하지 않는다.
기존 signed native 가중치·float64 dot·argmax 동률순서·class/block 순서를 그대로 쓴다.
k3는 FULL＋3개 omission, k5는 FULL＋5개 omission이며 padding/A0를 후보에 더하지 않는다.

보고할 것은 전39명×8조건×k3/5의624행이다.

1. 후보의 예측 불일치, 모든 후보 오답, 불일치하지만 모두 오답인 빈도.
2. 어떤 후보든 정답인 hindsight 상한과, 고정 Q/QM이 놓친 회복 가능 비교 수.
3. Oracle3−고정Q5, 36/60labels 및 기존0/3/5 grid의 최초80% 도달 **가능성 상한**.
4. 개인 안에서8조건을 평균한 뒤39명 기술적 구간과 모든8조건 표. Window/query를 독립 사람으로 세지 않는다.

Oracle3/5 후보군은 서로 중첩집합이 아니므로 비용 증가에 따른 단조성을 강제하지 않는다.
K0는 원 A0를 사용하며, 최초도달은 이후 성능 유지나 미측정k1/2/4의 최소비용을 보장하지 않는다.
원래 획득한 label 비용은 빼지 않으며 사용자 총 소요시간 절감으로 바꾸지 않는다.

## 결과 해석과 중단

| 관측 | 허용되는 해석 | 금지되는 결론 |
| --- | --- | --- |
| Oracle조차Q보다 거의 못 올라감 | 이 고정 후보공간의 남은 이득이 작다 | metadata 일반에 정보가 없다 |
| Oracle는 높지만Q/QM은 낮음 | 후보 안의 정답을 선택하는 데 미해결 여지가 있다 | M이 그 정답 후보를 식별한다 / 새 모델이면 성공한다 |
| Oracle3도Q5에 못 미침 | 해당 자료/후보공간에서 선택만으로5→3 비용 대체에 한계 | 모든 low-calibration 방법이 불가능하다 |
| Oracle3이Q5에 근접하거나 능가 | label-aware한 관측자료 상한은 허용한다 | 실제 M 기반 보정 절감 달성 |

어느 결과든 **이 진단 보고로 멈춘다.** 새로운 learner/threshold/후보pool/조건선택을 자동 추가하지 않는다.
다음 기작 연구가 필요하면 ‘M이 Q 외에 무엇을 알려주며 어떤 학습 결정을 바꾸는가’라는
가설을 새로 제시하고, 정보 접근/대조군/실용성/중단 기준을 별도로 고정한다.
이 진단은 M의 조건부 정보 여부를 직접 검정하지 않는다. 개선 상한은 그 질문의 답이 아니다.

## 실행·검증·기록

새 시작은 입력 byte 읽기보다 먼저, 새 출력 root `headroom-source39-v1`의 start/result만0400 배타 게시한다.
CPU1/BLAS1,120초,입력128MiB/출력4MiB. 소스/설정/입력 SHA와 commit을 기록하고 원본 해시를 재확인한다.
입력 검증 실패는 과학적 실패가 아니다. 같은 attempt 재시도·추가 입력으로 복구는 하지 않는다.
독립 scalar vs vectorized oracle, 실제 dot 순서와 독립 score 일치, 원 BA 재현, 모든 선택 인덱스/grid,
동답/모두오답/회복가능/중복·순열/범위오류 사례를 인공 테스트한다. Oracle action/훈련 target은 게시하지 않는다.

`academic-research`: 기존 측정/문헌/가설을 분리하고 external-gap을 보강 역할로 수정한다.
새 광범위 문헌 탐색이나 추가 자료 확보가 필요하다고 가정하지 않는다. 이번 새 검색/PDF는0이다.
`coordinate-worktree-changes`: base main cfc1f25, clean. 약263GiB 가용, 기존29worktrees 보존.
명세/코드/테스트/문서/SQLite는 main 단일 writer이고 두 reviewer는 공유 repository 읽기 전용이다.
작업이 같은 명세에 의존하므로 별도 writing lane/worktree는 추가하지 않는다. Temp예산1GiB, 환경 설치/push/cleanup0.

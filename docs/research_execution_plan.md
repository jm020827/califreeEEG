# califreeEEG 연구 실행 계획

> 2026-09-09 **새 goal 진행 중**: [Metadata 학습 후보2개 프로그램](metadata_learning_program_v1.md)의 가족·예산·중단 규칙을 고정했다. C1 temporal-R 다음 조건부 C2 pair-S이며 C2 수식/특징도 새 사람 학습 전에 고정했다. 이번 명시적 위임은 사전검증을 통과한 새 source39 개발 실행을 포함한다. 생성9명3outer 24,000updates·독립cold·전체2578tests가 통과했고 C1 source39 실행 명세를 고정했다. 다음은 최초의 새 C1 사람 실행이며 이 사전검증 시점 사람 학습·query0이다. Held60·외부 요청·유료 자원·옛 종료 후보 재개는 제외하며 문서/단위테스트만으로 연구 goal을 완료하지 않는다. 아래 engineering-only 권한은 이전 단계 이력이다.

> 2026-09-09 [Temporal Q/QM 학습기 통합 검증 완료](task_trca_temporal_v1_engineering.md): 같은 새 점수의 Q/QM/Q2/SHAM 및 원 FULL_NATIVE/별도 FULL_CENTERED 대조 설계를 고정했다. 실제 Q15·Q 동결·잔차·nested 학습을 생성6명에서 연결했고, 네 head CPU/CUDA 일치·독립 cold 감사·전체2376tests PASS를 확인했다. **사람 EEG 효능·metadata 보정량 절감은 미평가**다. 이번 source archive/실제 M/기존 실패 진단/사람 query/held60 접근0. 다음은 새 역할 제한 reader·완료 경로 감사·실행 manifest의 별도 구현·생성 검증이며, 기존 실패 후보 재개나 source39 자동 실행은 없다. 아래는 이전 단계 이력이다.

> 2026-09-08 [필터 부호 독립 수학·인공 검증 완료](task_trca_shape_signfree_v1_engineering.md): 새 C-normalized projector와 성분별 시간중심 점수의32tests·CPU/CUDA·독립 SciPy 검산을 통과했다. 보존된 실패 행렬 한 지점에서도 계산·1차 미분을 확인했다. **Native 점수와 일반적으로 다른 별도 후보이며 metadata 효과·보정량 절감은 미평가**다. 기존 실제 후보는 종료 유지, 새 사람 학습·query·held60 접근0(이미 노출된 작은 진단 JSON만 재검산). 다음은 동일 새 scorer의 Q/QM 및 native/centered FULL 대조 설계·실제 learner 결합 검증이다. 자동 사람 재실행은 없다. 아래는 이전 단계 이력이다.

> 2026-09-08 [task-shape 실제 실행 종료](task_trca_shape_source39_v1_results.md): `VALIDITY_FAILURE`. 첫 외부 분할10pipelines는 완료·독립 검산했으나 두 번째 분할의 SHAM 학습이 native 기준 정렬 하한을 위반했다. 같은 실패 지점의 CPU/GPU 재현으로 확인했으며 최종 query·held60 접근0이다. **Metadata 효과 없음이 아니라 효능 미평가**다. 연구목표는 유지하고 다음은 별도 필터 방향·부호 규칙의 수학/인공 검증 설계다. 원 후보 재실행·실패 대조군 제외·threshold 완화·held60 자동 개봉은 없다. 아래는 이전 단계 이력이다.

> 2026-09-08 [task-shape core 구현·인공 검증](task_trca_shape_engineering_v1_results.md) 완료. 다음은 실제 archive 역할 제한 reader → 독립 저장 score/판정 auditor → start/freeze/eval cold CLI → 입력·코드·자원 pinning을 갖춘 별도 source39 실행 계약이다. 현재 완료는 인공 검증까지이며 사람 EEG/held60을 열지 않는다. 아래 ‘설계만 완료’는 이전 상태다.

> 2026-09-08 새 후보 설계 완료: [task-aligned Q+M 설계](task_aligned_trca_shape_v1_design.md)와 [구현 계약](task_aligned_trca_shape_v1_implementation.md). 다음 순서는 mask-only 특징/연산자 → nested 분류 learner → 역할 제한 reader → 독립 auditor → 인공·회귀·runtime 검증 → 별도 사람 실행 계약이다. 현재는 **DESIGN_ONLY**, 새 학습·효능 실행0이며 이전 runner를 재개하지 않는다. 기존 후보 종료·held60 보호는 유지한다.

> 2026-09-08 후속 기하 진단 완료: [규제가 실제 support 필터를 얼마나 바꿨나](trca_support_geometry_v1_results.md). Metadata를 넣기 전 고정γ=.1만으로 필터 방향이 중앙값 약44.4°/42.3° 바뀌었다. 큰 연산자 변화는 확인했지만 정확도 손실의 인과 원인이나 새 metadata 이득은 입증하지 않았다. 원 후보 종료·held60 보호는 유지하며 새 gamma 탐색/분류 실행은 없다. 아래는 기존 효능 결과와 이전 단계 기록이다.

> 2026-09-08 실제 실험 종료: [Source39 metadata-prior 결과](metadata_prior_source39_v1_results.md). 39명×8조건의 단일 후보 학습·평가와 독립 검산을 완료했다. QM3−Q3 −0.006677%p, QM5−Q5 0%p이고312개 사람×조건의80% 최초 도달 단계가 모두 같아 보정량 절감은0이다. **이 구현의 metadata 이득 미확립으로 종료**한다. 공통 규제부터 native FULL보다 낮았다는 한계, 실행 복구와 정수 count 보고 정정은 결과 문서에 보존한다. 연구목표는 유지하되 새 후보 튜닝·held60 자동 개봉은 없다. 아래는 이전 단계 기록이다.

> 2026-09-08 검증 설계 수리 완료: [M-blind Q 학습·필터 전달성 검사](metadata_prior_validation_repair_results.md). 참가자 분리 nested Q 선택, proxy 수준/채널 모양 분리, 식으로 만든 배열의 필터→점수→선택 경로를 구현·검증했다. 새 M 효능 실험은 아니며 이전 합성 v1의 효과 미확립 판정은 유지한다. 다음은 M을 보지 않는 별도 난이도 검증 설계이며 실제 EEG/held60은 이번에도 접근하지 않았다. 아래는 이전 단계 기록이다.

> 2026-09-08 합성 단계 완료: [M-prior v1 실제 합성 결과](metadata_trca_prior_synthetic_v1_results.md). Native-compatible operator와 Q/QM·대조군 구현/검산은 완료했지만, 고정 screen은 **효과 미확립**이다. 주요3시나리오 Q 정확도100%로 분류 ceiling이 있었고, 반복 불일치 예측도 Q만 추가 적합한 Q2가 QM보다 좋았다. 실패 start와 같은 suite의 복구 결과를 모두 보존했다. 새 사람 데이터/held60 접근0이며 난이도·Q 적합의 검증 설계가 다음 검토 대상이다. 결과를 보고 설정을 바꾸거나 사람 실험으로 자동 승격하지 않는다. 아래는 이전 단계 기록이다.

> 2026-09-08 설계 검토 완료: [Metadata의 보정학습 삽입 재검토](metadata_learning_covariance_design_review.md). 목표는 그대로이며, native TRCA에 동일 총량의 Q/Q+M 채널 규제를 주는 후보 하나를 제안했다. V1·context-template도 이미 학습단계 M을 사용했으므로 ‘최초 metadata 학습’이 아니다. 관련 공개 PDF2편 선택 정독·인공 대수 검산 완료, 새 사람 데이터 접근·효능 결과는 0이다. 다음은 proxy·대조군 명세와 순수 합성 구현 검증이며, source-only 입력 준비·실제 평가는 별도 단계다. 아래 완료 결과와 종료 경계는 그대로 보존한다.

> 2026-09-08 현재 완료: [Headroom cold-r1 진단](native_subset_headroom_cold_r1_results.md)과 독립 수치 검산을 마쳤다. 실제 Q k3/k5=36.99/44.21%, 정답을 아는 사후 상한=46.67/56.06%이며, 이상적 선택도312개 사람×조건 중217개는 관측 grid에서80% 미도달이다. **선택 개선 여지는 있으나 M 효과·실제 보정량 절감을 입증한 것은 아니다.** 전체1561tests PASS, 실패한 이전 start 보존·held60 미개봉. [현재 자료 우선 계획](current_data_first_research_plan.md)의 외부 paired-M 선택적 보강 원칙을 유지한다. 새 learner/추가 조건 탐색은 자동 실행하지 않는다. 아래는 과거 단계 이력이다.

> 2026-09-08 현재 완료: [Known-zero 단일 수정 실제 결과](native_subset_known_zero_source39_results.md)를 개발39명에서 단회 평가하고 독립 전체 검산을 통과했다. 수정 QM−Q는 k3 **0pp**, k5 **+0.01068pp**이며 80% 최초 도달 보정량은312개 사람×조건 모두 같았다. 수정 자체도 k5 정답을 Q2개/QM1개 줄였다. **추가 M 보정비용 절감 미확립으로 이 구현 탐색을 종료한다.** 직접 M 평가1회+기작수정 평가1회 예산을 사용했으며 새 후보/재학습/held60 자동 개봉은 없다. 연구목표는 유지하고 다음 공백은 독립 paired-acquisition 자료와 사전 가설이다. 전체1520tests PASS, 새 독립M자료·외부요청0. 아래는 보존한 이전 단계의 이력이다.

> 현재 완료: [envelope-r1 결과](native_subset_m_envelope_r1_results.md) 개발39명 단회 실행·독립 검산 PASS. 입력11→6 연결만 복구했고 학습수식/SHAM/평가조건은 그대로다. QM3−Q3=0pp, QM5−Q5=+0.00534pp로 유용한 M 순증분과 추가 보정비용 절감을 입증하지 못했다. 다음은 동일답 후보 중복·known-zero gain이라는 관측된 선택 병목이 기작 수정1회를 정당화하는지 명세하는 단계이며, 아직 새 정책/후속 outcome은 없다. 전체1345tests PASS, held60 미개봉. 아래 ‘현재/최신/다음/미승인’은 보존한 이전 실행 이력이다.

> 최신 실행: [native-subset-m-source39-v1](native_subset_m_source39_v1_results.md)은 metadata envelope 연결 오류로 EEG 이전에 중단됐다. 재시도는 하지 않고 start-only 산출물을 보존했다. 다음은 연구목표·모델·SHAM namespace를 유지하는 입력-adapter 복구용 별도 attempt의 명시적 승인과 실제 연결 시험이다. 새 decoder 탐색이나 held60 개봉은 없다.

> **현재 완료 단계:** [native eTRCA source39 결과](author_etrca_source39_v1_results.md) 39명/2028행/52summary 단회 완료·독립 검산 PASS. 조건별 학습 신호는 있지만 전반적 utility·새 M 효과는 미입증이다. 한 baseline sprint와 두 bridge를 사용했으므로 새 decoder 계열 탐색은 종료한다. 다음은 공통 interface를 포함한 matched Q에 추가 사전 acquisition-M이 주는 효용을 비교하는 명세다. 전체1196tests PASS, held60 미개봉. 아래 latest/current는 역사적 단계다.

기준일: 2026-08-30
프로토콜 버전: 0.3 (평가 골격), 0.4-dev 최소 공정성 구현

> **2026-09-07 현재 실행 전략 수정:** [기존 판단 재검토·유한 연구 루프](research_decision_reaudit_20260907.md)를 우선한다. 연구목표와 기존 종료 판정은 유지하되 Q>A0 평균 우위를 M 검정의 입장권으로 요구하지 않는다. 저자 구현 호환성 확인1개 → 직접 M 가설1개 → 원인 기반 수정 최대1회 뒤 점검으로 제한한다. 아래 overlay의 다음 신규 EEG 결합식 제안은 자동 실행하지 않는다. Source39 개발/held60 보호와 강한 비교군·실용성·비용 요구는 유지한다.

> **2026-09-07 latest human closeout:** [reference-calibration-source39-v1 결과](reference_calibration_source39_v1_results.md)의 두 arm 모두 AQ_NOT_ESTABLISHED. 7956행 단회 실행/독립audit PASS. 실제 support가 wrong-label보다 낫지만 reference보다 낮고, notch도 손실을 충분히 회복하지 못했다. 모든 대역 및 사후 점수분해를 완료했다. Reference-preserving 결합은 당시 제안이고 현재 후속은 위 재검토 계획이다. 해당 실험 Metadata/held 접근0. [eTRCA 합성 API 확인](author_etrca_compatibility.md)은 새 준비 작업이며 human 성능 결과가 아니다. 아래 newest/current 표기는 당시 역사적 overlay이며 이전 후보를 재개하지 않는다.

> **2026-09-07 newest overlay:** [spatial-calibration-source39-v1 결과](spatial_calibration_source39_v1_results.md)는 AQ_NOT_ESTABLISHED로 단회 종료·독립 검산 PASS다. Metadata-free IT-CCA의 early-AUC 이득0.004748pp로 utility/supervised-specificity 기준을 실패했다. 다음은 새 source39-only per-band/참조기반 learner 진단이며 아직 다음 실행 없음. Metadata/manifest/held 접근0. 아래 context-template와 기존 Phase들은 종료·역사적 이력이다.

> **2026-09-07 current overlay:** [목표 중심 연구 루프](goal_aligned_research_loop.md)의
> `context-template-source39-v1`은 AQ_NOT_ESTABLISHED로 단회 종료·독립 검산 PASS했다.
> [실제 결과와 다음 learner 병목](context_template_source39_v1_results.md)을 우선한다.
> 이전에 노출된 source39의 raw EEG와 pre-block
> impedance만 새로 명시한 접근 범위에서 분석한다. 강한 EEG-only·eTRCA·shuffle/stale/order
> 대조군과 5→3-shot 비용 비교를 함께 평가한다. 아래 역사적 Phase3의 held60/독립 확인
> 권한을 복구하지 않는다. V3/V4 종료 후보와 결과는 불변이다.

> **2026-09-06 V2 overlay:** 현재 V2 기준원은
> [V2 설계](metadata_calibration_efficiency_v2_design.md)와
> `configs/analysis/metadata_calibration_efficiency_v2.yaml`이다. V11 synthetic
> lockbox는 claim과 미래 beacon을 소비한 뒤 efficacy 계산 전 validator 재귀 오류로
> `infrastructure inconclusive` terminal이 됐다. 같은 lockbox 재실행과 external·Choi·held
> outcome은 금지한다. 상세 증거는 [V2 terminal 결과](metadata_calibration_efficiency_v2_results.md)를
> 따른다. 아래 historical Phase 3의 39/60 실행 권한은 되살아나지 않는다.

> **2026-09-04 superseded / historical only:** `DEC-20260904-005`가 현재 목표를
> metadata-assisted calibration-efficient SSVEP로 복구했다. 아래 strict-k0 A0/A2 질문과
> Phase 3은 당시 계획이며 실행하지 않는다. 현재 설계는
> [metadata-assisted low-calibration 설계](metadata_calibration_efficiency_design.md)를 우선한다.

> 아래 본문은 Protocol 0.3의 **당시 연구 질문**을 유지했던 실행안이다. 당시 5-fold split·평가·iid 추론 계약은 `DEC-20260830-002`와 `0.4-lockbox`가 대체했고, 그 후 physical 후보도 no-go로 종료됐다. 현재 구현 범위와 남은 gate는 [프로토콜 현재 상태](research_protocol_status.md)를 우선한다.

> **2026-09-01 실행 결과:** `DEC-20260901-004`의 physical reveal #2는 18/18 training·3/3 intervention·atomic publication까지 완료됐다. Assay potency는 유효했지만 A0 BA `0.1292`, Full A2 BA `0.1194`, Δ `−0.0097`이고 네 substantive gate가 실패했다. 사전 중단 규칙에 따라 현 `physical_hybrid_v1`의 Phase 3은 no-go이며 39/60 confirmatory는 봉인한다. 같은 S1–S3를 재튜닝·재공개하지 않는다.

> 데이터 수를 늘리는 기준, exact-label family와 shared-frequency OOD의 구분, 최신 strict k=0 baseline, foundation-model target exposure gate는 [연구 의미·데이터 확장·최신 문헌 전략](research_significance_data_strategy.md)을 따른다.

> **2026-08-30 revision 경계:** 반대로 결합된 `wearable_v2`와 그 A2 결과는 무효다. 코드는 `[dry, wet]` numeric signature와 `wearable_v3` guard로 복구했고 전체 102명 v3 acceptance audit를 통과했다. Protocol 0.4 confirmatory freeze 전에는 아래 Phase 3을 시작하지 않는다.

> **DEC-20260830-001:** S1–S3는 사전 성능 노출 때문에 영구 development-only다. 전체 자산은 102명·24,480행이지만 confirmatory primary는 S4–S102의 99명·23,760행이다. 이 결정은 과거 `N=102 primary + N=99 sensitivity` 서술을 대체한다. 현재 기준원은 [프로토콜 상태](research_protocol_status.md)와 [append-only 연구일지](research_log.md)다.

> **DEC-20260830-002:** N=99 5-fold primary는 폐기했다. 겹치는 training fold가 만드는 model-shared dependence를 iid subject inference가 처리하지 못했고 target-free simulation에서 Type-I inflation을 확인했다. S4–S102는 성능을 보지 않고 39명 frozen training과 60명 independent lockbox로 사전 할당한다. 과거 문서의 `5-fold×3-seed`, `N=99 independent unit`, condition별 inferential test는 이 결정이 대체한다.

## 범위 고정

당시 연구 질문은 다음과 같았다.

> 구조화된 획득조건 metadata가 target 사용자 데이터로 적응하지 않는 k=0 조건에서 처음 보는 사용자·데이터셋·전극 조건의 closed-set SSVEP 분류 성능과 강건성을 개선하는가?

다음은 이번 연구 범위가 아니다.

- 새로운 stimulus frequency/class 발견
- target stream을 이용한 online/continual adaptation
- 일반적인 open-set OOD 탐지
- LLM `pass@K`의 직접 적용

## 사전지정 가설

- 당시 Primary H1: frozen 39명 source-only training에서 만든 fixed 3-seed A2 ensemble이 완전히 독립된 60명 lockbox의 k=0 balanced accuracy에서 구조와 parameter shape를 맞춘 A0 ensemble보다 사전지정 SESOI를 초과해 높다. Wet/dry별 결과는 descriptive다.
- H2: A3 adapter/consistency가 A0보다 held-out query의 clean→perturbed 하락을 줄인다.
- H3 acquisition replication: Wang/BETA/Dong2023의 exact 40-class semantics에서 source-only validation 후 held-out acquisition의 A2−A0를 평가한다. Wang과 BETA만의 전이는 metadata 이득 식별이 아니라 boundary test다.
- H4: 사전지정 성공 threshold에서 A2/A3의 `learned − forgotten` OOD cell 수가 양수이고 worst-group 성능이 개선된다.
- H5 exploratory: stochastic auxiliary latent가 class보다 acquisition condition을 더 잘 나타낸다.

## 프로토콜 수정 기록

### 0.1 → 0.2

원래 0.1 계획은 Wang→BETA를 primary metadata 실험으로 두었다. 원 논문과 실제 MAT를 full run 전에 감사한 결과, Wang과 BETA는 SynAmps2, Cz reference, 배포 sampling rate, 64채널이 같고 electrode/cap 재질은 공개 근거가 없었다. 기존 `gel`/`wet_cap` 값도 근거 없는 동일 상수였다. 따라서 이 전이는 metadata 유용성을 식별할 수 없고 architecture 차이만 측정할 수 있었다.

연구 질문은 유지하되 primary를 source subjects에서 wet/dry와 impedance variation을 실제로 학습할 수 있는 wearable LOSO로 옮겼다. 이 수정은 102명 full data나 confirmatory A0/A2 결과를 보기 전에 이루어졌고, 당시 3명·1 epoch pilot은 두 모델 모두 chance 수준이었다. Wang↔BETA는 closed-set external-domain boundary test로 유지한다. dry→wet/wet→dry는 target category가 source vocabulary에서 `UNK`가 되는 level-extrapolation stress test이지, categorical wet/dry 의미 학습의 증거로 해석하지 않는다.

### 0.2 → 0.3

코드·문헌 peer review에서 `wearable_loso`가 실제 LOSO가 아니라 단일 60/20/20 participant holdout임을 확인했다. 약 20명의 test subject만으로 102명 전체를 대표한다고 쓰지 않도록, primary를 5-fold outer participant CV와 fold 내부 source-only validation으로 수정했다. `data.split_seed`는 optimization seed와 분리해 모든 A0/A2 seed가 같은 fold를 사용한다.

또한 Dong2023이 Wang/BETA와 같은 40개 frequency/phase semantics를 쓰면서 NeuSenW, 8채널, Fp1 reference, pre-gelled semi-dry electrode, 비차폐 환경이라는 실제 acquisition contrast를 제공함을 확인했다. 따라서 Dong2023을 acquisition-metadata external replication에 추가한다. 이 수정도 confirmatory A0/A2 결과를 보기 전에 이루어졌다. Nakanishi 12JFPM도 wearable과 exact 12-class semantics를 공유하지만 데이터 재사용 license가 명확하지 않아 허가 확인 전에는 확증 자산으로 쓰지 않는다.

### 0.3 → 0.4-lockbox

5-fold prediction의 participant row는 서로 다른 독립 모델에서 나온 관측이 아니다. 같은 fold participant는 같은 model shock를 공유하고 training folds도 크게 겹친다. 99개의 paired difference를 iid sign-flip/bootstrap하는 초기 설계를 target-free simulation으로 점검한 결과 fold ICC 0.05+overlap에서 null rejection 약 0.122, 결합 stress에서 약 0.204로 상승했다.

따라서 confirmatory cohort를 성능 미접근 상태에서 39명 training과 60명 lockbox로 고정했다. 세 seed `[42,43,44]`의 A0/A2를 39명 전체에서 validation·early stopping 없이 10 epoch 학습하고, 여섯 completion 뒤 60명을 한 번만 예측한다. 새 독립 lockbox 추론은 10,000-replicate simulation의 네 core DGP에서 Type-I·coverage·power acceptance 기준을 통과했다. 5-fold N=99 결과는 primary가 잠긴 뒤 exploratory로만 허용한다.

정확한 배정 계약은 `allocation_decision_id=DEC-20260830-002`, seed 42, `numpy_default_rng_shuffle_sorted_subject_ids_train_first_v1`, allocation SHA-256 `0c5b7cd1ca8eedafbe9290833a283160f7bfe89897be8923b21568f70bb18898`이다. ID 목록과 hash가 달라지면 plan·execution manifest 검증이 실패한다.

## 구현 게이트

실제 데이터 결과를 만들기 전에 모두 통과해야 한다.

- [x] source training partition에서만 categorical vocabulary를 만들고 unseen target 값은 trained `UNK=0`으로 처리
- [x] training-only categorical metadata dropout으로 `UNK` 학습
- [x] k=0/1/3/5가 동일한 fixed query를 사용하고 support와 query가 겹치지 않음
- [x] channel/robustness/calibration 평가가 checkpoint의 held-out split 또는 명시적 target filter만 사용
- [x] canonical correlation과 filter bank를 실제 계산하는 CCA/FBCCA 기준선
- [x] wearable dry↔wet에서 source/validation/test participant가 완전히 분리된 joint subject-condition split
- [x] subject-level metric과 prediction artifact에 split/query identity 기록
- [x] Primary A0/A2가 같은 `physical_hybrid_v1` module graph·parameter schema·초기 state를 쓰고 `external_metadata_mode=null|observed`만 다름
- [x] Protocol 0.4-dev의 `m_struct` backbone 공통 입력, pre-zscore query-local QC 공통 권한, external-only missing/shuffle 구현
- [x] wearable electrode type+block impedance global FiLM과 per-channel impedance bounded gain 구현; reference/cap은 상수라 제외
- [x] wearable impedance 조건축을 원자료 수치 signature에 맞춰 `[dry, wet]`로 수정하고 regression test 추가
- [x] 기존 wearable v2/checkpoint를 guard로 거부하고 revisioned `wearable_v3` 경로로 분리
- [x] S1–S3 `wearable_v3` 720행 raw→processed deep audit와 receipt 생성
- [x] 전체 102명 `wearable_v3` 생성과 24,480-row acceptance audit
- [x] Wang raw channel order, BETA schema variants, wearable target frequency/phase를 공식 자료와 대조
- [x] cross-dataset/cross-condition/standard prediction이 checkpoint `split.csv`의 test ID와 target filter를 교차 적용
- [x] optimization seed와 allocation seed 분리, exact 39-train/60-lockbox participant split
- [x] S1–S3 development-only와 N=99 primary cohort role/hash 고정
- [x] 미동결 confirmatory dry-run 차단, governed training의 test-loader 제거, evaluation action allowlist
- [x] exact A0/A2×3-seed six-job manifest, subject 내 seed-first 평균, N=60 단일 paired inference 집계기
- [x] no-validation fixed-epoch confirmatory training과 train-only preload; lockbox prediction 전 six-completion/reveal-once gate
- [x] confirmatory config→six-job manifest binding, 개별 prediction/eval 우회 차단, six-bundle atomic publication
- [x] completion receipt의 final/last checkpoint·train-metric·split·development-control hash와 prediction의 manifest/reveal-receipt hash를 aggregate까지 재검증
- [x] canonical manifest/run root를 plan에 결합하고 create-exclusive 생성·unused-draft 전용 교체 정책 적용
- [x] generic evaluation route와 manifest staging 밖 개별 prediction을 reveal 이후에도 차단
- [x] completion의 exact epoch/terminal/runtime/resume 계약 및 staging/final prediction bundle 재검증
- [x] target-free inference simulation plan/receipt/hash와 frozen-plan acceptance gate
- [x] RNG/DataLoader/terminal-state를 포함한 two-slot exact-resume 및 uninterrupted 동등성 시험
- [x] 잘못된 impedance 조건축을 바로잡고 numeric signature·headband-order 검증을 갖춘 `wearable_v3`를 구 revision과 분리
- [x] Dong2023 Zenodo 선택 다운로드, MD5/schema 검증 및 40-class canonical adapter
- [x] S1–S3 physical metadata-only와 acquisition-block-coherent shuffle-train→clean-val control; missingness-only는 single-pattern invalid assay
- [x] 정상 A2 validation-only wet↔dry counterfactual과 manifest-only donor 교환; donor EEG 미접근
- [x] control mapping/fixed-point/coverage/field-change hash와 evaluation parent-checkpoint provenance
- [x] execution tag `physical-reveal2-freeze-20260901-r1`에서 18/18 training, 3/3 intervention, reveal-budget precommit과 atomic publication 완료
- [x] potency-valid 10-gate 판정: substantive 4개 실패로 `diagnostic no-go`, confirmatory authorization false
- [x] 성공·dry-run·OOM 실패를 같은 schema로 남기는 synchronized CUDA peak-memory sidecar
- [x] physical development control equivalence/mechanism/safety margin, potency threshold, invalid-assay와 `development_gate_only` 정책 동결(confirmatory 전체 분석계획은 별도 미동결)

## 비교 모델

| 모델 | 역할 | Primary 여부 |
|---|---|---|
| CCA / generic FBCCA | training-free protocol-matched 기준선 | 필수; 실제 채널·filter parameter 기록 |
| paper-faithful FBCCA | Chebyshev-I와 문헌 고정 설정 | strict Chen-2015 구현 완료; 데이터셋별 원 toolbox 재현 범위는 별도 표기 |
| recent learned calibration-free baseline | 최신 직접 비교 | 필수 또는 재현 실패 사유 공개 |
| F0 backbone + linear head | backbone 자체의 효과; compact scratch를 깨끗한 primary anchor, REVE는 exposure/license 표시 secondary | 필수 |
| A0 external-metadata-null | A2와 같은 구조·parameter shape와 query QC, external treatment만 missing | Primary control |
| A1 dataset-ID-only | opaque ID 진단용 negative control | 보조 |
| A2 structured metadata | metadata 주효과 | Primary treatment |
| A3 A2 + adapter/consistency | robustness 가설 | Secondary |
| A4 A3 + stochastic latent | mechanism 탐색 | Exploratory |

A2 primary에는 `dataset_id`와 의미가 이전되지 않는 임의 hardware ID를 넣지 않는다. 채널 ID/mask와 time grid는 공통 구조, query scale은 공통 QC로 둔다. A2 treatment는 `electrode_type`, block impedance mean/max, channel별 impedance/availability다. Wearable에서 상수인 reference/cap은 제외한다. External missing과 acquisition-block-coherent derangement는 donor block의 exact label/window를 맞추며 mapping과 실제 변경률을 보존한다.

## 실험 순서

### Phase 0 — 코드 타당성

1. impedance 축 `[dry, wet]` 수정, 논문 평균 261.67/19.63 kΩ numeric-signature test, 기존 processed revision 폐기
2. P0 수정과 전체 단위 테스트·ruff
3. synthetic frequency 신호로 CCA/FBCCA 정답 검증
4. split·support·query·vocabulary artifact 검증
5. subject×condition×block의 EEG condition과 impedance 방향을 전처리 전후로 감사

### Phase 1 — synthetic pipeline pilot

1. synthetic asset 생성
2. Tiny backbone 1-epoch smoke
3. A0/A2 dry-run과 짧은 training
4. held-out channel/robustness 평가
5. fixed-query k=0/1/3/5 실행

Synthetic 결과는 사람 EEG에 대한 연구 증거가 아니라 pipeline 검증으로만 기록한다.

### Phase 2 — 공개 EEG pilot

1. BETA S1/S16, wearable S1–S3, Dong2023 S1로 서로 다른 raw schema·label·metadata smoke
2. 실제 CCA/FBCCA와 Tiny backbone A0/A2 실행
3. subject×electrode prediction/metric 산출 확인
4. 오류가 없을 때 Protocol 0.4 metadata gate로 이동

### Phase 2.5 — metadata protocol 0.4 동결

1. [완료] channel ID/mask 같은 `m_struct`는 A0와 A2 모두에게 제공하고 treatment에서 제외한다.
2. [완료] wearable에서 관측 가능한 electrode interface, block impedance mean/max, per-channel impedance의 schema·provenance를 확정하고 reference/cap 상수는 primary에서 제외한다.
3. [완료] `physical_hybrid_v1`의 공통 query-QC FiLM, external global FiLM, channel-quality gain과 exact-null routing을 구현한다. Prompt/adapter는 legacy secondary다.
4. [완료·outcome-free] six-role×3-fold 18-job과 **각 fold당 Full A2 intervention bundle 하나, 총 세 개**, prediction·intervention·aggregate·gate를 함께 tree-hash한다. Donor mapping/potency는 canonical manifest에서 재구성하고 private tree와 staging parent를 fsync한 뒤 digest precommit→atomic rename/fsync→final receipt로 공개한다. 알려진 private partial artifact만 같은 동결 계산으로 삭제·재생성하고, 미신고 prefix나 digest drift는 거부한다.
5. [완료] `DEC-20260901-004`의 validator-only 정정 뒤 동일 모델·grid·seed·fold·margin·gate로 reveal #2를 재승인·동결했다. Natural missingness-only는 사전 선언대로 `invalid_assay`다.
6. [완료·no-go] Clean annotated tag에서 18-job train과 단일 reveal을 완료했다. Potency는 통과했지만 clean direction·counterfactual reliance·inference pairing·training pairing gate가 실패했다. S1–S3는 계속 development-only이고 reveal #2는 소비됐으며 추가 tuning/reveal은 금지한다.

### Phase 2.6 — 데이터·문헌 충분성 gate

1. P0 impedance mapping 수정·전처리 재생성·numeric-signature 회귀 테스트를 완료
2. 모든 후보를 `core_exact12`, `core_exact40`, `shared_frequency_ood`, `controlled_context`, `representation_only`, `context_robustness`, `external_lockbox`, `excluded_or_quarantined` 중 하나로 registry에 고정
3. `(frequency_hz, phase_rad, target_semantics, stimulus_method, display_context)` class key를 원 논문·versioned archive·실제 event 파일로 대조
4. dataset version·license·usable participant·checksum·metadata provenance와 source/derivative 변환 이력을 기록
5. backbone revision별 raw/same-subject/label exposure matrix를 만들고 target-exposed foundation 결과를 별도 표기
6. Nakanishi 원 repo s1–s10의 reuse 권리를 서면 확인한다. Kim2025는 frequency 10개만 겹치고 phase가 모두 π만큼 달라 exact class overlap이 0이므로 기존 `shared-10 lockbox` 역할을 폐기하고 별도 frequency-only protocol로 재지정
7. 직접 baseline 감사 결과를 적용: DG-Conformer는 permission/port 조건부, TST-CSFR는 literature-only, TFA-Net public runner는 제외, TBMSCCN-C는 독립 구현만, SSER은 training-only augmentation ablation으로 사용
8. [현 candidate 실패] Source-only gate에서 safety는 통과했지만 `correct > shuffled`와 `correct > wrong`의 기전 민감도를 보이지 못했다. 따라서 현 candidate로 Phase 3에 진입하지 않는다. 새 독립 data/model study가 생기면 이 gate를 결과 전에 다시 정의해야 한다.

### Phase 3 — confirmatory full runs

상태: **현재 계약에서 unauthorized/blocked.** Phase 2.5의 substantive gate가 실패했으므로 아래 항목은 실행 절차 기록일 뿐 현 candidate에 대한 작업 목록이 아니다. 새 독립 development 근거와 별도 사전등록 계약이 승인·통과되기 전에는 Protocol 0.4를 `frozen`으로 선언하지 않는다.

1. 사전 고정된 39명 전체로 parameter-matched A0/A2를 seeds `[42,43,44]`, 10 epoch, validation 없이 학습한다.
2. 여섯 final checkpoint와 completion receipt가 모두 유효한지 검사한 뒤, 별도 60명 lockbox의 clean k=0 prediction을 첫 confirmatory 공개에서 같은 manifest 아래 함께 생성한다. 각 prediction은 staging에 머물다가 6/6 완성 뒤 원자적으로 공개한다.
3. overall A2−A0만 primary로 검정하고 wearable condition별 및 worst-subject 결과는 descriptive로 보고한다.
4. primary report를 잠근 뒤 wearable 99명 cross-validation, dry→wet/wet→dry, A3/A4, k>0를 exploratory/secondary로 실행한다.
5. Wang/BETA/Dong2023 leave-one-acquisition-out replication을 별도 수행한다. Wang↔BETA만의 전이는 boundary test다.
6. 추가 shared-label multi-source dataset을 확보하면 leave-one-dataset-out metadata 실험을 수행한다.

## 통계와 판정

- 독립 단위: training과 model selection에서 완전히 제외된 N=60 lockbox subject.
- Primary endpoint: subject-level balanced accuracy의 paired A2−A0
- Optimization seed: `[42,43,44]`. 동일 lockbox subject×role의 seed BA를 먼저 평균하며 seed를 독립 피험자로 세지 않는다.
- Allocation seed: 42, `numpy_default_rng_shuffle_sorted_subject_ids_train_first_v1`. 39/60 exact ID는 plan에 명시한다.
- 보고: mean paired effect, Student-t one-sided lower confidence bound, paired subject sign-flip p-value, bootstrap-t sensitivity.
- 다중 비교: overall A2−A0 하나만 primary이므로 correction 없음. Condition 결과는 descriptive다.
- 누락: exact `60 subjects×12 classes×10 trials/condition×2 conditions×3 seeds×2 roles = 86,400` prediction rows를 요구한다. 일부 seed/subject가 복구 불가하면 primary를 `invalid`로 종료하고 complete-case claim을 만들지 않는다.
- capability 분석 단위: subject×condition×window×severity cell
- 성공 threshold와 MCID: target 결과를 보기 전에 확정
- 현재 query에서 계산한 mean/std·signal-QC는 A0/A2 모두에 같은 권한으로 제공하며, target participant의 다른 trial이나 target-set 통계는 양쪽 모두 금지한다.
- Metadata-only의 chance 판정은 단순 `p>0.05`가 아니라 사전 equivalence margin으로 한다. Missingness-only는 현재 natural pattern 하나라 invalid assay다. Block-coherent shuffle과 counterfactual은 label-aligned development diagnostic이며 deployable inference로 부르지 않는다.
- 전체 metadata mechanism 주장은 primary A2−A0 성공, shortcut control 유효성, correct−shuffle, correct−wrong, wrong-metadata safety gate를 모두 만족할 때만 허용한다. `failed`, `inconclusive`, `invalid_assay`를 구분한다.

평균만 개선되면 metadata efficiency로 해석한다. `learned > forgotten`, worst-group 개선, 양방향 재현까지 충족해야 OOD 적용범위 확장으로 해석한다.

## 필수 산출물

- 실행 config와 Git commit
- source/target dataset 및 license/version
- `split.csv`, vocabulary, class map
- fixed support/query manifest와 hash
- seed별 checkpoint 및 subject-level predictions
- 환경, hardware, wall time, parameter count
- 실패·제외 run과 사유
- aggregate 표와 learned/forgotten transition 표
- prediction CSV/sidecar/checkpoint/train-metric/split/analysis-manifest/execution-manifest/reveal-receipt/source/environment hash ledger와 frozen analysis-plan hash

## 중단 규칙

- target test로 checkpoint·threshold·hyperparameter를 고르면 해당 run은 confirmatory 결과에서 제외한다.
- A2가 A0를 이기지 못하면 A4 결과로 metadata 가설을 대체하지 않는다.
- 한 전이 방향에서만 효과가 있으면 dataset-specific 결과로 제한한다.
- split/query/vocabulary provenance가 없는 run은 재실행한다.

## 2026-08-30 실행 현황

- 전체 단위 테스트: 전용 환경의 최신 수치는 최종 연구일지 검증 항목을 따른다. 신규 statistical/governance/exact-resume/manifest/spectral 경로를 포함한다. 저장소 전체 Ruff에는 기존 범위의 style debt가 남아 있어 full-clean으로 주장하지 않는다.
- synthetic 640-trial pipeline: A0/A2 training, held-out robustness, fixed-query k=0/1/3 artifact와 동일 query hash 검증
- BETA S1/S16 pilot: `(320, 64, 400)`, subject당 160 trials, class당 4 blocks, 750/1000-sample 원본 schema 모두 검증
- BETA S1/S16 posterior-9, H=5 pilot: CCA BA 0.8656, generic Butterworth FBCCA BA 0.8719. 개발 sanity일 뿐이며 paper-faithful FBCCA는 아직 아니다.
- wearable S1–S3 v2 pilot: `(720, 64, 400)`, subject당 240 trials, wet/dry×class당 10 blocks. 후속 원자료 수치 감사에서 impedance mapping이 반대임을 확인했으므로 이 historical artifact와 결과는 무효다. 바로 아래 v3 deep smoke로 재전처리를 완료했다.
- wearable S1–S3 v3 deep smoke: 720 trials를 최종 parser로 재생성했고 raw EEG, pre-zscore query-QC vector, raw→canonical impedance vector의 최대 절대오차가 모두 0이었다. Figshare v4 5개 파일 MD5와 artifact receipt도 검증했다. 세 subject가 모두 dry-first이던 pilot 한계는 아래 full audit의 wet-first 49명 확인으로 해소했다.
- wearable full v3 acceptance: 102명·24,480 trials, dry/wet 각 12,240행, dry-first/wet-first 53/49를 확인했다. Raw EEG→processed signal, query-QC, per-channel impedance 전수 대조의 최대 절대오차가 모두 0이었고 receipt SHA-256은 `1e2c10d7922c329d8d28cd8bd3cb8ed94e0b22f6a2f3432e95c77deab3f0bc9b`다.
- wearable S1–S3 official native-8, H=5 pilot: CCA BA 0.6125( dry 0.4667 / wet 0.7583), generic Butterworth FBCCA BA 0.5750(dry 0.4194 / wet 0.7306). 3명 결과이므로 성능 결론이 아니라 pipeline sanity check다.
- wearable v1 pilot에서 class frequency 순서 오류 때문에 CCA/FBCCA가 chance 수준이었고, 공식 stimulation table 순서로 고친 v2에서 회복했다. v1과 그 파생 결과는 연구 결과에서 제외한다.
- parameter-matched wearable S1–S3 Tiny 1-epoch pilot: A0/A2 모두 40,860 parameters, 동일 split hash, test BA 0.0917. 1 test subject·chance 수준이고 A2 impedance도 반대로 결합돼 있어 효과 증거가 아니다. split/training smoke 외의 연구 결과에서는 제외한다.
- 역사적 Protocol 0.4-dev prompt dry-run: A0/A2 모두 총/학습 parameter 6,125,900이고 fairness, parameter-schema, initial-trainable-state, split hash가 동일했다. 새 physical architecture의 증거는 아니다.
- DEC-20260830-001 governed CUDA dry-run: S1–S3 development view에서 A0/A2 모두 cohort hash와 parameter/runtime contract가 같았고 outer-test access는 false였다. N=99 dataset access는 분석계획이 `dev_not_frozen`인 동안 dry-run도 차단된다.
- CUDA runtime: 프로젝트 전용 Python 3.10.12/Torch 2.2.2+cu121 환경에서 CUDA 12.1, RTX 4090 forward/backward·finite-gradient probe를 통과했다.
- Dong2023 S1 pilot: 공식 MD5와 `[8,1250,40,4]` schema, 160 trials를 검증했다. native-8 H=5에서 CCA BA 0.7813, generic FBCCA BA 0.7563이었다.
- Wang 28명 train/7명 validation source-learning: query-local complex spectral Transformer BA 0.8220, pure F0 spectral BA 0.8107. 동일 7명·1,680-trial scope의 CCA BA 0.7875, strict Chen-2015 FBCCA BA 0.7768이었다. 모두 development/legacy evidence이며 metadata 효과는 아니다.
- 독립 60명 lockbox inference simulation: 10,000 outer replicates와 10,000 sign-flip/bootstrap resamples에서 네 core DGP가 Type-I upper ≤0.0581, coverage lower ≥0.9413, SESOI+0.03 power lower ≥0.8769를 통과했다. Target EEG outcome은 접근하지 않았다.
- S1–S3 fixed-epoch outcome-gated **legacy `prompt_adapter_v1`** grid: 이 grid 안에서는 6/6 training이 끝날 때까지 결과를 보류한 뒤 prediction을 함께 공개했다. A0/A2 mean BA는 0.1222/0.1042, mean delta −0.0181이었다. 세 subject delta가 모두 음수지만 N=3·각 fold train 2명이라 모집단 결론은 불가능하다. Formal cutoff가 없는 방향성 경고와 `metadata benefit not demonstrated`로 두며, 새 physical A2의 결과로 재해석하지 않고 confirmatory는 계속 봉인한다.

Raw-access 상태도 분리한다. 전체 102명 raw/processed signal은 무결성 감사에서 읽혔다. 그러나 S4–S102는 모델 학습·선택·성능 계산에 쓰지 않았고, confirmatory 39명 학습과 60명 lockbox prediction은 아직 실행하지 않았다. 현재 봉인은 OS-level ACL이나 암호학적 enclave가 아니라 same-user raw access가 가능한 application/procedural gate다.

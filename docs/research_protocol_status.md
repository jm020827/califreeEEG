# Calibration-Efficient SSVEP 연구 프로토콜 현재 상태

기준일: 2026-09-06
현재 상태: **V2 synthetic V11 단회 소비 / infrastructure inconclusive / scientific result 없음 / external·held 금지**
현재 방법의 불변 pre-outcome 기준원은 `configs/analysis/metadata_calibration_efficiency_v2.yaml`과 frozen V11 plan SHA-256 `9a9683141ccd3d1fe9cf094aad955c6e317d112a73e35fbcfcdd2ef7d06e1f8c`이고, 실행·종료 상태 기준원은 [V2 terminal audit](../configs/governance/metadata_calibration_v2_synthetic_v11_terminal_audit.json)과 [V2 terminal 결과](metadata_calibration_efficiency_v2_results.md)다. V1 source no-go, held 60명 미개봉, `DEC-20260903-005`의 physical 후보 retirement와 terminal `reliability-spatial-v1` Stage-0도 그대로 유효하다.

이 문서는 현재 방향의 요약본이다. V1의 쉬운 전체 설계는 [metadata-assisted low-calibration 설계](metadata_calibration_efficiency_design.md), V1 수치·해석은 [source 결과](metadata_calibration_efficiency_results.md), V2 방법은 [V2 설계](metadata_calibration_efficiency_v2_design.md), 세부 이력은 [append-only 연구일지](research_log.md)를 따른다. V2 terminal은 scientific negative result가 아니다. 과거 physical, synthetic, query-only plan과 결과도 삭제하거나 새 후보의 양성 근거로 재해석하지 않는다.

## 연구목표

상위 응용목표는 **처음 보는 사용자가 유용한 closed-set SSVEP 성능을 얻는 데 필요한 labeled target calibration을 최소화하는 것**이다. `k=0`은 calibration-free anchor이고 `k>0`은 명시적인 low-calibration regime이다. 현재 연구 질문은 다음과 같다.

> deployment interface당 동일한 완전 calibration block과 동일한 고정 query에서, query 전에 관측한 wet/dry interface와 block별 채널 impedance가 EEG·구조·signal-derived QC만 쓰는 강한 기준보다 `k=0/1/3` early-budget curve를 개선하는가?

Metadata 자체가 목적이 아니다. 현재 처리 가능한 과학적 treatment는 pre-query acquisition context뿐이다. Dataset/subject ID와 query-derived QC를 metadata 이득으로 세지 않는다. 새 class discovery, open-set/OOD 탐지, 손상 EEG 복원과 LLM test-time scaling은 primary endpoint가 아니다.

## V2 synthetic gate 현재 상태

V2는 V1의 질문을 바꾸지 않고 `A_Q` common path를 strict FBCCA anchor 위의 안전한
support update로 고쳤으며, `A_QM`에는 query-support acquisition-context pairing만 작은
추가 보정으로 허용했다. 사람 EEG outcome을 보기 전에 7개 beneficial/null/adversarial
family × 64 synthetic participants에서 도움, correct-vs-shuffle pairing, adversarial
abstention, null 비열등성과 severe-harm를 동시에 검정하도록 사전 동결했다.

V11은 commit `832e579c0dec8774ffe8c3ac99abf110a74aa9e3`, tree
`d26656acd69c88e62cc5dd336103ddd76f33022f`, tag
`metadata-calibration-efficiency-v2-synthetic-freeze-20260906-r11`에서 전체 `682 passed`와
독립 `CODE-GO` 뒤 한 번 실행했다. Development replay 5,888 rows/19 contrasts는
`engineering_only`였고 lockbox 판정에 합치지 않았다. Exact NIST target pulse에 대한 global
claim과 beacon receipt까지 정상 생성했지만, post-beacon evidence validator가 authorization→
development→reserved-seed→beacon으로 순환해 efficacy participant 생성 전에
`RecursionError` terminal이 됐다. Scientific result 파일은 생성되지 않았다.

따라서 V2 P1/P2/AQM은 통과도 실패도 입증되지 않았다. Frozen 계약상 claim 이후 인프라
오류도 lockbox를 소비하므로 같은 V11 재시도, BETA/Dong selection·independent outcome,
Choi replication과 wearable held 60 outcome은 금지한다. 후속 연구는 validator 수정만으로
자동 재시도할 수 없고, 새 candidate/schema·새 seed·명시적 pre-outcome amendment·새 clean
freeze·새 future beacon을 갖춘 별도 루프여야 한다.

## 완료된 v1 설계와 결과

새 후보 `metadata-calibration-efficiency-v1`은 A_Q와 A_QM을 primary pair로 둔다. 먼저
spectral backbone과 Q-only calibration estimator를 학습해 고정하고, 그 공통 경로에는 M
gradient/statistic이 들어가지 않게 한 뒤 metadata residual만 source episode에서 fit한다.
A_QM만 이 residual을 target-support estimator의 positive-definite precision 또는 bounded
shrinkage prior로 사용한다. Query waveform을 직접 변환하거나 채널을 섞고 decoder 전체에
metadata prompt를 넣지 않는다. Residual off/all-missing은 같은 composite checkpoint의 frozen
A_Q 경로와 exact 같아야 한다.

Wearable의 한 block은 12개 class를 한 번씩 포함한다. 각 interface의 block 1–5가 nested
support이고 block 6–10은 모든 k와 역할에 공통인 immutable query다. Primary는 participant별
`eAUC=Y0/6+Y1/2+Y3/3`의 `A_QM−A_Q` 차이다. k=5는 saturation secondary다. H1은 metadata가
early-budget curve를 개선하는지 묻고, `H0: population mean≤0`인 one-sided alpha .05 검정과
관측 평균 `≥0.020` screen을 함께 요구한다. H1이 통과한 뒤 H2a도 같은 형태로
`A_Q(k3)−A_Q(k1)>0`과 관측 평균 `≥0.020`을 요구하고, H2b에서 A_QM(k1)이
A_Q(k3)에 margin 1/60으로 비열등해야만
equal-weight wet/dry estimand에서 interface별 보정 schedule을 36→12 trials로 줄였다고
주장한다. 선택한 interface의 support 양은 24 trials/two blocks 줄지만 wet과 dry 각각의
비열등성을 별도로 증명한 것은 아니다. Wet/dry 연구 평가 전체의 support 접근은 k=1/3/5에서
participant당 24/72/120 trial이다.
H1의 관측 평균 `≥0.020`은 실용성 screen이며 모집단 평균이 0.020 이상이라는 검정은 아니다.
따라서 H1만으로는 양의 metadata 순증분만, H2까지 통과해야 유용한 calibration saving을
주장한다.

정확한 metadata-prior 수식, 39명 수치 gate, SESOI·savings margin, H1/H2 reducer, N=39/60
200,000-draw simulation receipt, mandatory baseline method/core와 label-free query-adapter code,
private support/score validator, label-free query collator, 불변 cell dispatch와 phase/token/EEG
base-input에 결속된 실제 intervention context tensor resolver/usage hash, resolver 출력을
우회할 수 없는 governed `forward_cell_from_precomputed`, privileged input sealer, job capability,
Bubblewrap worker, producer/finalizer receipt와 full lifecycle이 구현됐다. 현재 exact contract는
source `candidate 9 + baseline 15 = 24`, held `candidate 3 + baseline 5 = 8` jobs다. Episode cache는
spectral embedding과 Q를 재사용하지만 raw M은 보관하지 않으며 Q-off cell에는 exact zero만
전달했다. Power receipt, 전체 536 tests와 clean annotated tag
`metadata-calibration-efficiency-v1-freeze-20260905-r1`를 봉인한 뒤 source 24 jobs를
실행했다. A_Q k5 `0.476282`, A_QM−A_Q eAUC `−0.004843`, correct−shuffle
`0`으로 hard gate가 실패했다.

권한은 순환시키지 않고 각 phase에서 따로 반복한다. Source seal decision/receipt 뒤 source
outcome decision을 만들고, immutable 39명 gate가 PASS한 뒤에만 held target 60명과 held-domain
source-refit 39명을 별도 seal한다. 그 뒤 held outcome decision을 새로 만든다. Seal 단계는
training·score·metric·query-label 공개를 금지한다. 최신 owner directive가 두 phase의 조건부
체인을 승인했고 source phase를 정상 완료했다. Source gate가 FAIL이어서 held
phase의 seal, claim, directory는 생성하지 않았다.

## Outcome 전에 철회된 query-only 후보

`query-reliability-spatial-v1`은 사람 training이나 prediction을 한 적이 없으므로 실패 결과가
아니다. 구현·contract·BETA asset hash와 CUDA forward 증거는 frozen baseline 후보로
보존한다. 다만 `DEC-20260904-001`의 BETA 35/20 primary 방향은 `DEC-20260904-005`가
supersede하며 그 outcome plan은 실행하지 않는다.

## 종료된 metadata 합성 선행실험을 가장 쉽게 설명하면

각 EEG 채널을 한 명의 관측자라고 생각한다. 현재 2초 EEG에서 계산한 label-free 품질 단서
`Q`가 주 판단이고, query 전에 측정한 impedance/interface `M`은 작은 보정만 허용한다. 네
모델 `A0`, `A_M`, `A_Q`, `A_QM`은 Q와 M의 접근 권한만 다르고 나머지는 모두 같다.

별도 합성자료의 생성 truth를 이용해 held-out participant에서도 M이 Q가 이미 아는 품질정보
너머의 순증분을 갖는지 Stage-0에서 먼저 확인했다. Q-only 품질 예측은 높았지만 M의 추가
이득은 8명 중 5명에서만 양수였고, query 안 채널 품질 순위의 평균 증분도 거의 0이었다.
따라서 사전 gate는 실패했다. 합성 truth는 probe 분석에만 쓰였고 decoder 입력에는 들어가지
않았다. 조건부 CUDA decoding Stage-1은 실행하지 않았으며, simulator·feature·threshold를
결과에 맞춰 바꾸어 같은 후보를 다시 통과시키는 것도 금지한다.

## 봉인된 선행 실험 설명 (historical physical_hybrid_v1)

두 모델은 같은 39명 EEG, 같은 backbone, 같은 parameter shape, 같은 세 optimization seed와 정확히 10 epoch를 쓴다.

- A0: EEG, 채널 ID/mask, sampling/time grid, query-local pre-zscore scale QC를 사용한다. Physical external branch는 동일하게 보유하지만 global residual 0·channel gain 1로 강제한다.
- A2: A0와 동일하며 electrode type, block impedance mean/max, channel별 impedance와 availability만 추가로 관측한다. Global 정보는 zero-init bounded FiLM, channel 정보는 shared bounded gain으로 넣는다.
- 선택: validation이나 early stopping 없이 세 seed의 final checkpoint를 고정한다.
- 시험: 모델 학습·선택·성능 평가에 쓰지 않은 60명에게 두 ensemble을 첫 confirmatory 공개에서 함께 적용한다. 원시 신호 byte는 전체 자산 무결성 감사에서 이미 읽혔지만 성능이나 모델 선택에는 쓰지 않았다.
- primary 값: 각 lockbox participant에서 세 seed BA를 먼저 평균한 뒤 `BA(A2) - BA(A0)`를 계산하고, 60개의 paired difference 평균을 검정한다.

Reference/cap은 wearable에서 상수라 당시 primary 입력에서 제외했다. Stable role 이름 `A2_structured_condition_prompt`는 남았지만 당시 physical primary는 prompt token/adapter를 쓰지 않았다. Dry/wet별 결과, worst participant, learned/forgotten은 설명적 진단이었다. 당시 계획의 단 하나 confirmatory contrast는 전체 `A2−A0` 평균이었으며 실행되지 않았고 현재 primary가 아니다. 정확한 역사적 모델 계약은 [A2 physical-hybrid 설계](a2_physical_hybrid_design.md)에 있다.

## 데이터 역할과 정보권한

| 역할 | 피험자 | 행 | 지금 가능한 작업 | 금지된 작업 |
|---|---:|---:|---|---|
| 전체 자산 감사 | 102명 | 24,480 | checksum, schema, raw↔processed, metadata 품질 감사 | 성능 기반 선택 |
| development | S1–S3 | 720 | 역사적 prompt 6-job과 physical six-role×3-fold 18-job·3 intervention bundle 공개 완료; reveal #2 소비, valid diagnostic no-go | 모집단 효과 주장, S1–S3 재튜닝·추가 reveal |
| synthetic engineering | 합성 8명 | 256 | terminal Stage-0 artifact의 read-only 무결성·해석 감사 | Stage-0 재실행/덮어쓰기, Stage-1, human EEG·population·architecture superiority 주장 |
| V2 synthetic lockbox | 7 families × 64명 | scientific rows 0 | terminal·claim·beacon 무결성 감사; development 5,888 rows는 engineering-only | V11 재실행, synthetic 효능 PASS/FAIL 주장, external·human outcome 진입 |
| source development | 사전 추출한 S4–S102 중 39명 | 9,360 | exact 24 jobs·4,680 query 완료; development no-go | 부분 grid, 결과 기반 재튜닝·same-cohort 재시도 |
| held-participant evaluation | 나머지 60명 | 14,400 | 미개봉 보존; source FAIL로 seal/claim 0 | 현 v1으로 접근, 부분 reveal, 재시도 |
| BETA query-only historical split | S16–S70의 35/20 | 8,800 | 기존 hash·forward 증거 보존 | 철회된 35/20 outcome plan 실행 |
| Wang/BETA/Dong | 164명 | 29,040 | signal-only calibration baseline과 protocol audit | corpus ID 차이를 metadata factor 효과로 해석 |

39/60 ID는 `DEC-20260830-002`, `allocation_seed=42`, 정렬한 S4–S102를 NumPy
`default_rng`로 한 번 shuffle한 outcome-blind allocation이다. Canonical allocation SHA-256은
`0c5b7cd1ca8eedafbe9290833a283160f7bfe89897be8923b21568f70bb18898`다. V1은 이
분할을 exact하게 사용했고, source 39명의 outcome은 이제 한 번 공개됐다. Held 60명은
계속 미개봉이다.
k>0에서 60명의 support label은 method input이므로 strict-k0 lockbox가 아니라 sealed
within-participant calibration partition이라고 부른다.

## V1에서 39/60 allocation을 선택한 이유와 현재 한계

원래 N=99 5-fold 계획에서는 같은 fold의 약 20명이 같은 모델을 공유하고, 다섯 모델의 training set도 크게 겹쳤다. 그런데 피험자 99명의 차이를 독립 표본처럼 sign-flip/bootstrap하면 이 model-shared 오차를 무시한다.

Target outcome을 전혀 읽지 않은 과거 physical strict-k0용 synthetic stress simulation에서 nominal alpha 0.05의 rejection rate는 독립 조건에서는 허용 범위였지만, fold ICC 0.05와 training-overlap이 있으면 약 0.122, 결합 stress에서는 약 0.204까지 상승했다. 따라서 그 5-fold primary inference는 폐기했다. 당시 60명 독립 lockbox 설계의 10,000-replicate simulation은 네 core DGP에서 Type-I upper bound ≤0.0581, one-sided coverage lower bound ≥0.9413, SESOI+0.03 power lower bound ≥0.8769로 사전 기준을 통과했다. 그 receipt는 새 설계의 근거가 아니다. 새 plan은 별도의 200,000-draw receipt를 만들었고, H1 효과 0.030·Q-only k3−k1 0.030·AQM k1−AQ k3 0.000이라는 계획 중심값에서 H1 통과는 약 0.903이다. 두 endpoint BA floor≥0.50을 충족한다고 가정한 held-core contrast sensitivity에서 H1 뒤 H2 intersection까지는 약 0.657이며, Shared-AQ3/adverse 상관, t5와 20% harmed에서도 약 0.656–0.661이었다. 같은 가정과 계획값에서 N=83이면 약 0.805여서 H2까지 80%를 원하면 현재 held 60명보다 약 23명이 더 필요하다. H2a/H2b를 각각 경계에, 나머지를 유한한 대립값에 둔 configured partial-null full-claim은 약 0.0044/0.0137이고 local component는 약 0.0049/0.0244다. Strong control은 후자의 component test와 intersection-union 구조에서 온다. Contrast 효과 0·정상 baseline 0.55인 development metadata-null의 hard false-go는 0.00953이며, baseline도 0.48인 joint-bad null의 0.00021과 구분한다. 이는 사람 EEG 결과가 아니라 outcome-free 설계 민감도다.

현 V1은 outcome을 보지 않고 만든 이 allocation을 바꾸지 않고 실행했다.
사전 plan은 exact 39/60 ID, 기존 allocation digest와 source/held complete-block query
identity digest까지 결속한다. 사전 고정한 normal global-null DGP의 full claim은
200,000회에서 0회였고 rule-of-three 95% 상한은 약 `1.5e-5`지만, 이는 해당 simulation에만
해당하며 보편 FWER 상한이 아니다. 계획 대립값에서 development hard gate의 false-no-go는
normal/t5/20%-harmed에서 약 0.216/0.210/0.214다. Owner는 최신 위임 지시로 현재 N=60의
full-H2 계획 민감도 약 0.657과 그 한계를 수용했다. 약 0.80을 원했다면 약 83명용 새
allocation·plan·power receipt가 필요했을 것이다. 단, 이전 `physical_hybrid_v1` 실행 권한이 되살아나는 것은
아니다. V1 source outcome을 본 현재, 같은 39명은 V2의 unbiased promotion cohort가 될 수
없다. 이 원칙으로 별도 V2 V11을 동결했지만 그 synthetic 실행은 infrastructure
inconclusive로 소비됐다. 앞으로도 새 방법·support/query·estimand·수치 gate와 독립
development cohort를 다시 사전 동결·검증한 별도 후보만 미개봉 60명의 사용 여부를 논의할
수 있으며, 현재는 그 권한이 없다.

## 현재 모델 방향과 근거

- 종료된 V2 V11의 방법 concept는 metadata를 decoder-wide prompt나 query-spatial transform이 아니라 **frozen Q-only calibration estimator 위의 nested precision/shrinkage residual**로 제한했다. Bounded diagonal-Gaussian 수식과 Q 0.25–4배, M 0.8–1.25배, absolute precision 0.05–20의 범위는 구현·계약 검증됐지만 efficacy는 평가되지 않았다.
- Primary pair는 한 composite source checkpoint의 A_Q residual-off와 A_QM residual-on이다. 공통 Q path를 먼저 고정하므로 M fit이 A_Q weights에 영향을 주지 않고, metadata missing 시 그 frozen A_Q path와 exact 같아야 한다.
- `query_reliability_spatial_v1`은 waveform 앞에서 `X'=(I+ΔQ)X`를 적용하는 outcome-free frozen baseline 후보다.
- 종료된 `reliability_spatial_v1`과 `physical_hybrid_v1`의 model, threshold나 S1–S3 outcome을 새 후보 tuning에 쓰지 않는다.
- Tiny time-domain Transformer는 synthetic 12-class에는 학습됐지만 Wang subject-independent source validation에서는 3 epoch 동안 chance였고 one-batch 자체 기준도 통과하지 못했다.
- Query-local complex RFFT의 real/imaginary/log-magnitude를 쓰는 spectral Transformer는 동일 Wang 7명 validation에서 10 epoch BA 0.8220을 보였다.
- 같은 7명·1,680 trial에서 CCA BA 0.7875, strict Chen-2015 FBCCA BA 0.7768이었다.
- pure F0 spectral run도 BA 0.8107이어서 metadata 모듈 없이 backbone 자체가 학습 가능함을 확인했다.

이 수치는 모두 legacy/development evidence이며 metadata나 low-calibration 효과가 아니다.
Few-shot SSVEP-DAN, one-shot CSDuDoFN/OS-SSVEP, LST/eTRCA/TDCA 계열은 직접 선행이다.
CSDuDoFN과 peer-reviewed OS-SSVEP는 같은 lineage라 한 comparator로 센다. SSVEP-DAN 원법은
class당 최소 2 trial이 필요해 현재 grid에서는 k=3/5만 호환된다. 현재 mandatory
direct transfer comparator는 Chiang-2021 LST+filter-bank eTRCA이며, TDCA는 optional이다. 새 기여는
few-shot 자체가 아니라 complete-block curve에서 external acquisition context의 순증분과
correct/shuffled/missing dependence를 분리하는 데 있다.

## 봉인된 physical A2의 Metadata 권한

| 묶음 | 예 | A0 | A2 |
|---|---|---:|---:|
| 공통 구조 | canonical channel ID/mask, sampling/time grid | 사용 | 사용 |
| 공통 query-local QC | 현재 trial의 pre-zscore channel std | 사용 | 사용 |
| 외부 global treatment | electrode type, pre-query impedance mean/max | algebraic null | observed |
| 외부 channel treatment | canonical channel별 pre-query impedance/availability | gain 1 | observed bounded gain |
| 상수라 primary 제외 | reference, cap type | 금지 | 금지 |
| 분석 covariate | headband order, condition period, block number | 모델 입력 금지; v1 descriptive balance/design-invariant audit only | 모델 입력 금지; confirmatory adjustment 없음 |
| 금지 proxy | subject/session/trial ID, label/frequency/phase, source file | 금지 | 금지 |

Target participant의 다른 trial, target batch 통계, target 기반 normalization/adaptation, 정답 frequency/phase는 strict k=0에서 금지한다. 현재 protocol에는 candidate frequency codebook도 넣지 않는다.

## 구현된 보호장치

- `physical_hybrid_v1`은 A0/A2가 같은 module graph·parameter schema·initial state를 갖고 mode만 null/observed로 바꾼다. FiLM과 channel score의 마지막 층은 zero-init이며 null/all-missing/inactive channel은 학습 뒤에도 exact no-op이다.
- Channel gain은 spectral signal projection 뒤·channel identity embedding 앞에만 적용한다. Unknown categorical은 fixed-neutral 0이고, 높은 impedance를 나쁘다고 hard-code하지 않는다.
- External metadata shuffle은 row별 mosaic가 아니라 subject×session×electrode×run acquisition block을 derange하고 exact label/window로 donor를 맞춘다. 공개 validator는 producer의 scope 문자열만 믿지 않고 canonical asset manifest에서 shuffle/counterfactual mapping과 changed/flip potency를 다시 만든다.
- Physical mechanism grid는 A0/full/global-only/channel-only/shuffle-train/metadata-only의 exact 18 jobs다. Shuffle role은 donor metadata로 train하고 correct metadata의 clean validation으로 평가한다. 자연 missingness pattern 하나인 missingness-only는 invalid assay다.
- Full A2의 all/global/channel/query-QC missing, block shuffle, wet↔dry counterfactual은 **각 fold당 intervention bundle 하나, 총 세 개**로 사전 고정됐고, 18개 clean prediction과 함께 hidden staging에서 검증된 뒤 하나의 directory rename으로 공개됐다.
- 첫 reveal receipt를 tracked immutable ledger로 묶고 새 grid를 canonical 전역 ledger의 reveal #2로 고정했다. 18개 prediction·3개 intervention·aggregate·10-gate 결과와 file-tree hash를 완성하고 private tree의 파일·하위 디렉터리와 staging parent를 fsync한 뒤, ledger precommit과 precommit receipt로 reveal 예산을 소비했다. 이후 directory atomic rename·parent fsync와 final publication receipt 검증까지 완료했다. Reveal #2의 재실행, 과거 v1 mutable 재생성, CLI protected-field 변경, generic training/evaluation 우회는 코드에서 거부한다.
- 미동결 confirmatory는 dry-run도 dataset 생성 전에 차단한다.
- 39명 training split과 60명 lockbox split을 exact ID로 검증한다.
- confirmatory training은 validation/test loader 없이 fixed 10 epoch final checkpoint만 만든다.
- exact resume는 model/optimizer/scaler/Python·NumPy·Torch·CUDA RNG/DataLoader state와 terminal reason을 two-slot atomic journal에 저장한다.
- 실행 manifest의 정확한 6 jobs, allocation, job ID, role, seed, fairness hash, resolved config hash, execution-contract hash를 train→checkpoint→prediction→aggregate까지 연결한다.
- 분석계획이 canonical manifest `outputs/confirmatory-primary/execution_manifest.json`과 run root `outputs/confirmatory-primary/runs`를 고정한다. 다른 경로의 두 번째 manifest/run tree는 생성·학습·예측·집계에서 거부하며, canonical 파일은 최초 생성 후 덮어쓰지 않는다. 유일한 예외는 실행·예측·reveal 산출물이 전혀 없는 unauthorized dev draft를 명시적으로 한 번 교체하는 경우다.
- 여섯 training completion과 final/last checkpoint·train metric·split·development-control hash뿐 아니라 exact 10 epoch, `max_epochs`, runtime-contract hash, exact-resume generation/state가 모두 유효해야 첫 confirmatory lockbox reveal receipt를 만들 수 있다. 개별 `predict.py`와 generic `evaluate.py` 경로는 reveal receipt가 생긴 뒤에도 dataset 생성 전에 실패한다.
- 6개 prediction은 manifest가 정한 숨김 staging 경로에만 생성된 뒤 디렉터리 단위로 공개한다. 재시작 staging과 이미 공개된 final bundle도 CSV·sidecar·checkpoint·split·job/manifest/reveal 계약을 전부 재검증한다. Lockbox receipt를 만들기 전 A0/A2 pair-wide initial state·parameter schema·split/runtime fairness도 검사한다.
- lockbox의 `60명×12 class×10 trials/condition×2 conditions×2 roles×3 seeds = 86,400` prediction-row exact grid가 하나라도 빠지면 primary는 invalid다. 부분 seed 평균·complete-case 구제는 금지한다.
- condition별 통계 검정은 하지 않고 descriptive 결과만 보고한다.
- target-free inference simulation receipt/hash가 깨지면 frozen plan 검증을 통과하지 못한다.
- Full asset 감사에서는 S4–S102 raw/processed 신호 byte와 metadata를 무결성 검사용으로 읽었다. 이후 사전 할당 source 39명은 v1 학습·선택·성능 계산에 한 번 사용했고, held 60명 lockbox prediction은 없다.

이 봉인은 암호학적 data enclave가 아니다. 같은 OS 사용자에게 raw HDF5가 읽기 가능하므로 보호 수준은 **application/procedural gate + hash provenance + append-only decision record**다. 기계적 상태를 구분하면 `(1)` raw audit 접근 완료, `(2)` v1 source training·outcome claim 완료, `(3)` held seal·training·prediction·reveal 미실행이다.

## 현재 중단·승인 상태

다음 source→conditional-held 권한과 N=60 sensitivity 수용은 **완료된 V1의 historical
directive**다. V1 source no-go로 conditional held 권한은 소멸했고, V2에는 synthetic-only
단회 권한만 있었으며 그것도 terminal로 소비됐다. 현재 external·Choi·held 실행 권한은 없다.

- [V1 historical 승인·동결] A_QM−A_Q complete-block eAUC, exact 39/60 allocation, SESOI·margin·N=60
  sensitivity와 source gate 뒤 조건부 held 실행
- [완료] diagonal-Gaussian prior, staged common-path freeze, 56-d Q/18-d M, spectral composite,
  full checkpoint와 correct/missing/shuffle/stale/opposite-interface 개입
- [완료] privileged input sealer, opaque q_/s_/f_ token, no-y HDF5, encrypted query-label/covariate
  sidecar, exact artifact receipt와 phase manifest
- [완료] strict FBCCA, target template, target filter-bank eTRCA, SAME3 one-shot component,
  Chiang-2021 LST port. CSDuDoFN/OS-SSVEP와 SSVEP-DAN은 license/protocol 사유로 audited NOT_RUN
- [완료] source candidate 9+baseline 15=24, held candidate 3+baseline 5=8 exact job contract,
  signed per-job capability와 Bubblewrap projection, producer/checkpoint/result provenance 검증
- [완료] M-free episode cache, batched base-input hash와 role×context×budget별 한 번의
  order-preserving probability CPU materialization. Q-off cell은 Q 값을 logits에 넣지 않음
- [완료] complete private-grid 검증 뒤 단 한 번 label join, participant reducer,
  source gate/H1/H2/mechanism/safety 분석과 atomic result publication
- [완료] permanent cohort attempt key, registry lease/claim, post-claim fail-stop, signed pending
  completion의 제한적 publication-only recovery와 private-key destruction gate
- [완료] plan/power, full test·lint·config parse, commit `56bff34`, annotated freeze tag 봉인
- [완료·no-go] source 24/24 jobs, private grid, one-shot claim·finalization·signed lifecycle
- [미개봉] source FAIL에 따라 held 8 jobs·60명 input seal/claim/result를 생성하지 않음
- [완료·inconclusive] V2 V11은 global claim·미래 beacon 뒤 validator 재귀로 scientific result 전 terminal; 자동 재시도 금지
- [금지] 현 V11에 대한 BETA/Dong selection·independent outcome, Choi replication, wearable held 60 outcome
- [기계적 차단] hash-pinned terminal deny-overlay가 repository의 synthetic 재실행 및 BETA/Dong request·prediction·label join·reduction·selection·gate를 입력 접근 전에 거부
- [다음 후보] 새 schema·seed·amendment·clean freeze·future beacon을 갖춘 별도 candidate만 검토
- [보존] query-only BETA 35/20 plan은 frozen baseline-only이며 실행하지 않음
- [불변] physical_hybrid_v1, S1–S3 추가 outcome과 reliability-spatial-v1 재실행 금지

아래 항목은 이전 후보의 historical completion ledger다. 결과를 새 후보 승인으로 바꾸지 않는다.

- [완료] physical six-role mechanism assay와 권고상 마지막 한 번의 S1–S3 reveal 사용 승인
- [완료] Full A2 개입의 8개 numeric 기준, 10-gate 판정과 중단 규칙 동결
- [완료] clean annotated physical freeze tag `physical-reveal2-freeze-20260901-r1`와 exact 18-job manifest 검증
- [완료] 18/18 train, 3/3 intervention, global S1–S3 ledger precommit, atomic publication과 complete aggregate 검토
- [완료] valid assay에서 substantive gate 네 개 실패를 확인하고 현 `physical_hybrid_v1` confirmatory 진입을 차단
- [완료] `DEC-20260903-005`로 `physical_hybrid_v1`, S1–S3 추가 outcome, 과거 39/60 실행을 deny overlay로 영구 종료
- [완료] `DEC-20260903-006`으로 승인한 `reliability-spatial-v1` 합성 asset과 one-shot Stage-0 실행
- [결과] integrity·Q-identifiability·leakage·pairing control은 통과했지만 metadata increment consistency와 channel-rank increment가 실패해 terminal `failed`
- [중단] passing receipt가 없으므로 A0/A_M/A_Q/A_QM CUDA Stage-1은 시작하지 않았고 canonical Stage-1 root도 생성되지 않음
- [승인] `DEC-20260904-001` 방향 승인으로 별도 query-only 후보의 코드·계약·outcome-free dry-run 설계 시작
- [완료] BETA 35/20 분할, 6-file asset bundle, 3,200 lockbox sample identity, within-participant/class wrong-Q mapping을 해시 고정
- [완료] Q0/Q1 exact graph, 2×2 corruption family, 잠금군 선차단, bundle semantic hash와 통계 reducer, RTX CUDA Q0/Q1 forward-only 검증
- [과거 대기/현재 취소] `+0.02/−0.01/12-of-20` utility, corrupted deployment gate, mechanism runner와 exact query-only owner receipt
- [과거 대기/현재 취소] BETA reveal 전 Dong 2×2 replication 사전등록
- [현재 금지] 철회된 BETA human training·lockbox outcome·Dong query-only replication
- [후속 결정] `DEC-20260904-005`가 metadata 방향을 새 후보로 복구했지만, untouched 39명을 acquisition-factor source development로 쓰려면 exact method·partition·gate와 별도 실행 승인이 필요함
- 선택 사항: Nakanishi 재사용 허가, REVE gated model 접근, 추가 외부 데이터 license 확인

현재 decision contract에서는 owner review가 실패한 substantive gate를 통과로 바꾸지 않는다. SESOI 등 나머지 confirmatory 값을 채우거나 clean tag를 새로 만드는 것만으로 39/60 실행을 허용할 수 없다. Nakanishi·REVE·외부 제3 데이터는 현 결과를 덮어쓰기 위한 것이 아니라, 새 독립 human development study를 정당화·검증할 때만 사용한다. 합성 결과 역시 이 제한을 해제하지 않는다.

## reliability-spatial-v1 합성 Stage-0 결과

Canonical one-shot receipt의 최종 상태는 `failed`다. Asset 8개 파일, 8명×256 trial,
2,048 sample×active-channel truth row, 참가자 분리 nested LOSO, predictor allowlist, truth 격리,
class balance와 label-only null은 모두 유효했다.

| 사전 gate | 관찰값 | 기준 | 판정 |
|---|---:|---:|---:|
| Q held-subject mean R² | 0.973349 | > 0 | PASS |
| M의 Q 초과 partial R² 평균 | 0.043065 | ≥ 0.02 | PASS |
| partial R² 양수 participant | 5/8 | ≥ 6/8 | **FAIL** |
| 채널 rank Spearman 증분 평균 | 0.000651 | ≥ 0.02 | **FAIL** |
| rank 증분 양수 participant | 5/8 | ≥ 6/8 | **FAIL** |
| correct-vs-block-shuffle relative SSE gain | 0.152518, 8/8 양수 | ≥ 0.01, ≥ 6/8 | PASS |
| correct-vs-channel-permutation relative SSE gain | 0.127443, 8/8 양수 | ≥ 0.01, ≥ 6/8 | PASS |
| label-only subject-macro R² | −0.137111 | ≤ 0.01 | PASS |
| class별 target mean range | 0.002847 | ≤ 0.02 | PASS |

즉 M은 simulator에서 실제 corruption과 연결돼 있고 잘못 짝지으면 예측이 나빠졌다. 그러나
Q가 이미 realized quality를 거의 다 설명하는 상황에서, M을 더한 mapping은 처음 보는
participant 전반에 안정적인 추가 이득이나 유의미한 채널 순위 개선을 주지 못했다. 이는
“현실의 모든 metadata가 무의미하다”는 결과가 아니라, 이 동결 DGP·feature·target에서 당시
synthetic H1은 성립했지만 당시 synthetic H2의 복합 gate가 성립하지 않았다는 engineering
no-go다. 당시 synthetic H3 decoding과 H4
missing/stale/wrong 안전성은 실행·평가하지 않았다. Receipt SHA-256은
`e8f3f87bbe47a8c8d5825f7d0525c318e87d42c6ecdd2840de21b1c657eefc35`다.

## S1–S3 첫 outcome-gated grid 결과

고정된 spectral·10-epoch·seed-42 **legacy `prompt_adapter_v1`** 설계로 A0/A2×3 folds 여섯 job을 모두 완료한 뒤 결과를 한 번에 공개했다.

| 지표 | A0 | A2 | A2−A0 |
|---|---:|---:|---:|
| 3명 평균 BA | 0.1222 | 0.1042 | −0.0181 |

피험자별 delta는 S1 −0.0167, S2 −0.0042, S3 −0.0333으로 모두 음수였다. 이 결과는 metadata 이득을 지지하지 않는다. 동시에 fold마다 학습 participant가 2명뿐이고 N=3 최소 단측 exact p-value가 0.125이므로, 39명 학습 ensemble의 모집단 효과를 반증하지도 못한다. 사전 지정된 formal red-flag cutoff는 없었으므로 판정은 `technical-valid / directional-warning / benefit-not-demonstrated / population-inconclusive`다.

이것은 S1–S3 개발 config가 허용한 recommended 2회·absolute 3회 중 **첫 grid outcome reveal**이다. 이 grid의 A2는 legacy `prompt_adapter_v1`이므로 새 physical A2의 성능으로 간주하지 않는다. Confirmatory를 자동 진행하지 않는다. 두 번째 reveal은 근거 없이 양의 결과를 찾는 용도가 아니라, 사전 선언한 physical routing mechanism/safety assay 또는 추가 wearable-like development data에만 사용한다.

## S1–S3 physical reveal #2 결과

`DEC-20260901-004`와 execution tag `physical-reveal2-freeze-20260901-r1`에서 18/18 fixed-epoch job과 3/3 Full-A2 intervention bundle을 완성하고 reveal index 2를 원자적으로 공개했다.

| 역할 | 3명 평균 BA | A0 대비 |
|---|---:|---:|
| A0 external-null | 0.1292 | — |
| Full A2 observed | 0.1194 | −0.0097 |
| Global-only | 0.1208 | −0.0083 |
| Channel-only | 0.1306 | +0.0014 |
| Shuffle-train→clean-val | 0.1194 | −0.0097 |
| Metadata-only | 0.0833 | chance excess 0.0000 |

Assay는 유효했다. Counterfactual condition flip과 train/inference bundle changed fraction은 모든 fold에서 `1.0`이었고 potency gate를 모두 통과했다. Metadata-only shortcut, observed clean-subject harm, wrong-metadata safety screen도 통과했다. 그럼에도 다음 네 substantive gate가 실패했다.

- clean A2 directional mean: `−0.0097222 < 0`
- counterfactual reliance: least-favourable `−0.0041667 < 0.0125`
- inference pairing shuffle: least-favourable `0.0000000 < 0.0125`
- training pairing shuffle: least-favourable `−0.0041667 < 0.0125`

피험자별 Full−A0는 sub001 `−0.0125`, sub002 `0`, sub003 `−0.0167`이다. 이는 N=3 wiring/mechanism 진단이므로 모집단에서 metadata 효과가 없다는 결론은 아니다. 그러나 조작 potency가 충분한데도 correct metadata의 방향성 이득과 pairing/counterfactual reliance를 보이지 못했으므로 사전 중단 규칙상 `valid assay / diagnostic no-go`이며 현재 A2로 confirmatory를 진행할 수 없다.

## 다음 연구 루프

1. [완료] Nested metadata calibration-prior 수식, staged common-path freeze, 39명 source gate와
   60명 held 평가 규칙을 고정했다.
2. [완료] Privileged input sealer가 support-label view, opaque-token query view, query-label sidecar,
   one-row-per-block context view를 서로 다른 권한의 artifact와 서명 receipt로 봉인한다.
3. [완료] Complete-block/query identity, sample-level support nesting, candidate와 baseline의 독립
   reset runner, private-grid seal, one-time label join, no-replace outcome claim을 구현했다.
   동결된 실행량은 source candidate 9 + baseline 15 = 24 job, held candidate 3 + baseline 5 = 8 job이다.
4. [완료] Strict FBCCA, target template, target filter-bank eTRCA, SAME component와 Chiang
   LST+FB-eTRCA를 직접 비교군으로 고정했다. CSDuDoFN과 SSVEP-DAN은 공개 구현의 license 및
   protocol 적합성을 감사한 결과 이 confirmatory grid에서는 `NOT_RUN`으로 사전 기록했다.
5. [완료] eAUC H1, calibration-value H2a, k1-vs-k3 NI H2b, mechanism Holm과 outcome-free
   200,000-draw power receipt 생성 절차를 구현했다.
6. [완료] Plan/power·전 셀 캐시 동등성·536 tests·clean annotated execution tag를
   봉인했다.
7. [완료·no-go] Source 24/24 jobs·75/75 private artifacts·4,680 query를 실행했고
   source gate FAIL을 한 번 공개했다.
8. [미개봉] 조건이 충족되지 않아 60명 held bundle의 seal, directory, claim, job,
   result는 모두 생성하지 않았다.
9. [완료] Ke2025 24명은 impedance가 전부 `n/a`이고 interface contrast가 없어 primary
   M 검정에 부적합하다. BETA/Wang/Dong도 paired block impedance가 없다.
10. [완료] A_Q k1 no-harm, epoch-0 M abstention과 pairing-aware M operator를 V2 후보로
    구현하고, 사람 EEG 전에 synthetic V11 gate를 사전등록·동결했다.
11. [완료·inconclusive] V11 단회 실행은 claim·beacon 뒤 validator 재귀로 efficacy 계산 전
    terminal이 됐다. Scientific PASS/FAIL 값은 없고 같은 lockbox는 소비됐다.
12. [차단] External·Choi·held outcome은 실행하지 않는다. 재개하려면 기술 수정 외에도 새
    candidate/schema와 seed, 명시적 amendment, clean freeze와 새 future beacon이 필요하다.

현 v1은 source 39명 outcome을 한 번 공개한 `development_no_go`이고, V2 V11은 과학 결과가
없는 `infrastructure inconclusive`다. Held 60명의 support-label access, query
prediction·outcome은 모두 미개봉이다. 어느 후보도 결과를 본 뒤 threshold를 바꾸거나 같은
lockbox/cohort로 재실행하지 않는다.

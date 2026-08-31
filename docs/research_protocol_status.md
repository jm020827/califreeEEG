# Calibration-Free EEG 연구 프로토콜 현재 상태

기준일: 2026-09-01
현재 상태: **physical A2 reveal #2 완료 — valid assay / diagnostic no-go / 현 39/60 confirmatory 경로 봉인**
현재 기준원: `DEC-20260830-001`, `DEC-20260830-002`, `DEC-20260901-004` (`DEC-20260831-003`은 superseded history)

이 문서는 현재 설계의 단일 요약본이다. 세부 이력은 [append-only 연구일지](research_log.md), 실행 절차는 [연구 실행 계획](research_execution_plan.md), 기계 판독 계약은 `configs/analysis/wearable_primary.yaml`을 따른다. 과거 문서의 `5-fold N=99 primary` 서술은 역사적 기록이며 `DEC-20260830-002`가 대체한다.

## 연구목표

연구 질문은 바뀌지 않았다.

> 처음 보는 사람의 EEG를 추가 보정 데이터 없이 분류할 때, 측정 전에 알 수 있는 획득조건 metadata가 EEG·공통 채널 구조·현재 query에서만 계산한 QC를 쓰는 동일 크기 모델보다 balanced accuracy를 개선하는가?

현재 primary는 **closed-set SSVEP, strict inductive k=0, unseen participant**다. 새로운 stimulus class를 발견하는 open-set/OOD 탐지나 target stream 적응은 연구 범위가 아니다. `arXiv:2608.11829`에서 빌린 `learned/retained/forgotten` 관점은 결과 해석용 보조 분석일 뿐 primary endpoint를 바꾸지 않는다.

## 가장 쉬운 실험 설명

두 모델은 같은 39명 EEG, 같은 backbone, 같은 parameter shape, 같은 세 optimization seed와 정확히 10 epoch를 쓴다.

- A0: EEG, 채널 ID/mask, sampling/time grid, query-local pre-zscore scale QC를 사용한다. Physical external branch는 동일하게 보유하지만 global residual 0·channel gain 1로 강제한다.
- A2: A0와 동일하며 electrode type, block impedance mean/max, channel별 impedance와 availability만 추가로 관측한다. Global 정보는 zero-init bounded FiLM, channel 정보는 shared bounded gain으로 넣는다.
- 선택: validation이나 early stopping 없이 세 seed의 final checkpoint를 고정한다.
- 시험: 모델 학습·선택·성능 평가에 쓰지 않은 60명에게 두 ensemble을 첫 confirmatory 공개에서 함께 적용한다. 원시 신호 byte는 전체 자산 무결성 감사에서 이미 읽혔지만 성능이나 모델 선택에는 쓰지 않았다.
- primary 값: 각 lockbox participant에서 세 seed BA를 먼저 평균한 뒤 `BA(A2) - BA(A0)`를 계산하고, 60개의 paired difference 평균을 검정한다.

Reference/cap은 wearable에서 상수라 primary 입력에서 제외한다. Stable role 이름 `A2_structured_condition_prompt`는 남지만 현재 primary는 prompt token/adapter를 쓰지 않는다. Dry/wet별 결과, worst participant, learned/forgotten은 설명적 진단이다. 단 하나의 confirmatory contrast는 전체 `A2−A0` 평균이다. 정확한 모델 계약은 [A2 physical-hybrid 설계](a2_physical_hybrid_design.md)에 있다.

## 데이터 역할과 정보권한

| 역할 | 피험자 | 행 | 지금 가능한 작업 | 금지된 작업 |
|---|---:|---:|---|---|
| 전체 자산 감사 | 102명 | 24,480 | checksum, schema, raw↔processed, metadata 품질 감사 | 성능 기반 선택 |
| development | S1–S3 | 720 | 역사적 prompt 6-job과 physical six-role×3-fold 18-job·3 intervention bundle 공개 완료; reveal #2 소비, valid diagnostic no-go | 모집단 효과 주장, S1–S3 재튜닝·추가 reveal |
| confirmatory training | 사전 추출한 S4–S102 중 39명 | 9,360 | 현 candidate에는 없음; 새 독립 study가 gate를 통과할 때만 재승인 검토 | 현재 학습, validation, checkpoint 선택, lockbox 신호 접근 |
| primary lockbox | 나머지 60명 | 14,400 | 현 candidate에는 없음 | prediction, 재튜닝, 재분할, 누락값 완성 분석 |
| post-primary exploratory | S4–S102 전체 99명 | 23,760 | 현 candidate에는 없음; 새 계약 전까지 봉인 | 성능 계산, primary와 혼합 |

39/60 ID는 `DEC-20260830-002`, `allocation_seed=42`, 정렬한 S4–S102를 NumPy `default_rng`로 한 번 shuffle한 `confirmatory_lockbox_v1` 결과로 plan에 명시돼 있다. Canonical allocation SHA-256은 `0c5b7cd1ca8eedafbe9290833a283160f7bfe89897be8923b21568f70bb18898`이다. 성능을 보고 분할하지 않았고 바꿀 수 없다.

## 왜 5-fold에서 39/60 lockbox로 바꿨나

원래 N=99 5-fold 계획에서는 같은 fold의 약 20명이 같은 모델을 공유하고, 다섯 모델의 training set도 크게 겹쳤다. 그런데 피험자 99명의 차이를 독립 표본처럼 sign-flip/bootstrap하면 이 model-shared 오차를 무시한다.

Target outcome을 전혀 읽지 않은 synthetic stress simulation에서 nominal alpha 0.05의 rejection rate는 독립 조건에서는 허용 범위였지만, fold ICC 0.05와 training-overlap이 있으면 약 0.122, 결합 stress에서는 약 0.204까지 상승했다. 따라서 기존 primary inference는 폐기했다. 새 60명 독립 lockbox 설계의 10,000-replicate simulation은 네 core DGP에서 Type-I upper bound ≤0.0581, one-sided coverage lower bound ≥0.9413, SESOI+0.03 power lower bound ≥0.8769로 사전 기준을 통과했다.

이는 39명만 학습에 사용한다는 비용이 있다. 그 비용을 lockbox를 열어 최적화하지 않기 위해 S1–S3와 공개 Wang 데이터에서 학습 가능성만 먼저 확인한다.

## 현재 모델과 근거

- Tiny time-domain Transformer는 synthetic 12-class에는 학습됐지만 Wang subject-independent source validation에서는 3 epoch 동안 chance였고 one-batch 자체 기준도 통과하지 못했다.
- Query-local complex RFFT의 real/imaginary/log-magnitude를 쓰는 spectral Transformer는 동일 Wang 7명 validation에서 10 epoch BA 0.8220을 보였다.
- 같은 7명·1,680 trial에서 CCA BA 0.7875, strict Chen-2015 FBCCA BA 0.7768이었다.
- pure F0 spectral run도 BA 0.8107이어서 metadata 모듈 없이 backbone 자체가 학습 가능함을 확인했다.

이 수치는 모두 legacy/development evidence이며 metadata 효과가 아니다. 문헌에서도 user-independent complex spectrum CNN과 SSVEPformer가 spectral input을 직접 지지하지만, A2의 인과적 이득은 오직 lockbox paired contrast로 판단한다.

## Metadata 권한

| 묶음 | 예 | A0 | A2 |
|---|---|---:|---:|
| 공통 구조 | canonical channel ID/mask, sampling/time grid | 사용 | 사용 |
| 공통 query-local QC | 현재 trial의 pre-zscore channel std | 사용 | 사용 |
| 외부 global treatment | electrode type, pre-query impedance mean/max | algebraic null | observed |
| 외부 channel treatment | canonical channel별 pre-query impedance/availability | gain 1 | observed bounded gain |
| 상수라 primary 제외 | reference, cap type | 금지 | 금지 |
| 분석 covariate | headband order, condition period | 모델 입력 금지 | 모델 입력 금지 |
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
- full asset 감사에서는 S4–S102 raw/processed 신호 byte와 metadata도 무결성 검사용으로 읽었다. 그러나 S4–S102를 모델 학습·선택·성능 계산에 쓰지 않았고 60명 lockbox prediction은 아직 없다.

이 봉인은 암호학적 data enclave가 아니다. 같은 OS 사용자에게 raw HDF5가 읽기 가능하므로 보호 수준은 **application/procedural gate + hash provenance + append-only decision record**다. 기계적 상태를 구분하면 `(1)` raw audit 접근 완료, `(2)` confirmatory training 접근 미실행, `(3)` lockbox prediction/reveal 미실행이다.

## 현재 중단 상태와 다음 결정

Confirmatory plan은 `dev_not_frozen`이며 현재 physical A2 경로는 사전등록 중단 규칙에 따라 no-go다. 아래 완료 항목은 결과를 승인한다는 뜻이 아니라 실행·공개·검토가 계약대로 끝났다는 뜻이다.

- [완료] physical six-role mechanism assay와 권고상 마지막 한 번의 S1–S3 reveal 사용 승인
- [완료] Full A2 개입의 8개 numeric 기준, 10-gate 판정과 중단 규칙 동결
- [완료] clean annotated physical freeze tag `physical-reveal2-freeze-20260901-r1`와 exact 18-job manifest 검증
- [완료] 18/18 train, 3/3 intervention, global S1–S3 ledger precommit, atomic publication과 complete aggregate 검토
- [완료] valid assay에서 substantive gate 네 개 실패를 확인하고 현 `physical_hybrid_v1` confirmatory 진입을 차단
- [결정 필요] 현재 연구 경로를 종료할지, 독립 외부 development data와 새 사전근거로 별도 모델/설계를 시작할지
- [새 연구에만 필요] SESOI, alpha/alternative, operational threshold, multiplicity, fairness와 external source lock
- 선택 사항: Nakanishi 재사용 허가, REVE gated model 접근, 추가 외부 데이터 license 확인

현재 decision contract에서는 owner review가 실패한 substantive gate를 통과로 바꾸지 않는다. SESOI 등 나머지 confirmatory 값을 채우거나 clean tag를 새로 만드는 것만으로 39/60 실행을 허용할 수 없다. Nakanishi·REVE·외부 제3 데이터는 현 결과를 덮어쓰기 위한 것이 아니라, 새 독립 development study를 정당화·검증할 때만 사용한다.

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

1. [완료] Outcome-free 단계에서 physical algebra, 정보 reachability, block mapping, CUDA forward/schema와 전체 회귀검증을 통과시킨다.
2. [완료] Six-role 질문·10 epoch·8개 numeric 기준·10-gate 중단 규칙·canonical output과 reveal index 2를 `DEC-20260901-004`로 재승인했다. 최초 exact 실행은 공개·precommit 전에 validator-only 오류로 중단됐고 성능 결과를 검토하지 않은 채 mode-0700 quarantine으로 옮겼다.
3. [완료] Clean annotated tag `physical-reveal2-freeze-20260901-r1`에서 18개 job을 처음부터 재학습하고 단일 transaction으로 18개 prediction·3개 intervention·aggregate를 공개했다. Reveal #2는 소비됐다.
4. [완료] Potency가 유효한 상태에서 네 substantive gate 실패를 확인해 `diagnostic no-go`를 적용했다.
5. [중단] 같은 S1–S3에서 threshold·architecture·seed·epoch를 바꾸거나 추가 reveal로 양의 결과를 탐색하지 않는다.
6. [차단] 현 A2의 39명 training과 60명 lockbox prediction은 실행하지 않는다.
7. [다음 결정] 연구를 여기서 종료하거나, S4–S102를 열지 않은 채 독립 wearable-like data·외부 source-only evidence로 새 후보를 개발하고 새 질문·gate·reveal budget을 별도 사전등록한다.

현재까지 S4–S102 성능 outcome과 60명 lockbox prediction은 모두 미접근 상태다. 최신 검증 수는 최종 연구일지 항목과 저장소 테스트 실행 결과를 따른다.

# 연구 의미·데이터 확장·최신 문헌 전략

> **2026-09-06 결과 갱신:** `metadata-calibration-efficiency-v1`의 source 39명·24 jobs는
> A_QM−A_Q eAUC `−0.004843`, correct−shuffle `0`, A_Q k5 `0.476282`로
> `development_no_go`였다. Held 60명은 미개봉이다. 현재 결론과 데이터 요청은
> [source 결과 문서](metadata_calibration_efficiency_results.md)가 이 historical strategy를 supersede한다.

> **2026-09-04 superseded / historical strategy:** 아래 strict-k0 A0/A2 문구는 당시 연구
> 전략이다. 현재 목표는 k=0 anchor를 포함한 metadata-assisted low-calibration curve이며,
> [현재 설계](metadata_calibration_efficiency_design.md)를 우선한다. 데이터·문헌의 한계 분석은
> 역사적 근거로 유지한다.

> **2026-09-01 상태 갱신:** impedance mapping, 전체 wearable_v3 acceptance와 physical-hybrid six-role/18-job 경로를 구현했고 `DEC-20260901-004`의 reveal #2도 완료했다. Assay potency는 유효했지만 A0 BA `0.1292`, Full A2 BA `0.1194`, Δ `−0.0097`이고 substantive gate 네 개가 실패해 diagnostic no-go다. S1–S3 재튜닝·추가 reveal과 현 candidate의 39/60 confirmatory는 차단한다. 이 문서의 문헌·데이터 확장 논리는 유지하되 실행 권한은 [프로토콜 현재 상태](research_protocol_status.md)를 따른다.

기준일: 2026-08-29
적용 범위: 연구 주제는 바꾸지 않고, Protocol 0.3의 평가 골격과 Protocol 0.4 metadata amendment를 보강한다.

> **한 줄 판정:** 연구 질문은 여전히 의미가 있지만 현재 구현은 개발 gate를 통과하지 못했다. 이 no-go는 `physical_hybrid_v1`이 S1–S3에서 benefit·pairing reliance를 보이지 않았다는 candidate-specific 진단이며, metadata가 일반적으로 도움이 되지 않는 조건을 확립한 결과는 아니다. 계속하려면 피험자 수만 늘리기보다 **독립적인 획득 regime와 같은 metadata factor의 반복**으로 새 candidate를 outcome-blind하게 개발해야 한다.

> 원자료 숫자, 직접 경쟁법의 공개 코드, foundation 선행연구까지 재정독한 변경 결론은 [2026-08-29 재정독 감사](deep_read_reassessment_2026-08-29.md)가 우선한다.

## 1. 의사결정 요약

| 질문 | 판정 | 바로 해야 할 일 |
|---|---|---|
| 연구가 과학적으로 의미가 있는가? | **조건부 Yes** | “decoder 하나 더”가 아니라, factorized physical metadata의 증분 가치와 실패 조건을 검증하는 연구로 claim을 고정한다. |
| 현재 4개 데이터셋으로 시작할 수 있는가? | **Physical 완료 / 현 confirmatory No** | Valid assay의 substantive no-go로 39/60을 열지 않는다. 독립 wearable-like development data 또는 새 사전근거가 먼저다. |
| 현재 데이터로 broad device/site OOD를 주장할 수 있는가? | **No** | 서로 독립적인 acquisition regime, 반복된 factor level, untouched external target을 추가한다. |
| 데이터셋을 더 많이 받아야 하는가? | **Yes, 그러나 선별적으로** | exact-label family, shared-frequency OOD, controlled-context, representation-only 역할을 분리한다. |
| 관련 연구 조사가 충분한가? | **설계 변경에는 충분, systematic review는 아님** | 핵심 12편의 원문/코드를 감사해 baseline과 novelty를 재분류했다. 미확보 원문·저자 clarification은 계속 추적한다. |
| 가장 먼저 추가할 데이터는? | **Nakanishi 권리·s1–s10 확인** | Nakanishi는 exact-12 replication이다. Kim은 frequency 10개만 겹치고 phase가 모두 π만큼 달라 exact lockbox가 아니다. |
| 가장 강한 후속 자료는? | **동일 참가자 randomized crossover 신규 수집** | wet/dry 또는 device를 교차하고, block 전 per-channel impedance와 전체 acquisition provenance를 기록한다. |

## 2. 당시 연구가 실제로 묻던 질문

쉽게 말하면 다음 비교이다.

1. 모델은 여러 source participant의 EEG와 획득조건을 학습한다.
2. 시험 때는 처음 보는 target participant의 교정용 EEG, target batch 통계, 과거 target stream을 전혀 쓰지 않는다.
3. 현재 query EEG와 공통 구조/QC만 보는 A0와, 여기에 query 전에 이미 알 수 있는 electrode type·block impedance·채널별 impedance를 더한 A2를 비교한다. Wearable에서 상수인 reference/cap은 primary treatment에서 제외한다.
4. 같은 backbone·parameter 수·학습 예산에서 A2가 더 잘하면 metadata의 **증분 가치**가 있다.

당시 주가설은 다음처럼 고정했다.

> **H1:** target participant의 labeled/unlabeled EEG나 target-set 통계를 adaptation에 사용하지 않는 inductive strict k=0 조건에서, factorized transferable acquisition metadata를 사용하는 A2가 동일 구조의 metadata-null A0보다 held-out participant balanced accuracy를 높인다.

이 연구에서 OOD는 새로운 stimulus class의 발견이 아니다. class/frequency 의미는 닫혀 있고, participant·session·장비·전극·reference·환경이 바뀌는 **closed-set domain OOD**이다.

## 3. 왜 의미가 있는가, 어디서부터 과장인가

### 3.1 의미가 있는 이유

- 실제 BCI 배포에서는 사람과 장비가 바뀌지만, 매번 많은 calibration EEG를 얻기 어렵다.
- 전극 위치, reference, 접촉 방식, sampling, impedance는 query label을 누설하지 않으면서 측정 전에 알 수 있는 정보다.
- 최신 SSVEP 연구는 spectral architecture, source-style augmentation, subject expert routing, target adaptation에 집중한다. 물리적으로 해석 가능한 획득 metadata를 factorized해 strict source-only cross-acquisition 조건에서 반증 실험까지 수행한 직접 근거는 드물다.
- 효과가 없더라도, 적절한 power와 negative control 아래 “어떤 acquisition metadata가 신호·geometry-aware model 위에 추가 가치를 주지 않는가”를 보인 benchmark는 의미 있는 부정 결과가 된다.

### 3.2 신규성의 안전한 경계

다음은 그 자체로 신규성이 아니다.

- generic cross-attention, FiLM 또는 prompt conditioning
- dataset ID embedding
- arbitrary-montage positional encoding
- 평균 정확도 1–2 percentage point 개선만 보고하는 것

[EEG-PRIME](https://arxiv.org/abs/2608.13072)은 2026년 프리프린트에서 task instruction, learned dataset embedding, subject-invariant learning, layer-wise conditioning을 이미 결합했다. 18개 downstream dataset과 두 held-out dataset을 다루지만 SSVEP는 없고, held-out dataset embedding을 어떻게 제공하는지는 원문상 불명확하다. 따라서 이 논문은 직접 SSVEP baseline은 아니지만 **“EEG에 conditioning을 넣었다”는 넓은 novelty claim을 막는 선행연구**다.

이 연구의 방어 가능한 기여는 다음 묶음이다.

> dataset-ID lookup 없이 factorized physical acquisition metadata를 구성하고, geometry와 deterministic preprocessing을 통제한 뒤 strict k=0 unseen-participant/factor-combination 성능을 높이는지 correct·missing·shuffled·swapped·counterfactual metadata로 반증하며, 실제로 새로운 subject-condition 성공 영역을 만드는지 검증한다.

REVE는 3D 좌표+시간의 pre-query encoding, 2026 JNE montage adapter는 좌표→pretrained channel embedding, EEG-PRIME은 task+dataset conditioning을 이미 수행한다. 따라서 “EEG에서 geometry/metadata conditioning 최초”는 철회한다. 남는 차별점은 **lookup-free physical factorization, strict target-set-statistics-free SSVEP, unseen factor combination, wrong-metadata causal controls**의 조합이다.

## 4. 현재 데이터가 무엇을 식별하는가

### 4.1 현재 핵심 데이터

| 데이터셋 | 독립 participant | 핵심 획득조건 | label 관계 | 현재 역할 |
|---|---:|---|---|---|
| [Wearable/Zhu2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7916479/) | 102 | 같은 사람이 wet/dry 모두 수행, 8 posterior channels, block 전 channel별 impedance, 순서 무작위 | exact 12 JFPM, 9.25–14.75 Hz | **Primary mechanism/subject DG** |
| [Wang2016](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Wang2016.html) | 배포 35, MOABB usable 34 | 64ch SynAmps2, Cz, shielded lab | exact 40 JFPM | 40-class source/boundary |
| [BETA/Liu2020](https://doi.org/10.3389/fnins.2020.00627) | 70 | 64ch SynAmps2, Cz, ordinary classroom | Wang과 exact 40 | near-matched environment boundary |
| [Dong2023](https://zenodo.org/records/18847318) | 59 | 8ch NeuSenW, semi-dry, Fp1 reference, 비차폐, 10–16세 | Wang/BETA와 exact 40 | independent acquisition replication |

Wearable은 공개 데이터 중 가장 좋은 within-participant wet/dry 자원이다. 동일 participant에게 두 interface가 존재하고 impedance도 변하므로, 새 participant에 대한 electrode-interface metadata 효과를 검증하기에 강하다.

Figshare v4 `Impedance.mat`의 전체 평균은 axis index 0=261.67 kΩ, index 1=19.63 kΩ이고 논문의 dry/wet 평균과 각각 일치한다. 실제 impedance order는 `[dry, wet]`이다. 기존 `[wet, dry]`로 결합된 asset/run은 무효 처리했고, config·numeric-signature test·`wearable_v3` 전처리·102명 전수 감사를 완료했다.

반면 Wang과 BETA는 hardware·reference·montage가 거의 같아 독립 acquisition 두 개로 세기 어렵다. Dong은 분명히 다르지만 device, channel 수, reference, electrode, 환경, 연령이 함께 바뀐다. 따라서 현재 물리적 regime는 대략 다음 세 묶음에 가깝다.

1. wearable wet/dry paired regime
2. Wang/BETA near-identical regime cluster
3. Dong semi-dry pediatric regime

즉, 현재 설계는 **unseen-participant와 wet/dry mechanism의 좁은 claim에는 좋지만, 임의의 새 device/site에 대한 일반화 법칙을 식별하기에는 부족하다.**

### 4.2 participant 수와 acquisition-regime 수를 분리해야 한다

- `N_subject`: 사람 간 효과와 paired A2−A0를 추정한다.
- `N_regime`: 장비·전극·reference·기관 변화에 대한 transportability를 추정한다.
- trial/window 수: 한 사람 안의 측정 정밀도를 높일 뿐, 독립 사람이나 독립 site 수를 늘리지 않는다.
- training seed: 최적화 불확실성이지 독립 participant가 아니다.

단순 paired normal approximation에서 two-sided α=0.05, power=0.8일 때 대략 `dz=0.5 → n≈34`, `0.4 → n≈52`, `0.3 → n≈90`이다. 현재 독립 lockbox N=60은 중간 크기 효과에는 현실적이지만 작은 효과에는 부족할 수 있다. 그래서 primary 추론은 target-free simulation으로 Type-I·coverage·SESOI+0.03 power를 먼저 검증했고, 최종 SESOI와 alpha는 성능을 보기 전에 승인해야 한다.

### 4.3 현재 데이터로 말할 수 없는 것

- 특정 reference의 독립적인 causal effect
- 특정 amplifier나 electrode material의 독립적인 causal effect
- metadata가 전 세계 임의 site/device로 일반화한다는 주장
- frequency가 다르거나 phase/target semantics가 다른 dataset을 합친 뒤의 한 개 pooled accuracy

공개 cross-dataset 자료에서는 device, reference, age, site, stimulus가 dataset 단위 상수로 함께 움직이는 경우가 많다. 이런 결과는 **association/transportability evidence**이며 단일 factor의 causal effect로 쓰면 안 된다.

## 5. 추가 데이터 전략: 많이가 아니라 역할별로

### 5.1 먼저 동결할 dataset role

모든 데이터셋을 다음 중 하나로 registry에 기록한다.

- `core_exact12`
- `core_exact40`
- `shared_frequency_ood`
- `controlled_context`
- `representation_only`
- `context_robustness`
- `external_lockbox`
- `excluded_or_quarantined`

class key는 정수 class ID가 아니라 최소한 다음 tuple이어야 한다.

```text
(frequency_hz, phase_rad, target_semantics, stimulus_method, display_context)
```

frequency가 같아도 phase, target 위치/의미, single·dual-frequency stimulation이 다르면 자동으로 같은 class로 합치지 않는다.

### 5.2 권장 도입 순위

| 우선순위 | 데이터셋 | 연구 가치 | label 호환성 | 접근·권리와 결정 |
|---|---|---|---|---|
| P0 | 현재 Wearable+Wang+BETA+Dong | 현재 주가설과 exact-family replication 완결 | exact 12 / exact 40 | 먼저 full protocol을 끝낸다. |
| P0 | [Nakanishi 12JFPM](https://nemar.org/dataset/nm000118) | wearable과 문자·위치·frequency·phase가 12/12 정확히 같은 외부 장비 replication | **exact 12** | 원 repo는 s1–s10이나 MOABB/NEMAR는 9명만 노출. data license `Unknown`; 저자 서면 허가 전 confirmatory/재배포 금지. |
| P1 | [Kim2025 BetaRange](https://www.nature.com/articles/s41597-025-06032-2) | 40명, BioSemi ActiveTwo, 31 EEG, 1024 Hz; 독립 장비/site와 within-day drift | frequency 10개 공유, **exact `(f,phase)` 0개** | CC BY 4.0. 겹친 10개 모두 phase가 π만큼 다르므로 shared closed-set lockbox에서 제외하고 frequency-only exploratory protocol로 분리한다. |
| P1 | [MobileBCI/Lee2021](https://doi.org/10.1038/s41597-021-01094-4) | 같은 trial의 scalp+ear와 standing/walk/run; motion×sensor stress | 3 classes, 직접 pooled head 아님 | CC BY 4.0. SSVEP usable 23명, running 16명. speed 순서가 고정돼 time/fatigue와 교란되며 MOABB derivative가 27 IMU channels를 EEG로 오인하므로 OSF source를 직접 쓴다. |
| P1 | [AR binocular SSVEP](https://doi.org/10.1038/s41597-025-05696-0) | 무작위 condition 순서의 binocular frequency/phase mechanism | 별도 binocular target semantics | CC BY 4.0 v8. 확인한 BIDS impedance는 전부 `n/a`이고 hardware JSON도 논문과 불일치하므로 acquisition-metadata dataset이 아니라 stimulus-mechanism positive control로 둔다. |
| P2 | [eldBETA](https://doi.org/10.1038/s41597-022-01372-9) | 100명 older adults, 동일 계열 hardware; population OOD | Wang grid와 8/9/10/11/12 공유 | CC BY 4.0. age OOD이지 새 acquisition의 독립 증거는 아니다. |
| P2 | [Han2024 Fatigue](https://doi.org/10.1109/TNSRE.2024.3380635) | 같은 장비의 training→fatigue/state shift | low band에서 integer 8–15 공유 | CC BY 4.0. state robustness용이다. |
| P2 | [Gu 1–60 Hz](https://www.nature.com/articles/s41597-024-03023-7) | 30명×4일, 1–60 Hz, modulation depth 조작 | shared-frequency/conditional head | CC0 배포. stimulus/session metadata positive-control 및 representation용이다. |
| P2 | [MAMEM I–III](https://physionet.org/content/mssvepdb/1.0.0/) | 256ch EGI와 14ch consumer device 대비 | 5 classes; 10/12 일부 공유 | source/version별 license·participant·hardware 표기가 달라 manifest를 원자료와 대조한다. |
| P1 | [Guttmann-Flury2025](https://doi.org/10.1038/s41597-025-04861-9) | 31명, 63 sessions, randomized paradigm position, EEG+eye tracking+video | 10/11/12/13 Hz, phase 미보고 | 원 논문은 session당 40 trials/총 2520인데 MOABB/NEMAR는 48/3024와 `[8,10,12,15]`, LED로 잘못 기술한다. 교정 manifest를 고정한 session/context stress로만 사용한다. |
| P3 | [OpenBMI SSVEP](https://nemar.org/dataset/nm000273), perturbation, YSU asynchronous, Schrag pediatric | session, cognition, non-control, pediatric/context OOD | 대부분 작은 교집합 | 별도 endpoint나 representation study에서만 사용한다. |

이 순위는 “가장 큰 데이터” 순서가 아니다. 현재 가설을 가장 싸게 반증하거나 외부 타당도를 늘리는 순서다.

### 5.3 데이터는 이렇게 확보한다

1. [MOABB의 현재 SSVEP 목록](https://moabb.neurotechx.com/docs/api.html)은 18개 adapter를 제공하므로 discovery index로 사용한다.
2. NEMAR/EEGDash는 BIDS derivative, versioned manifest, CLI·DataLad·S3 selective download에 사용한다.
3. Figshare, Zenodo, GigaDB에서는 exact version, license, file bytes, checksum을 registry에 고정한다.
4. 대형 dataset은 전부 받지 않는다. 먼저 1–2 participant 파일로 schema, event, phase, reference, checksum을 검증한 뒤 확대한다.
5. MOABB/NEMAR metadata를 ground truth로 가정하지 않는다. 원 논문, versioned archive, 실제 event/header 세 가지를 대조한다.

NEMAR의 예시는 다음과 같지만, Nakanishi는 재사용 권리 확인 전 confirmatory 분석을 시작하지 않는다.

```bash
nemar dataset download nm000118
# 또는 versioned DataLad clone 후 필요한 파일만 get
datalad clone https://github.com/nemarDatasets/nm000118 nm000118
```

각 field의 provenance도 함께 저장한다.

```text
raw_header | bids_sidecar | paper | repository | author_confirmation | inferred
```

`inferred` metadata는 primary model input으로 쓰지 않고 sensitivity analysis에만 사용한다.

### 5.4 저자에게 요청할 정보

license가 없거나 acquisition detail이 불명확하면 다음을 한 번에 요청한다.

- 연구·논문 재사용과 derived artifact 배포 허가
- 원 sampling, online/offline filter, firmware/gain
- reference/ground 회로와 raw reference 보존 여부
- electrode model/material, active/passive, wet/dry/contact medium
- block 전 channel별 impedance와 측정 단위
- cap reattachment, session 간 경과시간, condition order
- stimulus frequency/phase table, refresh timing, photodiode 기록
- 제외 participant/trial과 checksum

## 6. 공개 데이터로 끝나지 않는 경우: 신규 수집

metadata factor의 causal claim까지 원한다면 공개자료를 더 쌓는 것보다 작은 prospective randomized crossover가 낫다.

### 권장 기본 설계

- 동일 participant와 동일 exact-12 JFPM stimulus를 유지한다.
- `wet vs dry/semi-dry`를 participant 내 무작위·counterbalanced crossover로 둔다.
- 가능하면 `interface × device` 또는 `interface × reference` 2×2로 설계한다.
- 최소 두 day/session에 cap을 다시 장착해 reattachment shift를 포함한다.
- reference는 가능한 raw common reference를 보존하고, offline re-reference와 실제 acquisition reference를 구분한다.
- 두 site가 가능하면 하나 이상의 acquisition cell을 두 site에서 반복한다.

반드시 block 전에 per-channel impedance, 3D coordinate/cap placement, device·firmware·gain·filter, electrode/contact, reference/ground, display timing, 방 조도·잡음, order·carryover, fatigue·comfort를 기록하고 BIDS로 공개한다.

표본수를 30/50처럼 임의로 정하지 않는다. wearable의 within-participant A2−A0와 wet−dry variance로 simulation하고, 기대 interaction과 MCID에 맞춰 정한다. 계획 범위로는 moderate paired effect면 대략 40–60명, 작은 effect면 80–100명을 예상할 수 있으나 이는 pilot simulation 전의 heuristic이다.

## 7. 최신 직접 SSVEP 연구: 같은 protocol끼리 비교할 것

### 7.1 Primary와 같은 strict source-only k=0 후보

| 연구 | 상태 | protocol·데이터 | 이 연구에 주는 결정 |
|---|---|---|---|
| [Huang et al., cross-subject domain generalization](https://pubmed.ncbi.nlm.nih.gov/37578926/) | TNSRE 2023, peer-reviewed | target training 없이 3 public dataset의 cross-subject DG | strict k=0 직접 anchor; 원문 split·validation을 재현한다. |
| [TST-CSFR](https://pubmed.ncbi.nlm.nih.gov/39120991/) | TBME 2024, peer-reviewed | source-only analytical transfer, 2 public datasets | 개념적 anchor지만 full equations·공식 code 미확보이므로 literature-only다. |
| [DG-Conformer](https://pubmed.ncbi.nlm.nih.gov/39226201/) | JBHI 2024, peer-reviewed | Benchmark+BETA 보고; 공개 code는 Benchmark 0.8 s | 공개 path의 target privilege는 strict A0에 가장 가깝다. 무라이선스·제한된 구현이므로 허가 또는 독립 port 후 공통 protocol로 실행한다. |
| [TFA-Net](https://pubmed.ncbi.nlm.nih.gov/40030575/) | JBHI 2025, peer-reviewed | 한 public dataset, time-frequency attention, 1 s | 공개 code는 중첩 window의 shuffled sample K-fold이며 runtime 결함도 있어 strict runner에서 제외한다. 논문 실험과 동일하다고 단정하지 않는다. |
| [SSVEP-BiMA](https://arxiv.org/abs/2502.10994) | ICASSP 2025, peer-reviewed | Nakanishi와 MAMEM-II LOSO, raw+complex FFT/phase | 해당 dataset을 쓰면 필수 비교 후보다. |
| [Fast SSVEP](https://arxiv.org/abs/2506.01284) | 2025 preprint | Benchmark/BETA/Nakanishi LOSO, 0.3–0.7 s | short-window source-only 최신 후보; preprint로 명시한다. |
| [SSER](https://pubmed.ncbi.nlm.nih.gov/41615974/) | TBME 2026, peer-reviewed ahead-of-print | spatial/energy style mixing augmentation | 공개 test는 무작위 복원·중첩 crop과 unseeded Monte Carlo다. runner는 제외하고 training-only augmentation ablation으로 이식한다. |
| [TBMSCCN](https://pubmed.ncbi.nlm.nih.gov/41855051/) | TBME 2026, peer-reviewed ahead-of-print | 논문은 Benchmark/BETA LOSO, 0.2–1.0 s | 공개 code는 개인별 5-block train/1-block test로 headline과 다르다. CC BY manuscript 기반 독립 구현만 고려한다. |
| [DC-TRCA-net](https://pubmed.ncbi.nlm.nih.gov/41360014/) | BP&EE 2025, peer-reviewed | source selection과 positive transfer | source selection baseline 후보다. |

최소 **실행 가능한** primary 비교 묶음은 `CCA/paper-faithful FBCCA + compact scratch A0/A2 + protocol-matched DG-Conformer 계열 port + Fast SSVEP`이다. SSER은 orthogonal augmentation ablation으로 평가한다. TST-CSFR는 literature-only, TBMSCCN-C는 독립 구현이 검증된 뒤 승격한다. TFA public runner는 제외한다. Nakanishi/MAMEM을 쓰면 BiMA를 추가한다.

### 7.2 이름은 calibration-free지만 primary와 자원이 다른 연구

| protocol | 예 | target 접근 | 처리 |
|---|---|---|---|
| Transductive source-free DA | [SFDA-SSVEP](https://github.com/osmanberke/SFDA-SSVEP-BCI), Cross-domain Correlation Analysis | target participant의 unlabeled set/covariance | 별도 transductive 표 |
| Transductive self-training | [CSST](https://arxiv.org/abs/2601.21203) | 전체 unlabeled target과 pseudo-label | 별도 transductive 표 |
| Online adaptation | [Low-Latency TTA](https://www.mdpi.com/2076-3417/16/8/3799), OAST-CCA | 과거 target stream을 누적 사용 | 별도 causal-stream 표 |
| Supervised low calibration | SOFT, eTRCA/msTRCA 등 | labeled support 일부 | k=1/3/5 표 |
| Prior-enrolled expert bank | EvoMoE | 이전 사용자를 순차 등록 | fixed-source LOSO와 분리 |
| Unseen class zero-shot | [GZSL-Lite](https://pubmed.ncbi.nlm.nih.gov/40111769/) | target user의 일부 seen-class label | novel-frequency/class 연구로만 인용 |

이들을 strict k=0과 한 ranking에 섞으면 안 된다. 모든 표에는 target label 수, target unlabeled trial 수, batch 선행 접근, update parameter, online causal 여부를 적는다.

## 8. Foundation model·metadata 인접 연구가 바꾸는 것

| 연구 | 상태 | 직접성 | 반영할 점 |
|---|---|---|---|
| [REVE](https://proceedings.neurips.cc/paper_files/paper/2025/hash/20a917f77773ac0fa8bea2bdd6606b66-Abstract-Conference.html) | NeurIPS 2025 | SSVEP 직접 아님 | 3D electrode+time encoding과 arbitrary montage 근거. geometry는 A0/A2 공통 구조 정보로 둔다. |
| [LUNA](https://proceedings.neurips.cc/paper_files/paper/2025/hash/66969a9e6bd7a26dfeccea7227178ca7-Abstract-Conference.html) | NeurIPS 2025 | SSVEP 직접 아님 | learned query/cross-attention arbitrary topology의 인접 선행연구다. |
| [Are EEG Foundation Models Worth It?](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f0f39b7686634fc81ca0112566b2c05f-Abstract-Conference.html) | ICLR 2026 | EEG benchmark | 큰 FM이 항상 compact/classical model보다 우수하지 않으므로 FM은 유일한 backbone이 아니라 secondary여야 한다. |
| [Montage-agnostic frozen adapter](https://pubmed.ncbi.nlm.nih.gov/41997170/) | JNE 2026 | cross-dataset event segmentation | 좌표→pretrained channel embedding의 직접 선행기술이다. Facecat은 canonical frequency-coded SSVEP가 아니라 RSVP/P300 성격의 event endpoint다. 공개 code와 논문 수식에도 차이가 있다. |
| [Channel Adaptation for EEG FMs](https://arxiv.org/abs/2604.23091) | 2026 preprint | EEG 인접 | 4 adapter×5 FM 결과가 architecture-dependent이고 negative transfer도 있어 backbone×routing×seed ablation이 필요하다. |
| [EEG-PRIME](https://arxiv.org/abs/2608.13072) | 2026 preprint | heterogeneous EEG, SSVEP 없음 | task/dataset conditioning 자체의 novelty를 제거한다. held-out dataset의 learned embedding fallback은 원문에 없고 공개 repo는 감사 시점 404였다. |
| [MeCo](https://proceedings.mlr.press/v267/gao25p.html) | ICML 2025 | 언어모델 비유 | metadata dropout/cooldown의 아이디어만 전이한다. EEG 효능 근거로 쓰지 않는다. |
| [When Does Metadata Conditioning (NOT) Work?](https://arxiv.org/abs/2504.17562) | COLM 2025 | 언어모델 비유 | unknown/wrong context가 악화할 수 있으므로 missing·wrong·UNK control을 강제한다. |

LLM 논문 [arXiv:2608.11829](https://arxiv.org/abs/2608.11829)은 직접 EEG related work가 아니다. 평균이 좋아져도 새로 해결 가능한 문제의 집합이 늘지 않을 수 있다는 평가 비유만 채택해 `learned`, `forgotten`, `retained` subject-condition cell과 `learned−forgotten`을 보고한다. LLM의 `pass@K`와 EEG calibration의 `k`는 의미가 다르므로 직접 매핑하지 않는다.

## 9. Foundation-model target exposure 감사

“시험 때 k=0”과 “backbone이 pretraining에서 target raw EEG를 본 적 없음”은 다른 조건이다. 모델 revision마다 다음 matrix를 고정한다.

```text
backbone_revision × eval_dataset × exact_dataset_id
× raw_signal_exposure × same_subject_exposure × label_exposure
× evidence_source × confidence
```

현재 확인된 예시는 다음과 같다.

| backbone × 평가자료 | 공개 근거상 exposure | 해석 |
|---|---|---|
| REVE × Nakanishi / Lee2019 SSVEP / Kalunga | **raw pretraining exposure 확인** | inference k=0은 가능하지만 pristine unseen-representation이라고 쓰지 않는다. |
| REVE × Wang / BETA / Wearable / Dong / Kim / Mobile | appendix의 explicit list에서 exact ID 중복을 찾지 못함 | `no identified overlap`, 중간 confidence. 미노출로 단정하지 않는다. |
| EEGPT × Wang2016/TSUBenchmark | [공식 pretraining README](https://raw.githubusercontent.com/BINE022/EEGPT/main/datasets/pretrain/readme.md)에 S1–S35가 명시됨 | Wang 결과는 target-exposed foundation baseline으로 표시한다. |
| scratch/compact model × 모든 평가자료 | 외부 pretraining 없음 | 가장 깨끗한 비교 anchor다. |

REVE 논문은 Table 7에 MOABB 27개라고 쓰지만 바로 아래 명시 목록은 25개로 보인다. 따라서 목록에 없다는 사실은 “공개 부록에서 중복을 확인하지 못함”일 뿐, 노출이 없다는 증명이 아니다.

## 10. 충분한 실험설계의 세 단계

| 단계 | 요구사항 | 가능한 claim |
|---|---|---|
| 파일럿 | 현재 4 dataset, schema·label·split smoke, A0/A2, CCA/FBCCA | pipeline 실행 가능성과 방향성 |
| 좁지만 publishable | wearable 39명 fixed training×3 seeds + independent 60명 lockbox, exact-40 replication, parameter-matched A0/A2, 최신 strict k=0 baseline, full metadata falsification, exposure audit | unseen participant와 관측된 acquisition family에서 metadata의 증분 가치 또는 잘 통제된 부정 결과 |
| 강한 transportability | 실무 heuristic으로 ≥4 source acquisition regimes + model/spec 동결 뒤 untouched external target 2개, 각 핵심 metadata level이 독립 site/regime ≥2곳에 반복 | 여러 acquisition에 걸친 일반화 근거 |
| causal acquisition claim | prospective randomized within-person factorial/crossover, 가능하면 multi-site·multi-session | 특정 interface/device factor의 causal effect에 가까운 주장 |

`4 source + 2 target`은 보편적 정리나 통계적 최소치가 아니라, dataset ID shortcut과 단일-domain 우연을 드러내기 위한 실무적 권고다. 최종 충분성은 factor coverage, effect heterogeneity, target CI로 판단한다.

### 필수 반증 조건

metadata mechanism을 주장하려면 최소한 다음 순서가 성립해야 한다.

```text
correct metadata > shuffled ≈ metadata-null > deliberately wrong metadata
```

추가로 다음을 모두 본다.

- missing metadata 0/25/50/75/100%에서 graceful degradation
- unseen category를 trained `UNK`로 보냈을 때 붕괴하지 않음
- continuous physical descriptor와 categorical dataset ID 분리
- signal-derived proxy와 declared pre-query metadata 분리
- metadata-only가 label을 부당하게 예측하지 않음
- participant win rate, median, 10th percentile, worst group
- `fail→pass` learned와 `pass→fail` forgotten, 그리고 net expansion
- window overlap을 독립 N으로 세지 않은 subject-level paired CI

### 성공·축소·중단 규칙

- **Broad GO:** A2>A0가 사전지정 MCID를 넘고, negative control 방향이 맞으며, 최소 두 truly unseen acquisition에서 방향이 재현된다.
- **Narrow GO:** wearable에서만 강하고 external replication이 약하면 claim을 electrode-interface/unseen-participant로 제한한다.
- **Useful negative result:** 충분한 power·parameter matching·metadata quality·최신 baseline을 갖춘 뒤 효과가 없으면 benchmark와 함께 부정 결과를 보고한다.
- **Mechanism claim 중단:** shuffled/dataset-ID-only도 같은 효과를 내거나 wrong metadata가 악화시키지 않으면 physical metadata mechanism 주장을 하지 않는다.
- **Broad OOD claim 중단:** 한 acquisition cluster 또는 pretrained target-exposed backbone에서만 효과가 나면 일반화를 주장하지 않는다.

## 11. 문헌조사는 얼마나 더 해야 하는가

핵심 설계를 바꿀 만큼은 다시 읽었다. 현재 academic-research workspace에는 33개 search run에서 619개 discovery candidate, 13개 paper card, 17개 deep-read item이 있다. 이 숫자는 **619편을 검토했다는 뜻이 아니다.** broad search는 포화에 가깝고, 이번 재정독은 다음 결정축을 원문·공식 코드로 감사했다.

- 직접 strict k=0: DG-Conformer, TST-CSFR, TFA-Net, TBMSCCN-C, SSER, SSVEP-BiMA, Fast SSVEP
- foundation/metadata: REVE, EEG-PRIME, frozen montage adapter, Channel Adaptation
- data/provenance: Zhu wearable, Nakanishi, Kim2025, MobileBCI, AR binocular, Guttmann-Flury

남은 문헌 작업은 검색량 확대가 아니라 다음 공백 해소다.

1. Huang 2023 cross-subject DG의 full protocol과 reproducible artifact
2. TST-CSFR full equations 또는 저자 code
3. DG-Conformer 원 논문의 BETA/window/statistics와 사용 허가
4. SSER 원 논문의 headline statistical unit와 두 번째 dataset
5. EEG-PRIME의 unseen-dataset embedding fallback 및 code release
6. transductive SFDA/CSST 대표법의 공통 information-rights 표
7. Nakanishi data reuse/derived redistribution 서면 허가

각 논문에서는 다음을 계속 구조화해 뽑는다.

- 실제 target label/unlabeled 접근량
- subject/session/dataset split과 leakage 방지
- source-validation 존재 여부
- window와 channel, latency offset
- training seed와 통계 단위
- 사용 데이터가 backbone pretraining에 포함됐는지
- 공개 코드가 headline protocol을 실제 구현하는지
- 우리 A0/A2와 동일 자원점으로 재현 가능한지

2026년 프리프린트와 author-claimed acceptance는 peer-reviewed 결과와 분리한다. Semantic Scholar 429와 일부 publisher bot 제한이 있었지만 OpenAlex, Crossref, PubMed, 공식 proceedings, arXiv, OpenReview, 원 repository로 보완했다. 따라서 이는 evidence-grounded frontier review이지 완전한 systematic review라고 주장하지 않는다.

## 12. 바로 실행할 순서

### 48시간 내

1. wearable impedance mapping을 `[dry, wet]`로 수정하고 261.67/19.63 kΩ numeric-signature test를 추가한다.
2. 기존 impedance-conditioned processed asset/run을 무효화하고 새 dataset revision을 만든다.
3. per-channel impedance와 headband order를 manifest/schema에 보존한다.
4. `dataset_registry`에 role, exact version, license, usable participant, factor coverage, checksum, provenance를 동결한다.
5. Nakanishi 저자에게 data reuse·derived redistribution, s10 누락, acquisition detail을 문의한다.
6. Kim의 기존 shared-10 lockbox 지정을 폐기하고 exact-overlap 0인 frequency-only exploratory role로 바꾼다.

### 1주 내

1. [완료·outcome-free] Protocol 0.4의 A0/A2 공통 `m_struct`와 query-local QC, global `m_acq`, per-channel `m_quality`를 `physical_hybrid_v1`으로 구현한다.
2. [완료·diagnostic no-go] `DEC-20260901-004`의 exact six-role×3-fold 18-job과 **각 fold당 Full A2 intervention bundle 하나, 총 세 개**를 reveal #2로 공개했다. Potency·shortcut·observed safety screen은 통과했고 네 substantive gate가 실패했다.
3. paper-faithful FBCCA, Fast SSVEP, permission/independent-port DG-Conformer 계열을 공통 protocol로 실행한다.
4. MobileBCI는 OSF source의 modality typing을 복원하고, Guttmann은 40-trial/10–13 Hz 교정 manifest로 selective schema pilot을 한다.
5. AR는 v8 binocular codebook을 사용하되 impedance와 hardware BIDS field를 acquisition ground truth로 사용하지 않는다.

### protocol 동결 후

1. DEC-20260830-002의 39명 source는 v1에 한 번 사용했고 no-go였다. 나머지
   60명 independent lockbox는 미개봉 보존한다. V2는 독립 development cohort와 새
   decision contract가 있을 때만 confirmatory 절차를 검토한다.
2. Wang/BETA/Dong exact-40 replication을 실행한다.
3. 동결한 external dataset을 한 번만 연다.
4. subject-level paired inference와 learned/forgotten coverage를 산출한다.
5. 결과에 따라 narrow paper로 마감할지, prospective crossover를 수집해 causal/transportability claim으로 확장할지 결정한다.

지금은 held를 열거나 수십–수백 GB를 무차별 다운로드할 단계가 아니다. **V1 no-go 보존 → A_Q/M operator 수정 → 독립 wearable-like data·권리·provenance 확보 → 새 candidate/gate 사전등록 → 독립 development 판정 → 통과할 때만 held confirmatory freeze**가 현재의 가장 싸고 강한 경로다.

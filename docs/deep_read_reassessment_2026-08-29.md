# 연구설계 재정독 감사: 무엇을 고치고 나서 실험할 것인가

> **2026-09-04 superseded / 역사적 감사 문서:** 이 문서는 strict-k0 physical
> metadata 후보를 재감사한 당시 기록이며 현재 연구목표나 primary가 아니다. 현재는
> labeled target calibration 부담을 줄이는 `A_QM−A_Q`, k=0/1/3/5 설계를 따른다.
> [새 설계](metadata_calibration_efficiency_design.md),
> [프로토콜 현재 상태](research_protocol_status.md), [연구일지](research_log.md)를 우선한다.

기준일: 2026-08-29
상태: 설계·원자료·직접 경쟁법·foundation 선행연구 재감사 완료, confirmatory run 차단
적용 범위: 연구 주제는 유지한다. 즉, **target 사용자 EEG로 적응하지 않는 SSVEP 분류에서 물리적 획득 metadata의 증분 가치**를 검증한다.

> **후속 구현·결과 상태:** 이 문서는 패치 전 blocker를 기록한 감사다. P0 impedance 축·`wearable_v3` guard, 전체 102명 asset acceptance, structure/query-QC/external 권한 분리, `physical_hybrid_v1`, spectral anchor와 strict FBCCA는 이후 구현됐다. `DEC-20260901-004`의 physical six-role/18-job reveal #2도 완료됐으나, potency-valid assay에서 A0 `0.1292`, Full A2 `0.1194`, Δ `−0.0097`과 네 substantive gate 실패가 관찰돼 diagnostic no-go다. 현 candidate의 confirmatory는 차단되며 SESOI 등을 채우는 것만으로 재개할 수 없다. 현재 상태는 [연구 프로토콜 현재 상태](research_protocol_status.md)를 따른다.

## 1. 결론부터

연구 질문은 의미가 있고, wearable 자료는 좁은 주가설을 시험하기에 상당히 좋은 데이터다. 그러나 현재 상태로 full experiment를 시작하면 안 된다.

재정독에서 결론을 바꾼 핵심은 다섯 가지다.

1. **P0 데이터 결합 오류:** Figshare v4 `Impedance.mat`의 실제 수치상 axis index 0은 dry, index 1은 wet이다. 현재 설정은 `[wet, dry]`로 선언해 dry EEG에 wet impedance를, wet EEG에 dry impedance를 붙인다.
2. **Protocol 0.4는 문서 초안이지 구현된 실험계약이 아니다:** A0/A2 공통 구조정보, factorized acquisition 정보, 채널별 impedance, signal-derived QC 공정성은 아직 현재 schema/config에 구현되지 않았다.
3. **최신 baseline의 논문과 공개 코드는 자주 다른 실험을 한다:** TFA-Net 공개 코드는 sample-level K-fold, TBMSCCN 공개 코드는 개인별 block split, SSER 공개 평가는 무작위 중첩 crop을 사용한다. 이들을 그대로 strict k=0 headline baseline으로 실행할 수 없다.
4. **넓은 신규성 주장은 성립하지 않는다:** REVE와 2026 montage-adapter 연구가 좌표 기반 pre-query conditioning을 이미 수행했고, EEG-PRIME은 task/dataset conditioning을 수행한다.
5. **남는 신규성은 더 좁고 더 검증 가능하다:** dataset-ID lookup 없이 여러 물리 factor를 분해해 조건화하고, unseen factor level/combination에서 correct·missing·wrong·swapped metadata로 causal mechanism을 반증하는 strict SSVEP 연구다.

따라서 판정은 다음과 같다.

> **조건부 GO:** 주제는 유지한다. 다만 임피던스 매핑 수정·전처리 재생성·Protocol 0.4 최소 구현·기준선 프로토콜 정렬을 끝내기 전에는 확증 실험을 열지 않는다.

## 2. 연구설계를 가장 쉽게 설명하면

한 trial을 분류할 때 모델이 받는 정보권한을 두 상자로 나눈다.

```text
현재 query EEG ───────────────┐
                              ├─ A0: signal-only 예측
query 전에 아는 물리 metadata ─┤
                              └─ A2: signal + metadata 예측
```

여기서 target participant의 다른 EEG, target batch 평균·공분산, target label, 과거 target stream은 둘 다 보지 않는다. A0와 A2는 같은 backbone, parameter budget, source split, query, seed schedule을 사용한다. 유일한 처리 차이는 A2가 외부에서 미리 측정된 물리 정보를 받는다는 것이다.

쉬운 비유로는 같은 음성을 두 녹음기로 들었을 때, A0는 파형만 보고, A2는 “어떤 마이크·배선·접촉상태로 녹음했는지”도 안다. 단, 답을 맞힌 뒤 얻은 정보나 다른 시험 음성의 통계는 사용할 수 없다.

### 주가설

> **H1-W:** strict inductive k=0의 held-out participant에서, factorized electrode-interface와 pre-query contact metadata를 사용하는 A2가 구조정보와 query-local 신호정보를 동일하게 받는 A0보다 participant-level balanced accuracy를 높인다.

이 가설이 참이려면 평균 정확도만 오르면 부족하다.

- correct metadata가 null/shuffled/wrong metadata보다 좋아야 한다.
- dry와 wet 중 한쪽에서만 생긴 우연한 평균이 아니어야 한다.
- participant를 독립 단위로 paired CI를 계산해야 한다.
- A0가 실패하던 participant-condition을 A2가 새로 성공시키는 `learned`가 반대인 `forgotten`보다 많아야 한다.
- 외부 acquisition에서는 최소한 같은 방향이 재현돼야 broad OOD 주장을 검토할 수 있다.

## 3. 현재 wearable 실험이 실제로 식별하는 것

[Zhu et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7916479/)은 102명 모두에게 같은 12-target 과제를 wet와 dry headband로 실시했다. 각 headband에서 10 blocks를 수행했고 block 직전에 8채널 impedance를 기록했다. 착용 순서는 53명 dry-first, 49명 wet-first로 무작위화됐다.

이 설계가 강하게 보는 질문은 다음이다.

> source participant에서 이미 관찰한 wet/dry 두 조건을 바탕으로, 처음 보는 participant의 현재 EEG를 분류할 때 interface와 pre-block contact 정보가 도움이 되는가?

반대로 다음을 직접 식별하지는 않는다.

- 처음 보는 임의 device/site에 대한 보편적 일반화
- impedance만 변화시킨 독립적 인과효과
- electrode material, contact medium, geometry, amplifier를 각각 분리한 효과
- 처음 보는 stimulus class/frequency의 발견

### crossover라고 해서 모든 인과효과가 분리되는 것은 아니다

순서 무작위화는 wet/dry 비교를 강화한다. 그러나 첫 12-target 과제 뒤 unpublished 40-target 과제가 있고, 두 headband 과제 사이에 15–30분이 지나며, wet-first participant는 dry 전에 머리를 씻고 말렸다. 따라서 실제 비교는 다음이 묶인 **interface setup bundle**이다.

```text
wet/dry interface + 접촉상태 + 재장착 + 시간 + 피로 + 세척/carryover
```

headband order는 모델 입력이 아니라 통계적 covariate와 sensitivity analysis에 사용해야 한다. impedance는 headband와 강하게 결합돼 있으므로 `acquisition-only`, `quality-only`, `both` ablation이 필요하다.

## 4. P0: 현재 impedance 매핑은 반대다

원 논문 Figure 9는 dry 평균을 261.67 kΩ, wet 평균을 19.63 kΩ로 보고한다. Figshare version 4의 실제 `Impedance.mat`은 `(8, 10, 2, 102)`이고 전체 값을 직접 계산하면 다음과 같다.

| axis-2 index | 전체 평균 | 논문과 일치하는 조건 |
|---:|---:|---|
| 0 | 261.6748994896 kΩ | dry |
| 1 | 19.6316962168 kΩ | wet |

즉 배포 README/논문의 한 axis 설명과 실제 값이 모순되며, **실증적으로 맞는 impedance order는 `[dry, wet]`**이다.

현재 [wearable.yaml](../configs/data/wearable.yaml)은 `impedance_electrode_types: [wet, dry]`이고, [prepare_wearable.py](../src/cfeg/data/prepare_wearable.py)는 label의 list index로 값을 고른다. [test_wearable.py](../tests/test_wearable.py)는 이 반대 매핑을 정답으로 고정한다.

영향은 명확하다.

- dry EEG sample에는 wet 수준의 낮은 impedance가 붙는다.
- wet EEG sample에는 dry 수준의 높은 impedance가 붙는다.
- 지금까지의 impedance-conditioned processed asset과 A2 pilot은 metadata 효과의 증거로 사용할 수 없다.
- 이 오류는 단순 label typo가 아니라 treatment가 반대로 결합된 오류다.

confirmatory 실행 전 필수 조치는 다음이다.

1. mapping을 `[dry, wet]`로 수정한다.
2. paper-reported numeric signature를 검사하는 regression test를 추가한다.
3. 기존 wearable processed asset의 revision을 폐기하고 새 revision으로 전처리한다.
4. subject×condition×block별 impedance가 예상 범위와 방향인지 manifest audit를 만든다.
5. EEG electrode axis `[dry, wet]` 자체는 별도의 online-result/저자 확인으로 독립 검증한다.
6. 8채널 값을 mean/max로만 압축하지 말고 per-channel vector와 missing mask를 보존한다.

## 5. Protocol 0.4와 현재 구현 사이의 차이

Protocol 0.4 문서는 올바른 방향이지만 현재 코드는 그 실험을 아직 수행하지 않는다.

| 의도한 계약 | 현재 상태 | 왜 문제인가 |
|---|---|---|
| `m_struct`를 A0/A2 모두에 제공 | backbone 일부는 channel ID/mask를 쓰지만 treatment 계약이 명시적으로 분리되지 않음 | geometry 경로 차이를 metadata 효과로 오해할 수 있음 |
| factorized `m_acq` | 범주형 reference/electrode/cap과 소수 scalar 중심 | unseen category가 `UNK`가 되면 물리적 extrapolation을 하지 못함 |
| per-channel `m_quality` | impedance mean/max만 사용 | contact의 공간 패턴과 reference mismatch를 잃음 |
| query-derived QC는 A0/A2 공통 | 현재 per-trial channel z-score 뒤의 signal이 주 입력 | raw query mean/std·amplitude·quality 권한을 A0에 공정하게 주는 경로가 없음 |
| preprocessing×fusion 2×2 | 아직 개발 experiment/config가 없음 | learned fusion의 순증분을 식별할 수 없음 |
| frozen compact primary | 기존 README는 frozen REVE final로 서술했음 | REVE exposure·weight license·1 s 제약 때문에 primary로 동결되지 않으며 README를 정정함 |

특히 [Fast SSVEP](https://arxiv.org/abs/2506.01284)은 각 **현재 query 자체의** mean/std를 사용하는 query-local modulation을 한다. 이것은 target-set 통계가 아니므로 strict inductive 조건에서 허용할 수 있다. 따라서 A2가 외부 impedance를 받는 동안 A0가 z-score 때문에 query의 amplitude/scale 정보를 잃는다면 정보권한 비교가 불공정해질 수 있다.

권장 정보권한은 다음처럼 동결한다.

| 정보 | strict A0 | A2 | 비고 |
|---|---:|---:|---|
| 현재 query waveform | 허용 | 허용 | 동일 전처리 |
| 현재 query에서 causal하게 계산한 QC/mean/std | 허용 | 허용 | 같은 extractor와 parameter |
| channel mask/좌표/time grid | 허용 | 허용 | 구조정보, treatment 아님 |
| query 전에 외부에서 측정한 impedance/acquisition descriptor | 미사용 | 허용 | primary treatment |
| target participant의 다른 trial 통계 | 금지 | 금지 | transductive protocol로 분리 |
| target batch covariance/normalization fit | 금지 | 금지 | transductive protocol로 분리 |
| target label/target checkpoint selection | 금지 | 금지 | protocol 위반 |

## 6. 직접 SSVEP baseline을 원문·코드로 다시 판정한 결과

논문의 calibration-free라는 이름이 아니라 실제 target-data 권한, split, checkpoint 선택, query 생성으로 분류했다.

| 방법 | 논문상 위치 | 공개 artifact 감사 | 이 연구에서의 처리 |
|---|---|---|---|
| CCA / paper-faithful FBCCA | training-free | 가장 명확한 strict k=0 | 필수 공통 anchor |
| DG-Conformer | source-only cross-subject | held target은 최종 평가에만 사용; 공개 코드는 Benchmark 0.8 s만 지원, source trial-level validation, license 없음 | 가장 가까운 learned A0 후보. 저자 허가 또는 독립 port 후 공통 protocol로 실행 |
| TST-CSFR | source filters/templates, calibration-free 보고 | 원문·공식 코드 미확보 | literature-only; 수식/코드 확보 전 executable bundle 제외 |
| TFA-Net | calibration-free 보고 | 공개 코드는 5 s trial의 50% overlap window를 만들고 shuffled sample KFold; participant/trial group 없음. checkout에는 runtime-invalid 호출도 있음 | headline strict A0에서 제외; architecture만 participant/trial-disjoint 재구현 후보 |
| TBMSCCN-C | 논문은 Benchmark/BETA source-only LOSO | 공개 코드는 개인별 5-block train/1-block test이며 target block을 매 epoch 평가; 논문 headline code가 아님 | 공개 runner 제외. CC BY author manuscript 기반 독립 구현만 고려 |
| SSER | subject-independent augmentation | target 학습은 없지만 test가 block당 5,000개 무작위 복원·중첩 crop, seed 없음 | classifier headline 제외; training-only augmentation ablation으로 이식 |
| Fast SSVEP | training-only remix + query-local modulation | Benchmark/BETA/Nakanishi LOSO; validation·seed·filter 세부 일부 누락 | short-window signal-only comparator; preprint와 불확실성 표시 |
| SSVEP-BiMA | raw+complex FFT, LOSO | Nakanishi/MAMEM-II 결과; model selection·seed·multiplicity·code 미보고 | 해당 데이터 도입 시 phase-aware A0 comparator로 필수, literature/independent implementation |

핵심 원칙은 세 열을 분리하는 것이다.

1. 원 논문이 보고한 수치
2. 후속 연구가 재보고한 수치
3. 우리가 공통 protocol로 재실행한 수치

서로 다른 window, channel, sampling, split, target 접근의 headline 숫자를 한 ranking에 넣지 않는다. 공통 실행에서는 source-participant validation, 동일 fixed query, participant 단위, 최소 3 seeds, target final-only를 강제한다.

### 최소 실행 bundle

- CCA와 dataset별 paper-faithful FBCCA
- compact scratch A0와 A2
- protocol-matched DG-Conformer 계열 port 하나
- Fast SSVEP signal-only short-window comparator
- SSER mixing의 `augmentation × metadata` factorial ablation
- Nakanishi를 쓰는 경우 BiMA
- TST-CSFR와 TBMSCCN-C는 재현 가능한 독립 구현이 완성될 때만 승격

## 7. foundation·metadata 선행연구가 신규성을 어떻게 바꾸는가

### 이미 선행된 것

- [REVE](https://proceedings.neurips.cc/paper_files/paper/2025/hash/20a917f77773ac0fa8bea2bdd6606b66-Abstract-Conference.html)는 전극의 3D 좌표와 시간을 4D positional encoding으로 query 생성 전에 사용한다.
- [EEG-PRIME](https://arxiv.org/abs/2608.13072)은 task text와 learned dataset embedding으로 Q-Former를 layer-wise modulation한다.
- [Montage-agnostic frozen adapter](https://doi.org/10.1088/1741-2552/ae6142)는 좌표 거리로 pretrained channel embedding을 합성해 frozen EEGPT에 넣는다.
- [Channel Adaptation](https://arxiv.org/abs/2604.23091)은 spherical-spline interpolation, spherical harmonics, learned mapping, target-covariance alignment을 비교한다.

따라서 다음 표현은 철회한다.

- “전극 geometry를 pre-query에 넣은 최초 EEG 방법”
- “metadata conditioning을 사용한 최초 EEG 방법”
- “미지 montage에 물리적으로 일반화한 최초 방법”

### 아직 방어 가능한 질문

현재 정독 범위에서 다음 조합을 canonical frequency-coded SSVEP strict k=0으로 검증한 명확한 선행 사례는 찾지 못했다.

- 좌표뿐 아니라 reference, contact/interface, filter/sampling, impedance, stimulus physics를 독립 factor로 표현
- dataset-ID lookup 없이 관측 가능한 물리량에서 embedding 생성
- unseen factor level과 unseen factor combination을 분리 평가
- correct/missing/random/sample-swap/dataset-swap/impossible metadata로 causal robustness 검증
- target label뿐 아니라 target-set 통계도 사용하지 않는 family-level holdout

안전한 novelty 문장은 다음이다.

> **lookup-free, factorized physical pre-query conditioning을 strict family-level·compositional zero-shot SSVEP에서 평가하고, 잘못된 metadata가 예측에 미치는 방향까지 반증한다.**

REVE는 secondary backbone으로 가치가 있지만 깨끗한 primary anchor는 아니다. pretraining 목록에 Nakanishi, Lee2019 SSVEP, Kalunga가 있고, gated weights의 license는 code의 MIT와 다르다. scratch compact model을 primary로 두고 REVE는 exposure matrix와 사용조건을 명시한 secondary로 둔다.

## 8. 데이터는 더 필요하지만, 역할이 먼저다

### 당시 제안 primary

- **Wearable/Zhu2021:** 102명의 unseen-participant mechanism test. 두 이미 관찰된 wet/dry regime에 대한 좁은 claim.

### 가장 가까운 외부 복제

- **Nakanishi 12JFPM:** wearable과 문자·위치·frequency·phase가 12/12 정확히 같다. 원 repo는 s1–s10인데 MOABB/NEMAR는 9명만 노출하고, MOABB interval도 stimulus 뒤 구간을 포함할 가능성이 있다. 데이터 license가 명확하지 않으므로 원 repo s1–s10을 기준으로 저자 허가를 받은 뒤 사용한다.

### acquisition/context stress

- **Wang/BETA/Dong exact-40:** Wang과 BETA는 한 near-identical acquisition cluster로 세고, Dong이 별도 semi-dry pediatric regime를 제공한다.
- **Kim2025 BetaRange:** 40명, BioSemi ActiveTwo, 14.0–21.8 Hz, 0.2 Hz 간격이다. Wang과 frequency 10개는 겹치지만 시작 index가 30칸 달라 JFPM phase가 모두 π만큼 어긋난다. 따라서 exact `(frequency, phase)` overlap은 0이며, 현재 fixed class head의 “shared exact-10 lockbox”가 아니다. frequency-only remapping을 별도 exploratory protocol로 정의해야 한다.
- **MobileBCI:** SSVEP usable 23명에서 scalp/ear가 같은 trial에 동시 기록되어 sensor-location contrast가 강하다. 그러나 standing→slow→fast→running 순서가 고정돼 speed와 time/fatigue가 교란된다. MOABB/NEMAR derivative는 27 IMU channels까지 EEG로 취급하고 raw/processed reference도 혼동하므로 OSF source를 직접 사용한다. 3-class 별도 head다.
- **AR binocular dataset:** condition/subsession 순서가 무작위인 binocular frequency/phase mechanism positive-control다. acquisition hardware는 고정이고, 확인한 BIDS impedance file은 전부 `n/a`였으며 JSON의 amplifier도 논문과 불일치했다. 따라서 acquisition metadata 자원이 아니라 conventional 12/40-class pooled head와 분리된 stimulus-mechanism endpoint다.
- **Gu 1–60 Hz:** frequency·modulation-depth·multi-day representation 실험에 좋지만 primary label family가 아니다.
- **Guttmann-Flury2025:** 원 논문 기준 31명·63 sessions·session당 40 trials·10/11/12/13 Hz다. MOABB/NEMAR의 48 trials·총 3024·`[8,10,12,15]`·LED 표기는 틀렸다. 이 값을 교정한 manifest를 고정하면 randomized paradigm position과 repeated-session context stress로 가치가 있다. phase가 미보고이고 target semantics가 달라 pooled exact-class head에는 넣지 않는다.

participant 수보다 중요한 것은 독립 acquisition regime와 factor 반복이다. broad transportability를 쓰려면 같은 factor level이 서로 다른 site/regime에서 반복되고, model/spec 동결 뒤 untouched external target을 최소 두 개 두는 것이 바람직하다. 이는 통계적 법칙이 아니라 dataset-ID shortcut을 드러내기 위한 실무 gate다.

## 9. 수정된 연구설계

### Stage A — P0 복구

1. impedance 축 수정과 numeric-signature regression test
2. wearable 전처리 revision 재생성 및 기존 impedance-conditioned artifact 무효화
3. per-channel impedance와 headband order 보존
4. S1–S3 smoke 재실행: 성능 결론이 아니라 mapping·schema·split 검증

### Stage B — source-only 설계 동결

1. `m_struct`, `m_acq`, `m_quality`, confound를 schema에서 분리
2. A0/A2 공통 query-local QC 경로 구현
3. B0/F/P/P+F preprocessing×fusion 2×2
4. source participant만 이용한 compositional pseudo-OOD로 architecture 선택
5. correct/missing/shuffle/wrong/swap/noise controls 통과
6. compact scratch primary, REVE secondary를 선언
7. MCID, success threshold, seeds, multiplicity family, primary window 동결

### Stage C — primary confirmatory

- wearable 5-fold outer participant CV
- cohort freeze 결정: primary N=102를 유지하되 무효 v2 pilot과 v3 schema-only smoke를 공개하고, S1–S3 제외 N=99를 sensitivity로 별도 보고
- fold 내부 source-participant validation
- A0/A2 같은 fold·query·parameter budget·seed schedule
- 2.0 s를 primary window로 유지하되 0.3/0.5/0.7/1.0/1.5/2.0 s curve는 secondary family로 사전지정
- subject별 wet/dry BA, interaction, worst group, calibration, negative transfer 보고
- paired A2−A0 CI와 Holm family

### Stage D — replication과 lockbox

1. Wang/BETA/Dong exact-40을 protocol-matched acquisition replication으로 실행
2. Nakanishi 권리·10명 audit 완료 시 exact-12 replication
3. Kim은 phase-aware remapping이 사전지정된 별도 frequency-only stress로만 사용
4. MobileBCI/AR는 controlled-context secondary study로 사용
5. 모든 선택 후 external target을 한 번만 열기

## 10. Go/No-Go 규칙

| 결과 | 판정 |
|---|---|
| A2>A0가 MCID를 넘고 correct>null/shuffled>wrong, wearable 양 조건 및 외부에서 방향 재현 | broad GO 후보 |
| wearable에서만 재현 | narrow GO: electrode-interface/unseen-participant 논문 |
| 평균은 오르나 shuffled/wrong도 같음 | physical mechanism claim 중단; capacity/shortcut으로 해석 |
| A0 공통 QC를 주면 A2 이득 소실 | 외부 metadata가 아닌 signal-quality path가 기전이라는 유용한 부정 결과 |
| A2가 worst group/forgotten을 악화 | 배포 안전성 claim 중단 |
| 충분한 power·최신 기준선·정확한 metadata 후 효과 없음 | 잘 통제된 negative benchmark로 보고 가능 |

## 11. 최종 판단

이 연구의 약점은 “아이디어가 의미 없어서”가 아니다. 현재 가장 큰 위험은 **잘못 결합된 핵심 metadata와 불공정한 정보권한, 그리고 protocol이 다른 최신 방법의 숫자를 직접 비교하는 것**이다. 이를 고치면 wearable 102명은 좁은 주가설에 충분히 강하고, 부정 결과도 의미가 있다.

반대로 current four-dataset suite만으로 arbitrary device/site OOD나 metadata 각 factor의 인과효과를 쓰는 것은 과장이다. 가장 현실적인 첫 논문은 다음 범위다.

> 처음 보는 participant의 known wet/dry interface 조건에서 pre-query physical metadata의 순증분과 실패 조건을 엄격히 검증하고, exact-label acquisition family에서 방향을 복제한다.

이 범위를 통과한 뒤에만 prospective multi-site randomized factorial collection으로 확장한다.

## 12. 증거 한계와 실패 기록

- DG-Conformer, TST-CSFR, SSER의 publisher full text는 접근 제한이었다. 공식 초록·메타데이터와 pinned official repository를 사용했고, 논문 headline 통계는 인증하지 않았다.
- TBMSCCN은 University of Essex의 CC BY author manuscript를 전부 읽었지만, 공개 GitHub code는 headline cross-subject protocol과 다르다.
- TFA-Net journal full text는 확보하지 못했으므로 공개 code의 split 결함을 논문 자체에 그대로 귀속하지 않는다.
- JNE montage-adapter의 공식 PDF endpoint는 WAF로 막혀 공식 기관 metadata, 접근 가능한 full publisher-text mirror, pinned official code를 교차검토했다.
- EEG-PRIME의 논문상 GitHub 주소는 감사 시점에 404여서 unseen-dataset embedding policy를 검증하지 못했다.
- 이번 조사는 검색 후보 수가 많은 evidence-grounded frontier review이지 PRISMA systematic review나 meta-analysis가 아니다.

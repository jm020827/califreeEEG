# Protocol 0.4-dev 구현·의사결정 기록

기준일: 2026-08-30
상태: physical A2 reveal #2 완료, valid diagnostic assay에서 substantive no-go, confirmatory 차단

> **2026-09-04 superseded / historical only:** 이 문서는 retired `physical_hybrid_v1`의 구현
> 기록이다. `DEC-20260904-005` 이후 현재 질문과 실행 gate는
> [metadata-assisted low-calibration 설계](metadata_calibration_efficiency_design.md)를 따른다.

> **2026-08-30 update / DEC-20260830-001·002:** 이 문서의 과거 `N=102 primary + N=99 sensitivity`와 `N=99 5-fold primary` 결정은 superseded됐다. S1–S3는 영구 development-only이고 S4–S102는 39명 training·60명 independent lockbox다. 현재 권한·실행 상태는 [연구 프로토콜 현재 상태](research_protocol_status.md)와 [연구일지](research_log.md)를 우선한다.

## 1. 연구목표 결정

당시에는 연구목표를 strict-k0로 유지하고 한 데이터셋으로 입증할 수 있는 주장과 장기 목표를 분리했다.

> **당시 primary 질문:** source participant에서 wet/dry 조건을 학습한 뒤, target participant의 다른 trial이나 label을 전혀 사용하지 않는 strict k=0에서, query 전에 외부에서 관측한 물리 metadata가 waveform·공통 구조·query-local QC만 사용하는 parameter-matched control보다 순증분을 주는가?

현재 wearable 실험으로 바로 주장하지 않는 것은 다음이다.

- 처음 보는 임의 device/site/montage에 대한 보편적 OOD 일반화
- 새로운 stimulus class나 frequency의 discovery
- open-set OOD detection
- target stream을 사용한 test-time/continual adaptation
- impedance 하나만 조작한 독립 인과효과

따라서 주제가 바뀐 것이 아니라 `broad OOD`를 장기 외부복제 목표로 내리고, 먼저 식별 가능한 `unseen-participant × electrode-interface` 가설을 검증한다.

## 2. 실험을 가장 쉽게 설명하면

두 모델은 같은 EEG와 같은 시험문제를 본다.

```text
공통: 현재 EEG + 어느 채널이 있는지 + 현재 query에서 계산한 신호 QC
                         │
             ┌───────────┴───────────┐
             │                       │
 A0 external-null             A2 external-observed
 global residual=0            electrode type + block impedance
 channel gain=1               + channel별 impedance를 관측
```

A0와 A2의 backbone, physical conditioner module graph, parameter shape, participant split, optimization schedule은 같다. 처리 차이는 `external_metadata_mode=null|observed` 하나다. A2는 global 정보는 zero-init bounded FiLM으로, channel impedance는 spectral projection 뒤·channel embedding 앞의 bounded gain으로 넣는다. A0에서는 같은 module을 유지하되 FiLM residual은 0, channel gain은 1이다. 따라서 paired `A2 − A0`가 외부 pre-query metadata access의 순증분을 가장 가깝게 측정한다.

F0는 condition module이 없는 더 작은 signal baseline이고 A0의 대체물이 아니다. A1 dataset-ID prompt와 과거 prompt+adapter는 shortcut/architecture 진단용 legacy ablation이며 primary가 아니다. 안정적인 artifact ID `A2_structured_condition_prompt`는 유지하지만 현재 primary A2는 prompt token을 쓰지 않는다. 상세 수식과 한계는 [A2 physical-hybrid 설계 계약](a2_physical_hybrid_design.md)에 있다.

## 3. 정보권한 계약

| Bundle | 예 | A0 | A2 | external missing/shuffle |
|---|---|---:|---:|---:|
| `m_struct` | canonical channel ID/mask, processed time grid | 공통 | 공통 | 보존 |
| `m_query` | filter/crop 뒤 z-score 전 query channel std | 공통 | 공통 | 보존 |
| `m_external_global` | electrode type, impedance mean/max와 availability | null | observed | 제거/교환 |
| `m_external_channel` | canonical slot별 impedance와 availability | null | observed | 제거/교환 |
| analysis confound | headband order, first/second period, reattach, elapsed time | 미사용 | 미사용 | 해당 없음 |
| primary 제외/금지 | constant reference/cap, dataset/hardware/subject/session/trial ID, label/frequency/phase, source file | 금지 | 금지 | 해당 없음 |

`m_query`는 현재 query 하나만으로 계산하는 pure per-sample 정보다. target batch 평균·공분산이나 다른 target trial을 사용하지 않는다. signal perturbation으로 waveform과 저장 QC의 정합성이 깨지면 QC를 missing으로 바꾼다.

## 4. 이번 구현에서 바뀐 것

### P0 wearable 복구

- 배포 Readme의 impedance-axis 문구와 실제 수치가 모순됨을 기록하고, 논문 조건 평균과 일치하는 numeric order `[dry, wet]`로 수정했다.
- axis 전체 평균이 약 `261.6749/19.6317 kΩ`인지 runtime과 test에서 확인한다.
- 기존 `wearable_v2`를 무효 revision으로 남기고 새 출력은 `wearable_v3`로 분리했다.
- checkpoint와 evaluation asset의 revision이 다르거나 provenance가 없으면 실행을 거부한다.
- full preset은 102명·24,480행 completeness와 deep audit receipt 없이는 실행되지 않는다.
- impedance의 NaN 채널 위치를 제거하지 않고 native 8-vector를 canonical 64 slots에 보존한다.
- `Subjects_Information.mat`에서 `Headband to wear first`를 header로 찾아 102명·dry-first 53명·wet-first 49명을 검증한다.
- headband order와 파생 `condition_period=first|second`는 manifest 분석 covariate로만 저장한다.

### A0/A2 physical-hybrid 공정성

- Protocol version을 `0.4-dev`로 명시한다.
- Physical conditioner가 channel ID/mask를 중복 입력받지 않도록 primary config에서 `include_channels=false`로 고정했다. 채널 구조는 backbone 공통 경로에만 남는다.
- mixed legacy continuous vector와 별도로 `external_continuous[impedance_mean, impedance_max]`와 `query_qc[pre-zscore signal_std]`를 생성한다.
- Primary categorical treatment는 `electrode_type` 하나다. `reference`와 `cap_type`은 이 asset에서 상수라 제외하며, `dataset_id`, opaque `hardware_id`, `reattach_flag`도 입력하지 않는다.
- Global electrode/mean/max는 bounded zero-init FiLM, channel impedance는 shared non-monotonic bounded gain으로 분리한다. Gain은 signal token에만 적용하고 channel identity embedding은 보존한다.
- A0는 외부 categorical/continuous/channel impedance를 null+missing으로 받고 query QC는 유지한다. 같은 weights에서 external-null 및 all-external-missing은 학습 뒤에도 algebraic identity다.
- Unknown categorical ID는 학습된 임의 unknown 의미가 아니라 fixed zero/neutral이다. 높은 impedance를 무조건 downweight하는 단조 규칙은 사용하지 않는다.
- metadata missing/shuffle는 external bundle만 바꾸며 structure/query QC를 보존한다.
- shuffle는 acquisition block을 derange한 뒤 donor block의 exact label/window 행을 함께 교환한다. Donor mapping, mapping hash, label alignment와 effective metadata-change fraction을 저장한다.
- training/eval signal augmentation 뒤 stale query QC를 missing 처리한다.
- A0/A2 실행은 parameter 수와 protocol/fairness, parameter schema, 초기 trainable state, split, vocabulary, exact asset bytes, source tree, training/inference environment hash를 남기며 pair 완료 시 동일성을 강제한다.
- Prediction schema `cfeg.predictions.v3`는 fixed-epoch checkpoint와 completion receipt, held-out `split.csv`, analysis manifest, six-job execution manifest, reveal receipt, CSV sidecar를 연결한다. 다른 checkpoint·scenario·seed·source revision을 섞으면 분석을 거부한다.
- Confirmatory plan은 canonical manifest와 run root를 하나만 허용한다. Manifest는 create-exclusive이고, 실행 산출물이 없는 unauthorized dev draft만 명시적 교체가 가능하다. Generic evaluation과 canonical hidden staging 밖 prediction은 reveal receipt가 있어도 차단한다.
- Exact seeds `[42,43,44]`의 A0/A2 6 jobs를 39명에서 학습하고, 독립 60명 lockbox에서 subject별 seed BA를 먼저 평균한 뒤 paired inference를 한 번만 수행한다. Condition cell transition은 descriptive다.

### 구조·foundation secondary 안전성

- unknown channel이 우연히 canonical tensor slot 번호의 known electrode ID로 변환되지 않게 identity `0`을 보존한다.
- REVE는 mixed-mask batch의 union montage를 모든 sample에 넣지 않고, 동일한 active ID/mask를 가진 sample끼리만 묶어 forward한다.
- padding token을 mean pooling과 prompt attention에서 제외한다.
- REVE의 in-place position noise가 cache를 변형하거나 expanded view를 쓰지 못하도록 clone하고, active unknown channel은 조용히 버리지 않고 중단한다.

## 5. 완료된 physical assay와 남은 연구 항목

- [완료] physical six-role/18-job grid(A0/full/global-only/channel-only/full-shuffle-train/metadata-only)의 두 번째 reveal 승인과 margin 동결, validator-only 재시작 승인 (`DEC-20260901-004`)
- [완료·no-go] 18/18 train·3/3 intervention·atomic publication과 사전 margin 판정. Potency와 observed shortcut/safety screen은 통과했지만 clean direction·counterfactual reliance·inference pairing·training pairing gate가 실패
- factorized continuous descriptors의 충분한 외부 데이터 반복
- compact scratch backbone의 nominal xyz registry와 provenance 계약
- 최신 learned strict-k=0 baseline 전체(paper-faithful Chen-2015 FBCCA와 spectral scratch anchor는 구현 완료)
- source-only pseudo-OOD로 정한 architecture/MCID/success threshold/multiplicity와 `configs/analysis/wearable_primary.yaml`의 confirmatory freeze
- [차단] 현 candidate의 39-train/60-lockbox six-job confirmatory 실행
- confirmatory SESOI·alpha/alternative·operational threshold·multiplicity·fairness/source lock의 최종 동결(physical margin과 `development_gate_only`는 완료)
- REVE에서 sample-wise random montage가 만드는 직렬 forward/cache 증가에 대한 성능 gate
- frozen REVE/position-bank Hugging Face snapshot revision·remote-code·weight byte fingerprint를 checkpoint와 evaluation에 묶는 provenance gate

즉 현재 코드는 `완성된 보편적 acquisition mechanism`이 아니다. 물리 위치에 맞춘 fusion과 exact fallback은 구현·검증됐지만, **유효한 N=3 mechanism assay에서 correct metadata 이득과 의존성을 보이지 못한 physical-hybrid 후보**다. 첫 prompt grid와 두 번째 physical grid 모두 개발 결과이며 모집단 결론은 아니다.

## 6. 연구 시작 순서

1. [완료] S1–S3 `wearable_v3` 720행을 재생성했다. Figshare v4 5개 파일 MD5, raw EEG/query-QC/impedance 전수 대조의 최대 오차가 모두 0이었고 deep audit receipt를 생성했다.
2. [완료·역사적 prompt] 첫 `prompt_adapter_v1` Pilot A0/A2 dry-run에서 양쪽 모두 총/학습 parameter `6,125,900`, parameter-schema·초기-state·split·fairness hash가 동일함을 확인했다. 이는 성능 결과가 아니다. DEC-20260830-001에 따라 S1–S3는 영구 development-only이고, DEC-20260830-002에 따라 S4–S102의 99명 pool은 39명 training과 60명 independent lockbox로 고정됐다.
3. [완료] 전체 102명 raw asset의 고정 Figshare v4 inventory·MD5와 support MAT를 확인하고 `wearable_v3`를 생성했다.
4. [완료] acceptance audit에서 24,480 rows, 102 subjects, condition별 12,240 rows, order 53/49를 확인했다. Raw signal/query-QC/impedance/channel alignment의 전수 대조 최대 절대오차는 모두 0이었고 receipt를 고정했다.
5. [완료·역사적] Prompt A0/A2의 첫 fixed-epoch S1–S3 grid는 A0/A2 BA 0.1222/0.1042, delta −0.0181이었다. 이는 directional warning이며 새 physical A2 성능이 아니다.
6. [완료·outcome-free] `physical_hybrid_v1`, exact-null unit contract와 block-coherent donor mapping을 구현했다. Reference/cap은 primary에서 제외했다.
7. [완료·outcome-free] Six-role×3-fold 18-job manifest, Full A2의 사전 고정 missing/shuffle/wrong-metadata 개입 3개 fold bundle, prediction·intervention·aggregate·gate를 함께 tree-hash하는 hidden staging을 구현했다. 완성 digest의 ledger/precommit receipt로 reveal 예산을 먼저 소비하고, atomic rename·parent fsync 뒤 final publication receipt를 쓰는 two-phase 공개와 동일 private artifact의 deterministic crash recovery를 결합했다.
8. [완료] Mechanism/safety margin·중단 규칙과 owner 승인 ID/시각을 `DEC-20260901-004` decision receipt에 고정했다. 첫 기술 실행은 공개 전 validator 오류로 격리했고, 정정본은 clean tag에서 처음부터 18개 job을 재학습했다.
9. [완료·no-go] 정정 실행의 potency와 publication은 유효했으나 네 substantive gate가 실패했다. Reveal #2는 소비됐고 동일 S1–S3에서 physical margin·model·epoch·seed를 재결정하거나 추가 reveal하지 않는다.
10. [차단] 현 candidate로 exact 39/60×seed `[42,43,44]` plan을 `frozen`으로 바꾸거나 training/lockbox prediction을 수행하지 않는다. 계속하려면 독립 외부 development data와 별도 사전등록 모델/decision contract가 필요하다.

전체 asset acceptance 명령은 다음과 같다.

```bash
python scripts/audit_wearable_processed.py \
  --raw-dir "$EEG_DATA_ROOT/raw/wearable" \
  --processed-dir "$EEG_DATA_ROOT/processed/wearable_v3" \
  --expected-subjects 102
```

확증 결과의 최소 판정은 평균 BA만이 아니다. participant-level paired CI, wet/dry별 효과, worst-group, `learned − forgotten`, correct>null/shuffled/wrong metadata 방향을 함께 만족해야 한다. wearable에서만 재현되면 결론도 `unseen participant의 electrode-interface bundle`로 제한한다.

## 7. 연구목표를 실제로 바꿔야 하는 조건

다음은 연구목표를 재검토하게 만드는 사전 조건이었다. Reveal #2에서는 현재 candidate와 S1–S3에 한해 첫 두 조건에 해당하는 관찰이 나왔다.

- A0에 공통 query QC를 준 뒤 A2 이득이 반복적으로 0에 가까움
- shuffled/wrong metadata가 correct metadata만큼 좋음
- 효과가 participant ID나 dataset missingness shortcut으로만 설명됨
- wearable의 wet/dry 한쪽에서만 불안정하게 나타나고 외부 acquisition에서 방향이 뒤집힘

현재 가능한 결론은 **`physical_hybrid_v1`이 이 N=3 assay에서 external metadata benefit이나 correct-pairing reliance를 보이지 않았다**는 데 한정한다. 외부 metadata가 모집단에서 불필요하다거나 signal-derived QC가 핵심이라고 일반화할 수 없다. 새로운 class discovery/OOD detection 주제로 즉시 이동하는 것도 이 candidate의 실패를 해결하지 않는다.

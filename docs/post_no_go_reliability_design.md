# No-go 이후 reliability-spatial-v1 연구설계

기준일: 2026-09-03

상태: **합성 engineering Stage-0 terminal FAIL, 조건부 Stage-1 미실행**

기계 계약: `configs/analysis/reliability_spatial_v1.yaml`

candidate-plan SHA-256: `6227f2e06db09dec0c18768454f18c8e4866f7ef0b6d29e18aef71c00ad63b39`
Stage-0 plan SHA-256: `985b4f84ec8110d8e7a711b1d1bc0a873d81eb8a0bc5c5cf49febca79b945e91`

> **2026-09-04 terminal historical document:** 아래 문장은 이 candidate를 동결할 당시의
> 질문이다. Stage-0 실패 뒤 이 exact Q+M spatial operator는 재실행하지 않는다. 현재
> low-calibration 후보는 waveform operator가 아니라 frozen Q-only calibration estimator 위의
> target-support prior residual이며 [새 설계](metadata_calibration_efficiency_design.md)를 따른다.

## 한 문장 결론

동결 당시 연구목표는 **처음 보는 사람의 EEG를 그 사람의
labeled calibration 없이 분류할 때, query 전에 측정한 획득 metadata가 실제로 추가
도움이 되는가**였다. 당시 바뀐 것은 metadata를 넣는 위치와 그것을 검증하는 순서였다.

기존 `physical_hybrid_v1`은 metadata로 decoder 전체 표현을 조절했다. 유효한 S1–S3
mechanism assay에서 correct metadata 의존성과 성능 이득이 나오지 않았으므로 이 후보는
종료했다. 새 후보는 현재 EEG에서 채널 상태를 먼저 읽고, impedance 같은 metadata는 그
판단에 작은 보정만 더한다.

## 아주 쉽게 설명하면

8개 EEG 채널을 8명의 관측자라고 생각할 수 있다. 어떤 순간에는 한 관측자의 접촉이
나쁘거나 잡음이 심할 수 있다.

1. **Q 경로**는 지금 들어온 2초 EEG만 보고 각 관측자가 얼마나 시끄러운지, 다른
   관측자와 이상하게 같이 움직이는지, 특정 주파수에 에너지가 집중되는지를 계산한다.
2. **M 경로**는 EEG를 보지 않고 직전에 잰 채널별 impedance, block 평균/최댓값,
   wet/dry interface만 본다.
3. 모델은 Q를 주 판단으로 사용한다. M은 처음에는 정확히 0이고, 학습이 필요성을
   발견한 경우에만 작은 공간 보정을 추가한다.
4. 그 보정은 모든 시점에 같은 채널 혼합행렬을 적용한다. 따라서 새로운 시간 주파수를
   만들지 않고, SSVEP의 phase/harmonic 시간 구조를 보존한다.

metadata가 class 정답을 알려주는 힌트가 아니라 **센서 신뢰도에 관한 약한 prior**라는
가설이다.

## 연구가설

- H1 — query EEG의 품질 특징 Q만으로도 실제 채널 오염도를 어느 정도 예측할 수 있다.
- H2 — impedance/interface M은 Q가 이미 아는 것 너머의 채널 품질 정보를 갖는다.
- H3 — Q+M을 입력단 spatial operator로 사용하면 M을 뺀 같은 모델보다 decoding과
  acquisition-shift robustness가 좋아진다.
- H4 — M이 없으면 **같은 A_QM checkpoint 안에서** metadata operator가 정확히 0이 되고,
  stale/wrong M에서도 성능이 붕괴하지 않는가. 이 fallback은 공동학습된 A_QM의 Q branch이지
  별도로 학습한 `A_Q` 모델과 같은 가중치라는 뜻이 아니다. 따라서 각 개입 결과와 `A_Q`의
  참가자별 차이도 함께 기술적으로 보고한다.

H1–H2는 먼저 품질 추정 문제로 검증하고, 통과한 뒤에만 H3 decoding을 본다. 이 순서를
거꾸로 하면 “분류 점수가 우연히 올랐으니 품질을 배웠다”는 순환 해석이 된다.

## 모델이 실제로 받는 정보

Q의 exact 7-feature allowlist는 현재 trial의 다음 값뿐이다.

- 전처리 z-score 이전 채널 표준편차와 availability
- z-score 이후 log variance와 첫 차분 log energy
- 다른 활성 채널과의 mean/max absolute correlation
- label-free spectral concentration

M은 다음 값뿐이다.

- 채널별 log-normalized impedance, block 내 상대값과 mismatch, availability
- 같은 pre-query 측정의 impedance mean/max와 availability
- electrode type(wet/dry)

Label, stimulus frequency/phase, subject/session/trial ID, source file, target participant의 다른
trial, target batch 통계는 두 경로 모두 금지한다. `query_qc_extractor_version`은 저장된
pre-z 표준편차의 출처를, `query_window_reliability_features_v1`은 Q 전체 계산을 각각
기록한다.

## 공간 연산자와 안전 한계

각 trial에 적용되는 변환은 다음과 같다.

\[
X' = (I + \Delta_Q + \alpha\Delta_M)X.
\]

`ΔQ`와 `ΔM`은 각각 symmetric diagonal-plus-low-rank 행렬이다. 직접적인 full-rank 경험
상관행렬은 더하지 않는다. 상관정보는 Q encoder의 채널 특징으로만 들어간다.

- `||ΔQ||F ≤ 0.20`
- `||ΔM||F ≤ 0.05`
- `|α| ≤ 1`, 초기값 `α=0`
- 따라서 `||ΔQ + αΔM||F ≤ 0.25`

Q와 M을 합친 뒤 다시 projection하지 않는다. 그래서 M이 Q를 몰래 재스케일할 수 없고,
실제 적용값은 정확히 `Q + metadata residual`이다. 전체 Frobenius norm이 0.25 이하이므로
`I+Δ`의 eigenvalue도 `[0.75, 1.25]` 밖으로 나갈 수 없다. 비활성/padded 채널은 행과 열이
모두 0이다. M이 모두 없으면 학습 뒤에도 **그 A_QM checkpoint의** `ΔM=0`이라 입력 연산자는
`I+ΔQ`가 된다. 다만 backbone과 Q branch가 M과 공동학습되었으므로 별도 `A_Q` arm과 같은
모델 또는 같은 예측은 아니다.

저장되는 `noise_logit`은 calibrated 신호품질 점수가 아니다. 실제 작용은
`spatial_self_gain`, off-diagonal row L1, Q/M/전체 operator norm으로 별도 기록한다.

## 2×2 실험

새 역할 이름은 기존 역사적 `A0_eeg_only/A2_structured_condition_prompt`와 섞지 않고
항상 `reliability-spatial-v1:A_*`처럼 읽는다.

| 새 역할 | Q 접근 | M 접근 | 묻는 것 |
|---|---:|---:|---|
| A0 | 없음 | 없음 | 같은 graph의 identity operator 기준 |
| A_M | 없음 | 있음 | metadata만으로 품질 보정이 가능한가 |
| A_Q | 있음 | 없음 | 현재 EEG만으로 동적 품질 보정이 가능한가 |
| A_QM | 있음 | 있음 | Q에 metadata를 더했을 때 순증분이 있는가 |

Primary contrast는 `A_QM − A_Q`, secondary는 `A_M − A0`, interaction은
`(A_QM−A_Q)−(A_M−A0)`다. 네 arm은 접근 권한 두 축 외에 graph, parameter 초기값,
split, asset, seed, loader, source와 실행환경이 같아야 한다.

## Stage-0: decoder 결과보다 먼저 하는 검사

`synthetic_quality_v1`은 seed 42, hierarchical SeedSequence RNG, 8명, wet/dry 각각 2 block,
class당 block별 2회, 4 class, 200 Hz×2초, `c_max=64`로 고정한다. 총 256 trial과
`256×8=2,048` active-channel truth row다. Subject×interface×block마다 모든 class를 같은
횟수로 생성하고, 품질 latent와 metadata RNG는 class 수/순서와 분리했다. 채널별 impedance가
independent noise, 50-Hz noise, low-rank 공통 artifact, attenuation과 dropout에 영향을 준다.

생성 때만 알 수 있는 `realized_signal_fraction`은 별도 `quality_truth.jsonl`에 있고 일반
dataset/collate/model 경로는 이 파일을 읽지 않는다. Dropout 전 corruption만 보던 기존
oracle 값과 달리 이 primary target은 실제 생성된 waveform과 clean waveform의 MSE를
포함한다. Asset은 `asset_info.json`, `class_map.json`, JSONL/Parquet manifest,
`preprocess_config.yaml`, `quality_target_spec.json`, `quality_truth.jsonl`, `signals.h5`의
exact 8개 파일만 허용한다. Stage-1 시작 전 gate가 여덟 파일의 byte hash와 Stage-0 산출물을
직접 재검증하고, Stage-1 receipt는 그 Stage-0 receipt와 각 checkpoint의 asset provenance에
결속된다.

합성 asset과 Stage-0 결과 root는 둘 다 **exclusive reservation + fresh-only atomic
publication**이다. 특히 Stage-0는 assay 계산 전에 고정 이름 예약을 원자적으로 선점하고,
첫 결과는 pass든 fail이든 terminal outcome이며 overwrite/supersede 옵션이 없다. 실패 결과를
본 뒤 generator·feature·threshold를 바꾸어 같은 candidate의 새 PASS receipt를 만드는 경로는
금지된다. 기술 실패·크래시로 예약만 남아도 사용자 검토 없이 자동 재시도하지 않는다.

Stage-0는 참가자 LOSO를 사용한다. 각 outer held-out subject마다 남은 subject 안에서 다시
LOSO하여 ridge penalty를 선택하고, train subject에서만 표준화한 Q, M, Q+M probe를 같은
row/split에 맞춰 비교한다. 피험자를 통계 단위로 같은 가중치로 집계한다.

사전 gate는 다음과 같다.

- Q mean held-subject R² `> 0`
- M의 Q 초과 partial R² 평균 `≥ 0.02`, 양의 subject `≥ 6/8`
- query 안 8채널 rank Spearman 개선 평균 `≥ 0.02`, 양의 subject `≥ 6/8`
- 같은 subject/interface 안에서 block metadata bundle을 바꾸거나 채널 impedance를
  회전했을 때 correct pairing의 relative SSE 이득 평균 `≥ 0.01`, 양의 subject `≥ 6/8`
- label-only held-subject macro R² `≤ 0.01`, class별 target mean range `≤ 0.02`

이는 simulator와 wiring이 질문을 식별할 수 있는지 보는 engineering gate다. M과
latent badness의 관계 일부는 generator에 의도적으로 들어 있으므로, 통과해도 human EEG
근거나 논문의 핵심 결과가 아니다. 실패하면 downstream decoding 결과를 실행·해석하지
않는다.

## Stage-0 실제 결과와 현재 판정

2026-09-03 canonical one-shot Stage-0는 terminal `failed`로 공개됐다. 먼저 중요한 구분은
asset·분할·truth 격리·leakage 또는 control 조작이 실패한 것이 아니라, **M이 Q를 넘어
처음 보는 participant에 일관되게 기여해야 한다는 핵심 증분 gate가 실패했다**는 점이다.

| 항목 | 관찰값 | frozen 기준 | 결과 |
|---|---:|---:|---:|
| Q mean held-subject R² | 0.973349 | > 0 | PASS |
| M partial R² beyond Q 평균 | 0.043065 | ≥ 0.02 | PASS |
| partial R² 양수 participant | 5/8 | ≥ 6/8 | **FAIL** |
| Q→QM channel-rank rho 증분 평균 | 0.000651 | ≥ 0.02 | **FAIL** |
| rank 증분 양수 participant | 5/8 | ≥ 6/8 | **FAIL** |
| correct-vs-block-shuffle SSE gain | 0.152518, 8/8 양수 | ≥ 0.01, ≥ 6/8 | PASS |
| correct-vs-channel-permutation SSE gain | 0.127443, 8/8 양수 | ≥ 0.01, ≥ 6/8 | PASS |
| label-only macro R² / class target range | −0.137111 / 0.002847 | ≤ 0.01 / ≤ 0.02 | PASS |

Q-only가 realized signal fraction을 매우 잘 예측했고, Q+M의 평균 R²는 0.974493으로 Q의
0.973349보다 조금 높았다. 하지만 참가자 세 명에서는 partial R²가 음수였고, 채널 순위
개선은 평균 0.000651에 그쳤다. 반면 M block이나 채널 pairing을 깨뜨리면 모든 참가자의
SSE가 나빠졌다. 따라서 M은 generator에 연결되어 있고 probe도 이를 사용하지만, 그 사용이
Q-only보다 안정적으로 더 좋은 held-participant mapping은 아니었다.

가설별로는 H1(Q 식별성)은 simulator에서 지지됐고, H2(M의 안정적 순증분)는 composite
gate를 통과하지 못했다. H3 decoder benefit과 H4 decoder safety는 Stage-1을 시작하지 않아
미검증이다. 이 결과는 human metadata의 보편적 무용성을 뜻하지 않고, 반대로 synthetic
decoding 성능이나 human/OOD 효과도 주장할 수 없다.

Frozen stopping rule에 따라 같은 candidate의 generator·feature·target·threshold를 결과에
맞춰 고치거나 Stage-0를 재실행하지 않는다. Canonical Stage-1 root는 생성하지 않았고,
passing receipt validator도 이 terminal receipt를 거부한다. 다음 유효한 루프는 독립적인
acquisition-factor human development data 또는 결과와 독립된 새 외부 근거에 기반한 별도
candidate/decision contract다. 현 연구목표 자체는 여전히 미결이므로 바꿀 필요가 없다.

증거는 `outputs/engineering/reliability-spatial-v1/stage0/receipt.json`(SHA-256
`e8f3f87bbe47a8c8d5825f7d0525c318e87d42c6ecdd2840de21b1c657eefc35`)과 같은 디렉터리의
`contrasts.csv`, `subject_metrics.csv`, `predictions.csv`, `fold_models.json`에 있다.

## 동결된 실행 순서와 완료 상태

아래 recipe의 1–2단계는 완료됐고 Stage-0가 실패했다. 3단계의 비정식 contract dry-run은
outcome 전에 완료됐지만, 4단계 실제 Stage-1은 실행하지 않았으며 이제 실행해서는 안 된다.

~~~bash
# 1. 별도 합성 asset 생성
python scripts/prepare_synthetic.py --quality-aware \
  --out_dir data/processed/synthetic_quality_v1 \
  --n_subjects 8 --n_blocks_per_interface 2 \
  --n_repetitions_per_class_per_block 2 --n_classes 4 \
  --target_sfreq 200 --duration_sec 2 --c_max 64 --seed 42

# 2. participant-disjoint Stage-0
python scripts/run_synthetic_reliability_stage0.py

# 3. 선택적 outcome-free contract/forward dry-run (canonical Stage-1 root 사용 금지)
python scripts/run_ablation.py \
  --config configs/train/synthetic_reliability_2x2.yaml \
  --output-root outputs/dry-runs/reliability-spatial-v1/stage1-contract-v1 \
  --dry-run

# 4. Stage-0 receipt가 pass이고 같은 implementation/asset일 때만 4-arm CUDA 실행
python scripts/run_ablation.py \
  --config configs/train/synthetic_reliability_2x2.yaml \
  --output-root outputs/engineering/reliability-spatial-v1/stage1
~~~

Stage-1 runner는 passing Stage-0 receipt, asset bytes, implementation-contract hash가 맞지
않으면 학습 전에 중단한다. Dry-run은 receipt 없이 graph/contract를 검사할 수 있지만 fresh한
비정식 root만 쓸 수 있고 canonical Stage-1 root에는 한 byte도 쓰지 않는다. 실제
네 arm과 분석은 mode-0700 hidden staging root에서 완성·검증·fsync한 뒤 suite 전체를 한 번에
canonical root로 rename한다. 공개 직전 Stage-0와 여덟 asset, 네 checkpoint, 네 split,
네 validation metric, 네 runtime sidecar와 네 nested attempt를 다시 해시하고, checkpoint의
best epoch/accuracy가 metric history의 실제 첫 최댓값인지 확인한다. 또한 runtime row에서
초기에 결속한 source revision과 같은지도 확인한다. 실패 attempt는 canonical로 공개하지 않고
plan·source·부분 artifact hash가 든 영수증과 함께 보존하며, 새 사용자 결정 없이는 재시도를
거부한다. 고정 이름의 atomic execution reservation을 네 arm 중 어느 것도 시작하기 전에
선점하므로 동시 프로세스 두 개가 각각 outcome을 만드는 것도 막고, 완전한 공개 뒤에만
예약을 해제한다. `summary.csv` 역시 네 completed role, runtime-family 상태, canonical 경로와
analysis receipt hash를 재검증해야 공개된다. 저장장치 오류처럼 failure receipt 자체도 쓸 수
없는 crash에서는 고정 reservation을 그대로 남기는 것이 최소 fail-closed 기록이며, 역시
owner review 없이는 재시도하지 않는다.

Stage-1 수치는 checkpoint를 고른 동일한 두 held-out synthetic subject에서 나온 selection-set
기술통계다. 참가자별 `A_QM−A_Q`, `A_M−A0`, interaction 외에 같은 A_QM checkpoint의
correct/missing/stale/wrong M, 그리고 각 scenario와 별도 A_Q의 차이를 저장한다. 따라서 이
단계에는 효과크기 통과 기준이나 모집단 추론이 없다.

## 무엇을 주장할 수 있고 없는가

합성 Stage-0/1이 통과하면 “정의한 simulator에서 Q와 M 정보 경로 및 잔차 operator가
작동했다”까지만 말할 수 있다. Human EEG, population benefit, architecture superiority,
confirmatory 진입은 주장할 수 없다.

다음 승격에는 S1–S3 재사용이 아니라 독립적인 human development data가 필요하다. 가장
가치 있는 자료는 참가자 수만 많은 일반 EEG가 아니라, 같은 participant/task에서
interface·contact pressure·impedance level·시간 drift를 반복 또는 무작위화하고 amplifier,
humidity와 motion을 기록한 acquisition-factor 자료다. 그 자료에서 Stage-0 increment,
correct-versus-shuffled reliance, missing/wrong metadata safety를 새 기준으로 사전등록한 뒤
통과해야만 untouched cohort 사용을 별도 승인할 수 있다.

## 설계 근거가 된 인접 연구

- [Dynamic Spatial Filtering](https://doi.org/10.1016/j.neuroimage.2022.118994)은 현재 EEG의
  variance/covariance로 입력단 spatial matrix를 동적으로 만드는 근거다.
- [Reliable Components Analysis for SSVEP](https://pmc.ncbi.nlm.nih.gov/articles/PMC6583904/)
  는 SSVEP에서 단순 variance보다 trial reliability/covariance가 중요한 이유를 보여준다.
- [Electrode impedance and data quality](https://pmc.ncbi.nlm.nih.gov/articles/PMC2902592/)
  는 절대 impedance의 효과가 주파수·환경·시스템에 의존하므로 단조 품질점수로
  hard-code하면 안 된다는 근거다.
- [Wearable SSVEP dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC7916479/)은 wet/dry paired
  signal과 block impedance를 제공하지만 interface bundle과 impedance의 독립 인과효과를
  분리하지 못한다.
- [Missing-modality robustness](https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Are_Multimodal_Transformers_Robust_to_Missing_Modality_CVPR_2022_paper.html)
  는 auxiliary metadata가 틀리거나 빠질 때 unimodal 기준보다 나빠질 수 있어 exact
  fallback과 corruption control이 필요함을 보여준다.
- [DomainBed](https://openreview.net/forum?id=lQdXeXDoWtI)은 OOD 성능 주장에 architecture뿐
  아니라 동일한 model-selection protocol이 포함되어야 한다는 근거다.

`arXiv:2608.11829`의 learned/retained/forgotten은 나중에 성공 threshold를 넘는
subject-condition cell의 capability redistribution을 설명하는 보조 틀로만 쓴다. 그것은
quality operator의 근거나 Stage-0를 대체하지 않으며, 현재 합성 단계에서는 사용하지 않는다.

# Query-reliability 후속 연구설계

기준일: 2026-09-04

후보: `query-reliability-spatial-v1`
상태: **사람 outcome 전 primary 철회 / frozen baseline-only / 기존 outcome plan 실행 차단**

> 2026-09-04 `DEC-20260904-005`는 상위 목표를 calibration-efficient SSVEP로 복구하고,
> pre-query acquisition context의 저보정 곡선 순증분을 새 primary로 삼았다. 이 문서는
> 음성 결과가 아니라 **실행되지 않은 query-only 설계의 역사적·baseline 계약**으로
> 보존한다. BETA 35/20 training·prediction을 실행하지 않는다. 현재 설계는
> [metadata-assisted low-calibration 설계](metadata_calibration_efficiency_design.md)를 따른다.

## 당시 무엇을 연구하려 했나

BETA의 3초 자극 적격 모집단 S16–S70에서 처음 보는 참가자의 2초 SSVEP 한 개만 받았을 때,
그 신호 자체에서 “어느 채널을 얼마나 믿을지”를 정하는 작은 공간 연산자가 **같은 손상
augmentation으로 학습한** identity 모델보다 미리 정한 합성 채널 손상 아래에서 더 잘
분류하면서, 두 clean 비교에서 평균 성능을 1%p 이상 잃지 않는지 검증한다. 표준 clean-trained
identity보다 손상 성능이 낫다는 별도 비교는 현재 필수 보고이지만 아직 승격 gate는 아니다.

이 문서를 동결했을 당시에는 상위 응용목표를 **unseen-participant, strict k=0 SSVEP
decoding**으로 좁히고, 외부 metadata가 없는 query-only 후속 가설을 제안했다. 그러나 사람
outcome 실행 전에 `DEC-20260904-005`가 이 방향을 supersede했다. 현재 목표는 k=0을 anchor로
포함하는 low-calibration curve이며, acquisition context의 순증분을 다시 묻는다. 합성 Stage-0
실패는 그때의 동결 DGP·feature·target에만 적용되며 현실의 모든 metadata 가설을 종료하지
않는다.

## 가장 쉬운 비유

64개 EEG 채널을 64명의 관측자라고 생각한다. Q0는 모든 관측자의 말을 원래 방식대로
backbone에 넘긴다. Q1은 현재 2초 발언만 보고 서로 잘 맞는 관측자는 조금 연결하고, 불안정해
보이는 관측자는 조금 줄인다. 이 조정은 identity에서 시작하고 Frobenius norm `0.20` 안으로
제한되므로 입력을 마음대로 재작성할 수 없다.

Q1이 보는 일곱 값은 채널마다 다음과 같다.

- 저장된 6–90 Hz 전처리 단계의 pre-zscore scale QC와 그 availability 1개
- 현재 waveform의 6–60 Hz log variance, first-difference energy, 평균/최대 채널상관,
  최대 spectral-bin concentration 5개

즉 5개 waveform feature + scale 1개 + availability 1개다. Subject ID, dataset ID,
hardware ID, impedance, 정답 label, 정답 frequency/phase template, 다른 target trial이나 target
batch 통계는 배포 예측에 들어가지 않는다. Source label은 source-trained closed-set classifier를
학습하는 데 사용되므로 “label-free 학습”을 주장하지 않는다.

## 가설과 2×2 실험

핵심 가설은 다음처럼 좁다.

> Source에서 학습된 query-conditioned bounded operator가, 동일 source data·backbone과 동일
> corruption augmentation을 쓰는 exact identity arm보다 BETA의 3초 자극 적격 모집단
> S16–S70의 새로운 참가자에서 사전등록된 equal-weight 합성 손상 composite의 balanced
> accuracy를 높이는가?

두 축은 query operator 사용 여부와 손상 augmentation 학습 여부다.

| 역할 | query operator | 손상 augmentation | 목적 |
|---|---|---|---|
| Q0_CLEAN | exact identity | 없음 | 표준 배포 기준 |
| Q1_CLEAN | observed Q | 없음 | augmentation 없이 Q가 주는 효과 |
| Q0_AUG | exact identity | 있음 | augmentation 자체의 효과 |
| Q1_AUG | observed Q | 있음 | 제안하는 joint method |

Primary는 `Q1_AUG − Q0_AUG`다. 이는 augmentation을 동일하게 두고 정확한 Q-bundle-conditioned
system 전체의 순증분을 묻는다. 그러나 Q 정보, Q-dependent operator, 실제로 작동하는 함수
용량을 한꺼번에 바꾸므로 “신뢰도 feature 자체의 인과효과”를 곧바로 증명하지 않는다.

필수 기술 보고에는 다음도 포함한다.

- 손상 composite의 `Q1_AUG − Q0_CLEAN`: 제안 배포 조합이 표준 clean-trained 기준보다 실제로
  나은지 보여준다. 현재는 승격 gate가 아니라 필수 보고다.
- clean·corrupted 각각의 2×2 query×augmentation interaction
- participant×corruption cell, 역할별 최악 cell, participant composite의 10/25/50% 분위수
- class×corruption cell 및 50 Hz가 stimulus harmonic과 겹치는 class flag

따라서 gate가 모두 통과해도 주장할 수 있는 것은 **동일 augmentation의 identity 대비,
미리 고정한 여섯 synthetic-stress equal-weight score의 평균 개선**뿐이다. 별도의 corrupted
`Q1_AUG−Q0_CLEAN` utility gate를 outcome 전에 동결하지 않는 한 표준 clean-trained identity보다
우월하다고 말할 수 없다. 모든 손상, 새로운 손상 종류, 실제 motion/contact, device/site OOD에서
보편적으로 강건하다고도 말할 수 없다.

## 데이터와 분할

- BETA S16–S70만 2초 window에 시간적으로 적합하다. S1–S15는 2초 stimulation인데 현재
  0.63–2.63초 crop이 offset을 약 0.13초 넘으므로 outcome 전에 제외했다.
- 이미 개발 과정에서 노출된 S16은 training-only로 강제했다.
- 나머지는 고정 salt SHA-256 순위로 35명 training, 20명 lockbox로 나눴다.
- Training은 5,600 trial, lockbox는 3,200 trial이며 validation set은 없다.
- Seed `[11, 29, 47]`, 정확히 10 epoch, final checkpoint를 사용한다. Target outcome으로
  early stopping이나 hyperparameter 선택을 하지 않는다.

Runtime split SHA-256은
`d0afab8d17be77f80b3666722dc71036569b93b9370af6bbfeab9e0e7f14570c`다. BETA 여섯
자산의 bundle SHA-256은
`43ecb4936ec3abaf2f92e602142a94e93a7c3e8cb459cf3d11ad6358d379cb5c`다.

## 어떤 손상을 시험하나

Train/evaluation 모두 같은 여섯 cell을 쓴다.

- 12.5%, 25% channel zeroing
- 6–60 Hz white noise SNR 0 dB, −5 dB
- 50 Hz line noise SNR 0 dB, −5 dB

Evaluation은 cell마다 고정 draw 3개를 쓰고 participant 안에서 여섯 cell을 같은 가중치로
평균한다. 이는 **같은 BETA corpus/device/site 안의 held-participant + 이미 본 합성 손상**
실험이다. Corruption-OOD나 device/site/domain-OOD 실험이 아니다.

## 판정 규칙과 현재 통계 병목

참가자 20명의 paired balanced-accuracy difference가 통계 단위다. Seed는 독립 참가자로 세지
않고, 각 sample에서 세 seed 확률을 먼저 평균한 다음 participant BA를 만든다.

현재 수치는 아직 잠정이다.

- Primary: one-sided participant mean t lower bound가 0보다 크고, 관찰 평균이 `+0.02`
  이상이며, 20명 중 적어도 12명이 양수여야 한다.
- Clean safety: `Q1_AUG−Q0_AUG`와 `Q1_AUG−Q0_CLEAN` 각각 평균 비열등성 margin `−0.01`.
- 세 inferential test는 하나의 conjunctive IUT로 모두 통과해야 한다.

`+0.02`는 “모집단 효과가 2%p보다 큼”을 검정하는 null이 아니라 운영상 승격선이다. 12/20도
sign test가 아닌 이질성 화면이다. `−0.01` 실패는 harm 증거가 아니라 clean safety를 확립하지
못했다는 뜻이다.

Outcome-free 계산에서 N=20, true clean delta 0, paired SD 0.04라면 clean 비열등성 하나의
power는 약 0.286뿐이다. 80% power에는 SD 약 0.0173 이하 또는 SD 0.04에서 약 101명이
필요하다. 그래서 현재 20명 설계는 엄격한 safety screen 대신 높은 비승격 위험을 감수하는
설계가 될 수 있다. 자세한 표는 [power/sensitivity 감사](query_reliability_power_sensitivity.md)에
있다.

## “정말 현재 query의 품질을 쓴 것인가”를 어떻게 확인하나

Primary 결과만으로 품질 메커니즘을 주장하지 않는다. 같은 Q1_AUG checkpoint에 두 분석
전용 개입을 추가한다.

1. Identity intervention: 학습된 operator를 정확히 identity로 끈다.
2. Wrong-query intervention: 같은 participant·같은 class의 다른 trial Q로 operator만 만들고,
   backbone에는 원래 target waveform을 넣는다. 3,200개 전체의 nonself 일대일 mapping은
   SHA-256 `e1d7be88b55a1ad35d975787b95183ff65909507eace648109e069164344027f`에 고정했다.

Wrong-query는 다른 target trial과 label을 offline pairing에 쓰므로 strict k=0 배포 결과가
아니다. 표준 prediction 뒤 같은 숨김 atomic transaction 안에서만 생성하고 primary, clean,
deployment estimate에는 절대 섞지 않는다. 두 intervention, 손상 채널 localization, Q-feature
label probe, leave-one-feature-family-out ablation, 50 Hz/class 집중 여부가 모두 만족되어야
bad-channel recovery 또는 reliability mechanism을 주장한다. 실패해도 primary
Q-bundle-conditioned system utility가 자동으로 뒤집히는 것은 아니며 메커니즘 주장만 막는다.

## 인접 연구와 신규성 경계

- Dynamic Spatial Filtering은 현재 window의 통계로 sample-specific spatial operator를 만드는
  일반 아이디어를 이미 보였다. 현재 연구는 이 아이디어의 발명 주장을 하지 않는다.
- MAPS-CS는 SSVEP에서 signal quality 기반 동적 channel selection을 이미 보고했다.
- Fast SSVEP는 calibration-free query-local alignment/spectral denoising에 가장 가까운 learned
  comparator다.
- SSVEPPoolformer는 adaptive denoising과 cross-channel pooling을 결합했다.
- TMW-CCA와 EMD-QC도 quality-aware SSVEP의 “최초” 주장을 막는다.

따라서 조건부 기여는 “source-trained query-only bounded operator를 엄격한 정보권한,
participant-disjoint split, clean 평균 비열등성, 합성 손상 localization, shortcut/mechanism 감사,
독립적인 within-Dong 재학습 replication으로 평가한 것”이다. Full text와 공통 protocol port가
끝나기 전에는 novelty clearance가 완료되지 않는다.

## 현재 완료된 것과 시작하지 않은 것

완료:

- query-only bounded operator와 exact Q0 identity graph
- 2×2 family, 고정 corruption, BETA partition·asset·sample identity 계약
- generic train/eval/baseline의 BETA lockbox 선차단
- 결과 bundle의 complete-grid, provenance, input/mask hash, seed/init/checkpoint 검증
- same-checkpoint identity/wrong-query model hook
- 결과를 쓰지 않은 RTX CUDA Q0/Q1 한-batch forward 검증
- outcome-free N=20 analytic/Monte Carlo sensitivity 코드

아직 시작하지 않음:

- 사람 EEG training, lockbox prediction, accuracy/loss 해석
- identity/wrong-query/feature ablation을 수집하는 전용 atomic runner
- MAPS-CS·SSVEPPoolformer·TMW-CCA·EMD-QC full-text 기반 common-protocol baseline
- numeric utility margin 최종 승인, clean source tag, 외부 owner receipt
- BETA reveal 전에 exact recipe·estimand·threshold·stopping/atomic contract를 먼저 동결하고,
  BETA 통과 뒤 실행만 별도 승인할 within-Dong replication

현재 파일은 active 실행계획이 아니라 철회된 fail-closed 구현·baseline 계약이다. 위 blocker를
닫는 것만으로 기존 BETA outcome plan이 되살아나지 않으며, 별도의 새 결정 없이는 human
outcome 실행을 계속 거부한다.

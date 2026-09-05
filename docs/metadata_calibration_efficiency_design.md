# Metadata-assisted calibration-efficient SSVEP 연구설계

기준일: 2026-09-05

후보: metadata-calibration-efficiency-v1
상태: **방향 승인 / 수식·통계·private-staging validator·dispatch·phase별 execution contract 구현 / 실제 manifest·producer receipt·full human runner·owner freeze 전 outcome 실행 차단**

## 결론부터

연구목표는 metadata가 유용하다는 것을 증명하는 일이 아니다.

> 처음 보는 사용자가 쓸 만한 SSVEP 성능에 도달하기 위해 제공해야 하는
> labeled calibration EEG의 양을 줄이는 것이 목표다.

Metadata는 이 목표를 위한 하나의 수단이다. 여기서도 막연한 개인정보나 dataset ID를
뜻하지 않는다. 현재 직접 시험할 수 있는 것은 **EEG block 전에 이미 알 수 있는 획득
상황**, 즉 wet/dry 전극 interface와 채널별 impedance다. 따라서 연구의 정확한 이름은
“pre-query acquisition-context-assisted low-calibration SSVEP”이다.

이 복구는 기존 실패 결과를 지우지 않는다. physical_hybrid_v1은 그대로 retired no-go이고,
합성 reliability-spatial-v1 Stage-0도 terminal failure다. Query-only 후속안은 사람 outcome을
전혀 열지 않은 상태에서 primary 자리에서 내려와 frozen baseline 후보가 된다.

## 가장 쉽게 설명하면

새 사용자가 12개의 SSVEP 명령을 쓰려고 한다고 하자.

- k=0: 이 사람의 라벨 EEG를 하나도 받지 않고 바로 시작한다.
- k=1: 한 deployment interface에서 12개 명령을 한 번씩 본 완전한 block 하나, 총 12
  trial을 보정에 쓴다.
- k=3: interface당 세 block, 총 36 trial을 쓴다.
- k=5: interface당 다섯 block, 총 60 trial을 쓴다.

연구에서는 wet과 dry를 모두 별도로 평가하므로 participant 전체에서 실제로 읽는 labeled
support는 각각 24/72/120 trial이다. k=1의 12 labeled-trial 자원량은 한 사용자가 실제 배포에서
택한 interface 하나당 수치다. 현재 confirmatory 절약 주장은 k=3에서 k=1로 줄이는
것이므로, 통과 시 interface당 24 trial 또는 두 complete block 절약이다.
다만 통계 검정은 wet과 dry를 참가자 안에서 동일 가중 평균한 estimand이므로, 이는 선택한
interface에서 적용되는 **보정 schedule의 양**이 24 trial 줄었다는 뜻이다. Wet과 dry 각각에서
별도로 비열등함을 증명한 것은 아니며 condition별 효과는 descriptive다.

보통은 EEG만 보고 모델을 보정한다. 새 방법은 “현재 wet/dry 중 무엇을 쓰는가”, “각 채널의
block 전 impedance가 어떠한가”를 추가로 이용해 데이터가 적을 때의 측정 잡음 또는 채널
신뢰도에 더 나은 prior를 준다. 보정 trial이 늘면 실제 사용자 EEG가 prior를 점차 대체한다.

핵심 질문은 단순하다.

> 같은 k를 주었을 때 acquisition context를 아는 모델이 모르는 모델보다 더 좋은가?

그리고 실제 제품 관점의 후속 질문은 다음이다.

> 같은 성능을 얻는 데 필요한 완전한 calibration block을 줄였는가?

두 질문은 다르다. 첫 질문을 먼저 검정하고, 두 번째는 “추가 두 block이 원래 유용했는가”와
미리 정한 비열등성 margin을 모두 통과할 때만 “두 block 절약”으로 해석한다.

## 연구목표와 가설

### 상위 목표

처음 보는 참가자와 사전 지정된 획득조건에서 closed-set SSVEP decoding의 labeled
target-calibration 부담을 최소화한다. k=0은 calibration-free anchor이고, k>0은
low-calibration regime이다. k>0 결과를 calibration-free라고 부르지 않는다.

### Primary 가설

각 참가자에서 k=0, 1, 3의 고정 query 성능으로 만든 early-budget AUC를 비교한다.

eAUC = Y(0)/6 + Y(1)/2 + Y(3)/3

여기서 Y(k)는 wet과 dry를 각각 평가한 뒤 동일 가중으로 평균한 participant balanced
accuracy다. Primary contrast는 다음 하나다.

> eAUC(A_QM) − eAUC(A_Q) > 0

- A_Q: EEG, 채널 구조, 동일한 signal-derived QC만 사용
- A_QM: A_Q와 동일하지만 block 전에 관측한 interface와 impedance를 추가 사용

Participant가 독립 추론 단위다. Seed, class, block, support draw를 표본 수로 세지 않는다.
H1의 모집단 귀무가설은 평균 차이 `≤0`이다. 관측 평균 `≥0.020`은 실용성 screen이지
`모집단 평균≥0.020`을 검정한 것은 아니다. 따라서 H1만 통과하면 “external context의 양의
순증분”까지만 말하고, “쓸 만한 보정 절약”은 아래 H2까지 통과했을 때만 말한다.

### 주요 secondary

- k=0의 A_QM−A_Q: calibration-free anchor
- k=1과 k=3의 A_QM−A_Q: 가장 중요한 low-calibration 자원점
- k=5: metadata 이득이 충분한 보정 뒤 사라지는지 보는 saturation point
- dry와 wet 각각의 효과
- 참가자 하위 10/25%와 metadata로 악화된 참가자 비율
- model×budget interaction

### Calibration 절약 주장

Confirmatory core는 두 단계, 세 논리조건이다.

1. H1: eAUC(A_QM)−eAUC(A_Q)가 통계적으로 0보다 크고 관측 평균도 0.020 이상이다.
2. H2a: A_Q(k=3)−A_Q(k=1)가 통계적으로 0보다 크고 관측 평균도 0.020 이상이다.
3. H2b: A_QM(k=1)−A_Q(k=3)의 one-sided 97.5% lower bound가 −1/60보다 크고 두 endpoint
   평균 BA가 모두 0.50 이상이다.

H1이 먼저 통과한 뒤 H2a와 H2b가 모두 통과할 때만 “M-assisted k=1이 Q-only k=3을
대체했다”고 말한다. H2a는 Q-only curve가 원래 평평한데 비열등성만 쉽게 통과하는 허점을
막는다. 이 경우 equal-weight wet/dry estimand에서 interface별 보정 schedule을 36→12 labeled
trials로 줄였다고 말한다. 즉 선택한 deployment interface에서는 24 trials/two blocks가 줄고,
wet과 dry 연구 평가 전체 support 접근은 48 trials 줄어든다. Condition별 비열등성은 별도로
주장하지 않는다.
A_QM(k=0) 대 A_Q(k=1), A_QM(k=3) 대 A_Q(k=5), 사후 curve 보간은 descriptive secondary다.
자극 시간은 cue, gaze shift, rest, feedback을 재지 않고 총 소요시간으로 바꾸어 쓰지 않는다.

## 새 방법은 이전 A2 재실행이 아니다

이전 physical_hybrid_v1은 metadata를 decoder의 global FiLM과 채널 gain에 직접 넣었다.
유효한 S1–S3 mechanism assay에서 clean 이득과 correct-pairing reliance를 보이지 못했으므로
그 방법을 이름만 바꾸어 다시 실행하지 않는다.

새 후보는 metadata를 **소량 보정 estimator의 bounded predictive-precision residual**로
제한한다.

1. Source participant에서 spectral backbone과 Q-only target-calibration estimator를 먼저
   학습하고 **고정한다**.
2. 그 공통 경로에는 더 이상 gradient나 M-derived statistic이 들어가지 않게 한 뒤,
   interface와 block 전 impedance가 calibration prior의 precision/shrinkage에 주는 작은
   bounded residual만 별도 source episode에서 학습한다.
3. 이 residual은 대각 Gaussian predictive precision의 각 축을 최대 0.8–1.25배만 바꾼다.
   Query waveform을 직접 변환하거나 채널을 섞지 않는다.
4. k>0에서는 target support의 sufficient statistics 또는 작은 estimator만 갱신한다. Backbone
   전체를 participant별로 fine-tune하지 않는다.
5. M을 끄거나 모두 missing으로 만들면 metadata residual이 정확히 0이 되어, 같은 composite
   source checkpoint의 frozen A_Q 경로와 byte-level로 같은 계산을 해야 한다.

Frozen spectral embedding을 (z), source class anchor를 \(\mu_c^S\), signal-derived Q가 만든
대각 precision을 \(\lambda_Q\)라 한다. M residual은 다음처럼 정의한다.

\[
\delta_M(m)=a(m)\log(1.25)\tanh g_M(m),\qquad
\lambda_{QM}=\operatorname{clip}\{\lambda_Q\odot\exp(\delta_M),0.05,20\}
\]

여기서 metadata가 전부 missing이면 \(a(m)=0\)이고, 구현은 재계산하지 않고 원래
\(\lambda_Q\) tensor를 그대로 반환한다. 따라서 all-missing A_QM과 A_Q가 bitwise 같아야 한다.
Q residual 자체는 base precision의 0.25–4배, M residual은 Q precision의 0.8–1.25배로
제한된다.

Class \(c\)의 support를 \(S_{c,k}\)라 하면 posterior는 대각 sufficient statistic으로만
갱신한다.

\[
P_{c,k}=P_0+\sum_{i\in S_{c,k}}\operatorname{diag}(\lambda_i),\qquad
\hat\mu_{c,k}=P_{c,k}^{-1}\left(P_0\mu_c^S+
\sum_{i\in S_{c,k}}\operatorname{diag}(\lambda_i)z_i\right)
\]

Query \(j\)의 predictive variance와 score는 다음과 같다.

\[
v_{cjk}=\lambda_j^{-1}+P_{c,k}^{-1},\qquad
s_{cjk}=-\frac12\sum_d\left[
\frac{(z_{jd}-\hat\mu_{c,kd})^2}{v_{cjkd}}+\log v_{cjkd}\right]
\]

이 정의 덕분에 k=0에서도 source anchor는 움직이지 않지만 M이 query predictive metric을
작게 바꿀 수 있다. M을 support shrinkage에만 넣으면 k=0에서 A_QM=A_Q가 되어 첫
calibration-saving 비교를 metadata 효과로 해석할 수 없으므로 그 안은 버렸다.

즉 metadata는 정답을 알려 주는 prompt가 아니라 현재 embedding 축별 관측 불확실도를
조정하는 작은 source-learned residual이다. 이는 과거 `X'=(I+ΔQ+αΔM)X` query-spatial
operator 재실행이 아니다. 수식, Q/M builder, composite backbone 연결, label-free query
collator, role/context별 불변 execution dispatch와 full-checkpoint 검증, signed lifecycle이
구현됐고 owner가 exact source→conditional-held 실행을 위임 승인했다.

고정 입력 schema는 다음과 같다.

- Q: `[POz, PO3, PO4, PO5, PO6, Oz, O1, O2]` 각 채널의 7개 waveform/QC 특징,
  총 56차원
- M: dry/wet one-hot 2개, 같은 순서의 normalized impedance 8개, impedance availability
  8개, 총 18차원

Impedance는 collate 단계에서 이미 `log1p(kOhm)/log(101)`로 정규화되므로 두 번 정규화하지
않는다. Q와 M builder는 별도 함수다. Low-level A_Q off-path는 M builder를 호출하지 않고,
A_M의 logits에는 signal-derived Q가 전달되지 않는다. Production은 한 isolated job에서 모든
factorial cell을 완성하기 위해 observed Q를 episode당 한 번 계산해 공유하지만, Q-off cell에는
동일 shape의 exact zero만 전달한다. `electrode_type`의 generic numeric ID는 source별 vocabulary에서
wet/dry 번호가 바뀔 수 있으므로 사용하지 않고 governed donor의 문자열만 fixed one-hot으로
바꾼다. Backbone에는 전체 condition dict가 아니라 `channel_ids`와 `channel_mask`만 새 dict로
전달한다.

향후 deployment 단계에서는 metadata로 필요한 k를 0/1/3/5 중 선택하는 calibration-allocation
policy도 만들 수 있다. 다만 이는 성능곡선과 utility threshold가 필요한 별도 정책 문제다.
현재 primary는 먼저 metadata가 곡선 자체를 개선하는지 검정한다.

## 공정한 2×2 정보권한

모든 역할은 같은 source data, frozen backbone·Q estimator, parameter graph, task codebook,
support, query와 adaptation schedule을 사용한다. A_Q와 A_QM을 별도로 end-to-end 학습하지
않는다. 먼저 공통 A_Q를 고정한 뒤 M residual만 추가로 fit하므로 A_Q weights가 M 학습의
영향을 받지 않는다.

| 역할 | EEG backbone | Signal/QC Q | External context M | 목적 |
|---|---:|---:|---:|---|
| A_0 | 있음 | 없음 | 없음 | 최소 EEG·구조 기준 |
| A_M | 있음 | 없음 | 있음 | EEG backbone+M, Q 제거 attribution |
| A_Q | 있음 | 있음 | residual exact-off | 강한 signal-derived 기준 |
| A_QM | 있음 | 있음 | 있음 | 제안 방법 |

Primary는 같은 composite checkpoint에서 residual을 on/off한 A_QM−A_Q system contrast다.
A_Q가 이미 알고 있는 realized signal quality를 넘어 M이 주는 순증분만 묻기 때문이다.
All-missing, shuffled, stale는 같은 A_QM checkpoint의 mechanism intervention이며 별도로
학습한 A_Q checkpoint와 비교하는 것이 아니다. A_0와 A_M은 attribution용 secondary다.
Held atomic core에는 A_0/exact-off, A_M/correct, A_Q/exact-off, 그리고 A_QM의 correct,
all-missing, block-shuffle, stale, opposite-interface까지 총 8개 cell을 모두 넣는다. 이 이름은
runner의 임의 문자열 분기가 아니라 하나의 동결 dispatch table에서 Q on/off, M on/off/missing,
feature ablation과 intervention으로 변환된다. Source-development 전용 진단 cell은 held dispatch가
명시적으로 거부한다.
각 cell은 block donor key에서 끝나지 않는다. Label-free resolver가 sealed block-context view를
실제 interface 값과 정규화된 8채널 impedance·missing-mask로 바꾸고, 최종 float32 값과 cell
dispatch·mapping·phase·opaque row token·실제 EEG base-input hash를 함께
`resolved_context_values_sha256`에 묶는다. 전체 job에서는 예상 token을 정확히 한 번씩 덮는
batch-order-independent context-usage ledger를 다시 계산한다. Mapping은 content hash만 보지 않고
correct/shuffle/stale/opposite donor 규칙과 shuffle seed를 행별로 재계산한다. Block context의
impedance 순서는 canonical ID `[56,55,57,54,58,62,61,63]`으로 명시한다.
`opposite_interface`는 donor impedance뿐 아니라 interface one-hot도 함께 교체하고, stale
block01처럼 row-level all-missing인 경우 availability를 포함한 18개 M feature 전체가
value=0/missing=true가 된다. Producer receipt와 private prediction staging은 dispatch hash,
mapping hash와 resolved context-usage hash를 모두 결속해야 한다.
Support-M-only, query-M-only, interface-only와 impedance-only는 39명 development 진단 cell이며
held core grid를 사후 팽창시키지 않는다. Metadata-only shortcut은 별도 예측 cell이 아니라,
block-constant M·block당 각 class 한 번·row ID/order 부재가 정확히 BA=1/12를 함의하는 구조적
불변식이다.

Metadata로 세지 않는 정보:

- 채널 이름/identity와 mask
- sampling rate와 time grid
- 현재 query 또는 support EEG에서 계산한 pre-zscore scale, variance, spectrum, covariance
- 모든 역할이 똑같이 쓰는 12-class frequency/phase task vocabulary

External treatment로 인정하는 정보:

- wet/dry interface
- 해당 block 전에 측정한 채널별 impedance와 availability

모델 입력에서 금지하는 정보:

- dataset, subject, session, file, trial ID
- target query label 또는 성능
- post-hoc artifact rejection 결과
- label/order proxy
- target query batch 전체의 covariance나 normalization 통계

Headband order, first/second condition period와 block number는 암호화된 finalizer-only
sidecar에 보존하되, 이번 v1에서는 descriptive balance와 설계 불변식 감사에만 쓴다. 참가자별
wet/dry 동일가중 estimand와 고정 query block 6–10 때문에 이 세 값은 primary contrast 안에서
구조적으로 균형을 이루며, 별도의 사후 조정모형이나 confirmatory claim adjustment에는 쓰지
않는다. Decoder 입력에는 넣지 않는다.

## Support와 query를 어떻게 나누나

Wearable은 각 interface에 10개 block이 있고, 각 block에 12개 class가 한 번씩 있다.
실제 onboarding 순서를 흉내 내는 primary partition은 다음과 같다.

| Budget | Support | 고정 query |
|---|---|---|
| k=0 | 없음 | block 6–10 |
| k=1 | block 1 | block 6–10 |
| k=3 | block 1–3 | block 6–10 |
| k=5 | block 1–5 | block 6–10 |

Support는 S0 ⊂ S1 ⊂ S3 ⊂ S5이고 query와 겹치지 않는다. Wet과 dry를 별도로 보정하고
평가한 뒤 participant 안에서 두 조건을 동일 가중 평균한다.

Generic calibration 코드는 label별 sample을 따로 SHA 정렬하므로 k=1이 서로 다른
12개 block의 혼합이 될 수 있다. 이는 class-balanced oracle sample이지 실제 한 calibration
block이 아니다. 따라서 이 경로는 primary에서 금지했다. 전용 계약과 sample-level support
seal은 complete block을 재계산해 검증하며, 이를 실제 학습·추론 producer와 연결하는 full
runner만 아직 남아 있다.

또한 각 role×k×condition은 동일한 frozen composite source checkpoint에서 독립적으로 시작한다.
k=1 모델을 이어서 k=3으로 만드는 방식은 쓰지 않는다. 모든 k의 query는 byte-level sample
identity까지 같아야 한다.

## 데이터가 무엇을 식별하는가

### Wearable v3

102명 모두가 같은 12-target task를 wet과 dry에서 각각 10 block 수행했다. Block마다
8채널 impedance가 자극 전에 측정됐다. 로컬 manifest 감사 결과는 24,480행,
204 participant×interface session, 2,040 participant×interface×block이다. 각 block의
metadata는 12 class 전체에 공유되므로 간단한 label lookup은 아니다.

따라서 다음의 좁은 예측효용 주장은 시험할 수 있다.

> 같은 NeuSen W 장비, 같은 8채널 posterior montage와 이미 관측된 wet/dry interface에서,
> pre-query acquisition context가 unseen participant의 calibration 효율을 높이는가?

이는 randomized interface나 impedance manipulation의 독립 인과효과가 아니다. Device, site,
reference, montage 전체로 일반화한다는 주장도 할 수 없다. Wet/dry
사이에 시간, 재착용, 세척, 피로와 carryover가 섞여 있고, impedance도 interface와 강하게
연관된다. Continuous impedance의 별도 효과는 participant×interface 내부 block 변동으로
분리해야 한다.

### Wang, BETA, Dong2023

세 자료는 low-calibration signal baseline과 transport sensitivity에는 유용하다. 그러나 각
corpus 안에서 reference, hardware, montage와 interface가 고정되고 실제 block/participant
impedance가 없다. Dataset 간 차이를 metadata 효과로 회귀하면 site, device, population,
task와 preprocessing bundle을 dataset ID로 학습하게 된다. 따라서 A_QM−A_Q의 primary
식별자료로 쓰지 않는다.

### 현재 사용 가능한 보수적 분할

S1–S3은 두 차례 outcome reveal이 끝나 영구적으로 추가 학습, prediction, 새 metric과
model selection에 사용할 수 없다. S4–S102 query 성능은 아직 열리지 않았다.

새 exact owner freeze가 생긴다면 기존 outcome-blind 39/60 allocation을 그대로 유지하는
것이 가장 보수적이다.

- 39명: 각 outer 13명이 자기 fold의 fit/epoch 선택에서 빠지는 cross-fitted nested
  mechanism-development. Fold별 모델의 나머지 training participant는 서로 겹치며 이 gate는
  formal population confirmatory test가 아니다.
- 60명: held-participant evaluation
- 60명의 block 1–5 label: atomic runner 내부의 허용된 calibration support
- 60명의 block 6–10 label: 모든 role×budget이 끝날 때까지 봉인되는 query outcome

k>0에서는 이 60명을 “target label을 전혀 보지 않은 strict-k0 lockbox”라 부르지 않는다.
정확한 표현은 “sealed within-participant calibration partition을 가진 held-participant
evaluation”이다. 39명에서 사전 수치화한 eAUC increment와 correct>shuffle의 평균·outer-fold
일관성, A_Q(k5) viability, exact-missing 및 metadata-only 구조 불변식이 hard promotion gate를
통과하기 전에는 60명을 열지 않는다. Median·참가자 양수 비율과 stale 세 지표는 필수
diagnostic report다. Wrong-context 평균·두 tail과 correct-context 두 tail의 다섯 safety check만
robustness deployment qualification을 정한다. 이 11개 report는 일곱 hard gate에 들어가지 않고
held efficacy 접근을 막지 않는다. 이 수치 gate의 판정 코드는 이제 고정했지만 owner execution
receipt가 없으므로 현재
39명과 60명 모두 실행 승인 상태가 아니다. Development checkpoint key는 `(seed, outer_fold)`,
held checkpoint key는 `(seed)`다. 서로 다른 outer fold를 같은 checkpoint로 묶어서는 안 된다.

Source recipe는 39명을 outcome-blind seed로 13명씩 세 outer fold로 고정한다. 각 fold에서 남은
26명 중 20명으로 fit하고 6명으로 epoch를 고른 뒤, 같은 seed로 다시 초기화해 26명 전체를 그
epoch 수만큼 refit한다. Stage 1은 M을 exact-off한 채 backbone·source anchor·Q precision을
학습한다. Stage 2는 common state를 hash로 고정하고 M residual만 학습한다. Outer 13명 결과는
hyperparameter나 epoch를 바꾸는 데 쓰지 않고 one-shot development gate에만 쓴다. Gate가
통과하면 세 inner-selected epoch 수의 seed별 median으로 39명 전체를 refit해 held checkpoint를
만든다. Held 참가자에서는 어떤 gradient update도 하지 않고 support sufficient statistics만
closed form으로 더한다.

## 통계와 검정력

Primary는 participant별 eAUC difference 60개에 대한 one-sided paired t inference다.
세 source-training seed `[42,43,44]`의 class probability를 query별로 먼저 평균한 뒤 argmax와
BA를 계산한다. 결정론적 고전 baseline은 seed 없이 raw class score 한 세트만 만들고, 방법 간
confidence calibration 비교에는 쓰지 않는다. Sign-flip과 bootstrap은 sensitivity로만 함께
보고한다. Missing role, budget,
condition 또는 seed를 complete-case로 버려 구제하지 않고 primary 전체를 무효로 처리한다.

Outcome-free analytic sensitivity에서 N=60, one-sided alpha 0.05일 때:

| paired SD | effect +0.01 | effect +0.02 | effect +0.03 |
|---:|---:|---:|---:|
| 0.03 | 0.818 | 약 1.00 | 약 1.00 |
| 0.04 | 0.606 | 0.985 | 약 1.00 |
| 0.06 | 0.356 | 0.818 | 0.985 |
| 0.08 | 0.246 | 0.606 | 0.890 |

따라서 N=60은 paired SD가 0.06 이하라면 2%p 정도의 평균 차이를 검출할 가능성이 있지만,
1%p 효과나 calibration-saving 비열등성은 여전히 약할 수 있다. 구현 초안은 실용적 eAUC
SESOI를 `0.020`으로 둔다. Primary는 lower CI>0 및 one-sided p<.05이고 평균 효과도
0.020 이상일 때만 promotion한다.

Calibration-saving 비열등성 margin은 `1/60=0.01667`로 둔다. 한 interface의 고정 query
60회에서 평균적으로 추가 오답 한 개, 즉 dry/wet를 합친 participant outcome에서는 평균 두
오답까지 허용한다는 뜻이다. “두 interface 전체 120회 중 한 오답”을 뜻하는 `1/120`이 아니다.
One-sided alpha는 .025이고,
두 endpoint의 평균 BA가 모두 0.50 이상이어야 chance 부근의 무의미한 비열등성을 절약으로
부르지 않는다. 이 SESOI·margin·utility floor는 검정력 때문에 넓히지 않으며, human 실행 전
owner receipt에서 최종 승인해야 한다.

여기서 test power와 최종 claim 확률은 다르다. H1은 유의성뿐 아니라 관측 평균≥0.020도
요구하고, 절약 주장은 H2a와 H2b를 모두 요구한다. 최종 200,000-draw outcome-free receipt의
계획 중심값(각각 H1 효과 0.030, calibration-value 0.030, k1-vs-k3 차이 0.000)에서 H1 통과는
약 0.903이었다. 두 endpoint의 평균 BA≥0.50 floor를 충족한다고 가정한 held-core contrast
sensitivity에서 H1 뒤 H2 intersection까지의 절약 주장 확률은 약 0.657이었다. 공통 A_Q(k3)를
반대 부호로 공유하는 상관, adverse correlation, t5, 20% harmed mixture에서도 전체 확률은
약 0.656–0.661이었다. **사전 고정한 normal global-null DGP**에서 full claim은 200,000회 중
0회 관측됐고, 이는 그 simulation에서의 관측 빈도이며 95% rule-of-three 상한은 약
`1.5e-5`다. 보편적인 FWER 상한을 뜻하지 않는다. Q-only curve가 평평한 plateau는 0.00122,
H1과 H2a가
각각 SESOI 경계인 경우는 약 0.197이었다. 이는 결과가 아니라 설계 민감도다.

같은 planning mean/SD, shared-A_Q(k3) 상관과 BA-floor 충족 가정에서 full H2 확률은
N=60/75/80/83/90에서
0.657/0.761/0.789/0.805/0.834였다. 따라서 현재 60명은 H1 검정에는 충분한 편이지만 두-block
절약까지 80% 수준으로 만들려면 같은 구조의 외부 held participant 약 23명이 더 필요하다.
H2a 또는 H2b 하나만 귀무경계, 나머지를 유한한 0.030 대립값에 둔 configured partial-null의
full-claim 확률은 각각 0.00435와 0.01372였다. 이는 union-null supremum이 아니다. Strong
control의 근거는 각 component local test와 intersection-union 구조이며, 해당 local 확률은
0.00486과 0.02440이다. Held correct>shuffle/stale Holm family의 standalone planning
sensitivity에서 두 endpoint 동시 claim은 약 0.946–0.957이다. 이는 core가 열렸다고 가정해 따로
생성한 분포이지 엄밀한 조건부 확률이나 전체 pipeline 성공확률이 아니다.

39명 metadata-only shortcut은 임의 classifier 성능을 재는 셀이 아니다. M이 block 내에서
상수이고 각 block에 12 class가 정확히 한 번씩 있으며 API에 row identity/order가 없으면,
metadata-only balanced accuracy는 구조적으로 정확히 1/12다. 이 불변식이 깨지면 실행을
중단한다.

39명 hard development gate는 다섯 stochastic 조건과 두 structural 조건만 사용한다.

- A_Q(k=5) 평균 BA≥0.50
- eAUC increment 평균≥0.010 및 세 outer-fold 평균 모두>0
- correct−shuffle 평균≥0.010 및 세 outer-fold 평균 모두≥0
- all-missing probability exact equality
- metadata-only 구조적 BA=1/12 증명

Median, 양수 참가자 24/39와 stale 세 지표는 필수 diagnostic report다. Wrong-context 평균·두
tail과 correct-context 두 tail의 다섯 safety check만 robustness deployment qualification을
정한다. 이 11개 report는 일곱 hard gate에 포함되지 않으며 held efficacy 접근을 막지 않는다.
계획값에서 18-check all-check 통과확률은 normal/t5/harmed 약 0.565/0.535/0.567이었지만, 이 최소
비중복 hard gate는 약 0.784/0.790/0.786이다. Contrast 효과만 0이고 A_Q(k5) baseline은 정상
0.55인 metadata-null의 hard promotion은 0.00953이었다. Baseline도 0.48인 joint-bad null은
0.00021이다. 전자를 metadata-null **false-go** 해석에 사용한다. 반대로 계획 대립값에서 hard
gate의 false-no-go는 normal/t5/harmed에서 약 0.216/0.210/0.214다. 이 development screen은
formal population test가 아니라 outcome-free no-retry 승격 규칙이다.

최종 simulation은 H1, H2a, H2b의 participant contrast 세 개를 함께 생성한다. 특히
`C=A_Q(k3)-A_Q(k1)`와 `S=A_QM(k1)-A_Q(k3)`는 같은 A_Q(k3)를 반대 부호로 공유하므로 음의
상관을 포함한다. Independent, shared-A_Q3, adverse correlation matrix와 normal/t5/20%-harmed,
null/plateau/harmful/SESOI-boundary를 200,000 draw씩 고정했다. Receipt는 사람 EEG outcome을
전혀 사용하지 않았으며, 현재 plan hash에 결속되어 있다.

## 반드시 들어갈 반증 실험

같은 A_QM checkpoint에서 다음을 바꾸어 본다.

- correct context
- all-missing context
- 같은 participant×interface 안에서 support pool(1–5)과 query pool(6–10)을 넘지 않는
  block-level derangement
- 실제 이전 block의 stale impedance; block01은 명시적 all-missing
- 반대 interface context
- support에만 M을 쓰는 진단과 query에만 M을 쓰는 진단
- interface-only와 impedance-only
- block-constant M과 class-balanced block 구조가 metadata-only BA=1/12를 강제하는 proof

Correct가 shuffled보다 낫지 않으면 metadata-specific pairing mechanism을 주장하지 않는다.
Stale은 temporal-specificity를 보는 supportive sensitivity다.
Missing 또는 wrong metadata가 A_Q보다 크게 나쁘면 배포 안전성 실패다. Metadata-only가
비정상적으로 높으면 ID/condition/label shortcut을 먼저 의심한다. Synthetic channel
corruption은 primary가 아니라 engineering robustness로만 남긴다.

Held confirmatory core가 통과한 뒤 correct>shuffle과 correct>stale 두 mechanism endpoint를
Holm alpha 0.05로 검정한다. Held wrong-context 평균과 participant tail은 별도 deployment-safety
보고이며 correct-context efficacy core를 사후에 뒤집는 추가 검정으로 쓰지 않는다. 39명에서도
wrong/correct tail은 robustness deployment qualification만 정하며, 60명 efficacy access는 앞의
일곱 hard gate만 결정한다.

## 직접 비교해야 할 선행방법

Few-shot SSVEP는 이미 활발한 분야다. 따라서 “적은 calibration을 처음 해결했다”는 주장은
불가능하다.

- [CSDuDoFN/OS-SSVEP](https://doi.org/10.1016/j.neunet.2024.106734)은 arXiv CSDuDoFN에서
  이어진 같은 one-shot lineage이며, target class마다 한 trial을 쓰고 source transfer,
  SAME, eTRCA와 TDCA를 결합한다. 두 개의 독립 baseline으로 중복 계산하지 않는다.
- [SSVEP-DAN](https://arxiv.org/abs/2311.12666)은 source EEG를 target 보정자료처럼 정렬해
  limited-calibration 성능을 높이는 직접 선행이다. 원법은 class당 최소 2 trial이 필요하므로
  우리 complete-block grid에서는 k=3/5만 호환되며 k=1 재현으로 부르지 않는다.
- [Cross-domain template transfer](https://doi.org/10.1088/1741-2552/abcb6e)는 target
  class마다 2–5 trial과 고정 test set을 사용한 LST/TRCA 계열 비교다.
- [Wearable SSVEP dataset paper](https://doi.org/10.3390/s21041256)는 wet/dry,
  block-level impedance와 원래의 cross-electrode transfer 근거를 제공한다.
- [Dynamic Spatial Filtering](https://doi.org/10.1016/j.neuroimage.2022.118994)은
  query-derived channel reliability의 강한 인접 기준이다.

우리의 조건부 차별점은 few-shot 자체가 아니다. **동일 calibration budget에서 pre-query
physical context의 순증분을 강한 Q-only 기준과 correct/shuffled/missing controls로
검증하고, 이를 complete-block calibration curve 및 절약량으로 연결하는 것**이다. 현재
검토한 문헌에서는 이 조합의 직접 SSVEP 선행을 찾지 못했지만, systematic-review 완전성이나
최초 주장은 아직 하지 않는다.

현재 CCA/FBCCA 구현은 retired physical 경로의 보호장치 때문에 wearable_v3 outcome을
의도적으로 거부한다. 새 baseline도 이 보호를 우회하지 않고, 향후 모든 A_Q/A_QM
role×budget×condition과 같은 atomic runner·query identity 안에서만 별도 승인한다. Mandatory
resource-matched core는 k=0 strict FBCCA, k=1/3/5 supervised template correlation, k=3/5 target
filter-bank ensemble TRCA, 동일-checkpoint A_Q, 그리고 k=1/3/5 Chiang 2021 LST+filter-bank
eTRCA다. LST는 source EEG를 target 공간으로 옮기는 직접 transfer comparator로 선정했고
clean-room core와 label-free query adapter를 구현했다. LST k=1은 protocol-adapted 결과이지
원 논문의 재현이라고 부르지 않는다. Target-only single-trial TRCA는 수학적으로 퇴화하므로
k=1 결과를 만들지 않는다. Plain TDCA는 source-transfer comparator가 아니어서 k=3/5
optional이고, CSDuDoFN/OS-SSVEP lineage의 k=1과 SSVEP-DAN의 k=3/5는 owner freeze 전
code·license·정보권한 가능성을 반드시 판정하고, 실행하지 못하면 이유를 명시한다. 논문의
보고 숫자를 같은 표의 직접 순위로 사용하지 않는다.

## 더 필요한 데이터

Wearable만으로 좁은 within-device 연구는 가능하다. Broad acquisition generalization에는
새 자료가 필요하다. 가장 가치 있는 수집은 다음과 같다.

- 같은 participant와 동일 12-target task
- wet, dry 또는 semi-dry interface의 무작위·counterbalanced 반복
- 가능하면 interface×device 또는 interface×reference factorial
- 최소 두 day/session과 cap 재장착
- 매 block 전 per-channel impedance와 bad/contact flag
- digitized 또는 제조사 기준 3D channel 좌표
- reference/ground, amplifier, gain, hardware/software filter, firmware, 전원주파수
- 실제 cue, gaze, rest, feedback를 포함한 calibration 시간

신규 수집에는 IRB/윤리심의, 동의, 장비와 연구자 절차가 필요하다. Nakanishi는 signal
replication 후보지만 paired interface/impedance 자료가 아니고 재사용 권한 확인도 필요하다.

## 실행 동결 상태

방법, 통계, 입력 봉인, job capability, 격리 worker, producer/finalizer receipt와 lifecycle은
구현됐다. 실행계약은 source-development `candidate 9 + baseline 15 = 24` jobs, 조건부 held
`candidate 3 + baseline 5 = 8` jobs다. Baseline에는 strict FBCCA, target template,
target filter-bank eTRCA, SAME3 one-shot component와 Chiang-2021 LST port가 들어간다.
CSDuDoFN/OS-SSVEP와 SSVEP-DAN은 저자 코드에 license가 없고 현재 protocol과 구현이 모호하여
감사된 `NOT_RUN`으로 고정했다. 이는 숫자를 조용히 누락한 것이 아니라 재현이라고 부를 수 없는
구현을 억지로 포함하지 않겠다는 결정이다.

Owner는 N=60에서 full-H2 계획 민감도가 약 0.657임을 포함한 exact source→conditional-held
실행을 위임 승인했다. 이 승인은 외부 전자서명이나 별도 OS principal 권한이 아니다. Lifecycle은
run마다 password-encrypted Ed25519/RSA private key를 만들고, query label은 finalizer 공개키로
암호화하며, worker는 Bubblewrap namespace에서 raw asset·ciphertext·private key·다른 job을 보지
못한다. Phase별 영구 attempt key로 outcome-access claim을 `renameat2(RENAME_NOREPLACE)` 공개한
직후에만 finalizer가 query outcome을 복호화한다. 같은 Unix UID가 registry 파일을 지울 수 있다는
한계 때문에 이 irreversibility는 별도 보안 주체가 아니라 governance·audit 강제다.

정상 완료는 source/held 전체 artifact chain을 서명된 pending receipt 상태에서 먼저 검증하고,
두 private key 삭제와 directory fsync 뒤에만 canonical completion으로 원자 공개한다. Key 삭제 뒤
publication만 중단된 경우에는 pending 전체를 다시 검증한 뒤 rename만 복구할 수 있다. 반면 어떤
phase든 outcome claim 이후 계산이 실패하면 같은 cohort 자동 재시도나 held-only 이어달리기를
금지하는 terminal fail-stop이다. Source gate가 FAIL이면 held directory 자체를 만들지 않는다.

남은 동결 작업은 current plan에 power receipt를 재결속하고 전체 회귀검사를 통과한 clean commit과
annotated tag를 만드는 것뿐이다. 실행은 그 tag의 단일 commit clone에서만 허용한다.

Private staging에는 raw `sample_id` 대신 실행 중 secret으로 만든 opaque `q_<HMAC-SHA256>`
token만 둔다. Token은 seal-decision, phase, checkpoint group과 query/support/source-fit purpose로
domain separation한다. Secret 자체는 manifest에 저장하지 않고 domain-separated commitment만
결속한다. Unlabeled reader는 HDF5의 `x`와 mask만 읽고 `y`가 없어도 작동해야 하며, 각 row의
base-input hash를 실제 signal bytes에서 재계산한다. HDF5 dtype/shape도 exact float32
`[N,64,400]`와 bool `[N,64]`로 제한한다. Confirmatory model은 raw low-level forward가 아니라
resolver가 정한 interface·impedance·mode·ablation만 받는
`forward_cell_from_precomputed`를 사용한다. Episode별 spectral embedding과 observed Q는 한 번
계산하지만 캐시에는 structural/Q field만 들어가고 raw impedance/interface M은 들어가지 않는다.
Q-off cell은 준비된 Q 값을 logits에 전달하지 않고 exact zero를 사용한다. 따라서 shared job이 Q를
미리 계산할 수는 있어도 A_0/A_M 수치 결과는 Q 값에 의존하지 않는다.
Staging의 예상 밖 column은 모두 거부한다.
A_Q와 all-missing A_QM의 exact equality, query·support identity, complete score grid, phase dispatch
hash, resolved context-usage와 producer receipt를 label join 전에 검증한다. 그 뒤에만 sidecar
label을 한 번 join하고 전체 결과를 원자적으로 공개한다. Development는 fold별 composite
checkpoint를, held는 seed별 composite checkpoint를 사용한다. Partial metric은 publish하지 않는다.

이전 query-only CUDA forward와 저장소 회귀검증은 준비 증거일 뿐 사람 outcome 결과가 아니다.
Clean tag 검증 뒤에도 source 39명 gate를 먼저 완결하며, PASS 전에는 held 60명 label·prediction을
열지 않는다.

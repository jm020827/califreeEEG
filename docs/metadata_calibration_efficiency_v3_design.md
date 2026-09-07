# Metadata-assisted low-calibration SSVEP — V3 design과 terminal development 결과

상태: **V3 synthetic development `DEVELOPMENT_NO_GO` terminal / development-v5 0/9 eligible /
selected method 없음 / exact V3 candidate 종료 / scientific lockbox·외부 EEG·wearable EEG
outcome 미개봉**

기준 후보는 `metadata-calibration-efficiency-v3`, 결정 ID는 `DEC-20260906-024`다.
기계 판독 기준원은
`configs/analysis/metadata_calibration_efficiency_v3.yaml`, synthetic 기준원은
`configs/analysis/metadata_calibration_v3_synthetic.yaml`, V2와의 최초 경계는 byte-identical
`configs/governance/metadata_calibration_v3_preoutcome_amendment.json`, 최종 pre-development recovery 경계는
`configs/governance/metadata_calibration_v3_4_recovery_amendment.json`이다.
최종 수치·해석·artifact integrity는
[V3 synthetic development 결과](metadata_calibration_efficiency_v3_results.md)에 기록한다.

## 결론부터 말하면

상위 연구목표는 바꾸지 않지만, 이 문서가 동결한 **exact V3 candidate는 종료됐다**.

> 처음 보는 사용자가 정답이 붙은 SSVEP calibration EEG를 아주 조금만 제공할 때,
> EEG 자체에서 얻은 품질 단서 `Q` 외에 query 전에 측정한 전극 interface와 impedance
> `M`을 알면 필요한 labeled calibration block을 실제로 줄일 수 있는가?

Development-v5는 family당 48 participants와 3×3 grid를 완전 평가했지만 eligible cell이
`0/9`였고 `selected_grid_cell_id=null`인 `DEVELOPMENT_NO_GO`로 끝났다. 모든 cell에서 AQ
viability, B3/B4 metadata efficacy, B4 in-reference/interface-scale value, k3 pairing mechanism과
pairing potency가 함께 실패했다. 반면 불변식, null/harm, deployment viability, adversarial
abstention과 anti-triviality는 통과했다. 안전성 통과를 효능으로 읽지 않는다.

B2 A0 balanced accuracy가 `.5850694444444444`로 개선 여지가 있었는데 AQ eAUC gain은 0이어서
baseline ceiling만으로 no-go를 설명할 수 없다. k1에서 M은 AQ의 나빠진 log probability를
일부 완화했지만 BA는 A0와 같았고, k3/5는 gate가 exact A0로 abstain했다. Forced-on 사후 진단도
유의미한 BA gain을 만들지 못했으므로 gate가 양성 효과를 숨긴 것이 아니다.

V3에서 바꾸는 것은 질문이 아니라 **M을 넣는 위치와 실패 판정 방식**이다. V2는 M을
support prototype과 최종 fusion에 두 번 반영했다. V3는 M을 오직 “이 calibration을 현재
query에 얼마나 믿을 것인가”라는 최종 support-residual 계수 한 곳에만 쓴다. Metadata가
없으면 AQM은 AQ와 bitwise exact하게 같고, k=0이면 둘 다 strict FBCCA인 A0와 같다.

V3는 V2의 버그 수정 재시도가 아니다. V2 V11은 scientific participant 생성 전에
post-claim validator 재귀로 terminal이 됐고 결과가 없지만, 이미 단회 lockbox가 소비됐다.
더구나 공개된 engineering-only 진단에서도 support 이득과 metadata pairing 효과가 작았다.
따라서 새 candidate/schema, 새 개발 seed, 새 operator, 새 promotion intersection과 독립된
권한 DAG를 사용한다. V2 deny overlay와 artifact는 그대로 보존한다.

## 가장 쉬운 모델 설명

| 방법 | 무엇을 보는가 | 쉬운 비유 |
|---|---|---|
| A0 | 현재 query EEG와 전체 주파수 codebook | 아무 개인 보정 없이 쓰는 강한 기본 판독기 |
| AQ | A0 + 정답이 붙은 소수 support EEG + EEG-derived Q | 새 사용자가 자주 내는 오류 패턴을 조금 반영 |
| AQM | AQ + query 전에 얻은 interface/impedance M | 그 support가 지금 측정 조건에도 믿을 만한지 조절 |

핵심 비교는 `AQM-AQ`다. A0와 AQM만 비교하면 labeled support와 metadata의 효과가 섞여
“metadata가 도움됐다”고 말할 수 없다. 같은 support, 같은 Q, 같은 query, 같은 FBCCA를
쓰고 M 접근권한 하나만 다른 AQ와 AQM을 비교해야 한다.

실제 wearable primary는 dry 안에서 dry support/query를, wet 안에서 wet support/query를
따로 평가한다. 따라서 단순한 interface same/different 값은 항상 `same`이라 정보가 아니다.
V3는 이 상수 shortcut을 제거한다. 대신 source의 outcome을 보지 않고 wet/dry별 impedance의
통상 변동척도를 먼저 동결한 뒤, 같은 interface 안에서 calibration block과 query block의
impedance가 평소 변동에 비해 얼마나 멀리 떨어졌는지를 본다. AQ가 불일치 support를
따라가려 할 때 AQM은 residual을 줄인다. 이 가설은 “dry가 wet보다 나쁘다”도 아니고
“impedance가 정답 class를 알려 준다”도 아니다. **해당 interface의 통상 척도로 해석한
support-query 접촉상태 불일치가 support의 전이 가능성을 알려 주는가**가 가설이다.

## 정확한 V3 연산

먼저 strict FBCCA의 확률을 `p0`, class-labeled support score prototype으로 만든 확률을
`ps`, support block 수를 `k`, FBCCA normalized entropy를 `H(p0)`라 둔다.

각 complete support block `b`에서는 그 12개 trial의 관측 label에 strict FBCCA가 준
확률의 평균을 `w_b`로 두고 `[1e-12,1]`로 clip한다. `pi_b=w_b/sum(w_b)`이며, block 하나씩
독립적으로 만든 P3 query 확률 `ps_b`를 `ps=sum(pi_b*ps_b)`로 합친다. 이 상대 가중치는
AQ와 AQM이 bitwise 동일하게 쓴다. FBCCA가 체계적으로 틀린 경우가 바로 calibration으로
고쳐야 할 대상이므로 `w_b`가 낮다는 이유로 support 전체를 끄는 절대 gate는 두지 않는다.
k=1에서는 `pi_1=1`이다. k≥3의 실제 사용 여부는 아래 prequential 성능 gate가 결정하며,
uniform block-weight 결과도 사전 지정 sensitivity로 함께 보고한다.

`lambda_Q = lambda_max * k/(k+1) * H(p0)`

`p_AQ = (1-lambda_Q) * p0 + lambda_Q * ps`

AQ prototype은 M을 전혀 보지 않는다. 각 block에서 class `c`의 smoothed ideal prototype
`u_c`, pseudocount `nu`, 그 block에서 관측 label이 `c`인 한 support score probability
`p_bc`를 이용해 다음처럼 만든다.

`r_bc = (nu*u_c + p_bc) / (nu + 1)`

query score distribution과 각 `r_bc` 사이의 negative Jensen–Shannon divergence가 `ps_b`의
class score가 된다. `nu ∈ {1,4,16}`, `lambda_max ∈ {0.1,0.2,0.3}`의 9개 조합은
clean implementation commit과 별도 development-bundle receipt 뒤 nonreserved synthetic
development에서 완전 grid로 검사했다. 모든 gate를 통과한 조합이 없어 V3를 종료했고,
`selected-method-freeze`는 만들지 않았다. 두 YAML은 scientific 실행계약이 아니라 selection
계약이었다. 이제 operator·DGP·threshold·grid·rank rule을 바꾸려면 새 candidate/scientific
revision과 새 미관측 evidence가 필요하다.

Raw interface/impedance는 metadata adapter만 읽는다. 먼저 signal을 전혀 받지 않는 preflight가
schema·pairing·query OOD와 block affinity를 검사한다. Malformed/OOD면 support EEG·label을
읽기 전에 A0로 끝낸다. 통과한 경우에만 별도 M-free builder의 block reliability capability와
preflight capability를 finalizer가 결합한다. Reliability capability는
candidate/operator, support block 순서, support EEG+label manifest와 자신의 hash를 묶는다.
최종 adapter 출력도 두 입력 hash와 query/support key, context-reference/pairing hash를 모두 묶고,
scientific operator는 현재 signal state와 일치하는 typed capability만 받는다.
AQM은 interface로 정규화한 query-support impedance 거리를 `g_M ∈ [0,1]`로 바꾸고 다음
한 줄만 다르다.

`p_AQM = (1-lambda_Q*g_M) * p0 + (lambda_Q*g_M) * ps`

- Domain별 outcome-free reference metadata만으로 interface×channel별 `log1p(kOhm)` median과
  `1.4826*MAD`를 fit한다. Scale floor는 0.05, 최소 count는 128이다. 부족하면 pooled-channel
  center+scale pair, 그것도 부족하면 그 channel을 unavailable로 처리한다. Pooled center는
  모든 interface의 raw `log1p` 값을 합친 channel별 median, pooled scale은 같은 값들의 pooled
  center 기준 `1.4826*MAD`다. Unknown interface의 OOD z-score도 이 pooled pair를 쓴다.
- Synthetic은 EEG/outcome key와 독립인 seed `20260910` covariate-only reference만 쓴다.
  향후 wearable human 평가는 별도 human-metadata 입력 권한 뒤 exposed source39의 metadata만
  쓰며, 두 domain의 receipt는 서로 대체할 수 없다. Held metadata로 다시 중심화·정규화하지 않는다.
- Query block `q`와 각 support block `b`에서 함께 관측된 채널의 standardized `log1p`
  차이 median을 `d_qb`, affinity를 `a_qb=2^-clip(d_qb,0,16)`으로 둔다. 비교 채널이 없는
  support block은 missing 자체가 불일치 증거가 되지 않도록 `a_qb=1`의 exact neutral로 둔다.
- 여러 support block에서는 위 M-free `w_b`를 그대로 써
  `g_M=sum(w_b*a_qb)/sum(w_b)`를 계산한다. `w_b`는 AQ/AQM의 공통 `ps` 혼합에도 이미 쓰이므로
  AQM만 signal-quality main effect를 독점하지 않는다. M은 이 마지막 scalar에만 들어간다.
  Metadata packet을 block 간 옮겨도 `w_b`는 원 EEG+label block에 남으므로 pairing이 깨진다.
  k=1에서는 weight가 상쇄되고 그 한 packet의 `a_qb`가 된다.
- 유효 support block이 없거나 계산된 `g_M=1`이면 산술 없이 AQ를 그대로 반환한다.
- Interface category는 normalization table의 lookup key일 뿐 one-hot/logit 입력이 아니다.
  Impedance가 없으면 interface 이름만 바꾸어도 언제나 bitwise AQ다. 같은 standardized
  mismatch라면 wet/dry 이름이 달라도 결과가 같아야 한다.
- Unknown interface는 pooled scale로 fallback하고, 유효 채널이 없으면 AQ다. 음수·비유한
  값은 해당 채널 missing으로 취급한다. Schema/pairing digest 오류는 support를 읽기 전에
  A0로 돌아간다. 관측 채널 과반이 source 기준 |z|>8이면 predeclared OOD guard로 A0다.

이 형태에서 M은 support를 AQ보다 더 강하게 쓰게 만들 수 없고, 조건 불일치 시 덜 믿게만
한다. 그래서 V3의 주장은 “M이 새 class evidence를 제공한다”가 아니라 “M이 잘못 전이될
calibration을 식별하는가”로 좁아진다. `L1(p_AQ-p0) ≤ 2*lambda_Q`이고 AQM의 변화는 그보다
작다. 이 수치 bound와 exact fallback은 소프트웨어 성질이지 실제 사람에게 손상이 없다는
임상적·통계적 보장은 아니다.

## k=1을 정직하게 다루는 방법

k=3 이상이면 block 1로 block 2를, blocks 1–2로 block 3을 예측하는 chronological
prequential 검사가 가능하다. 모든 fold에서 BA가 나빠지지 않고 correct log-probability가
개선되며 적어도 한 fold의 BA가 좋아질 때만 support update를 켠다. 이 gate는 M을 보지 않는
AQ 대 A0로만 계산하고 그 결정을 AQ와 AQM이 bitwise 공유한다. Metadata가 gate라는 두 번째
경로로 들어가는 것을 막기 위해서다.

k=1은 하나뿐인 trial/class를 hold-out하면 학습 표본이 사라진다. 따라서 target-person
내부에서 독립 validation을 만들 수 없다. V3는 다음만 허용한다.

- V3 개발계약에서 결과 전에 `true`로 고정한 global k=1 enable 결정
- query 전에 관측 가능한 missing/unknown/schema/OOD context guard
- 그럴듯하게 잘못 붙은 support label을 k=1에서 탐지한다는 주장 금지
- risk–coverage와 harmed-participant 비율을 경험적 진단으로 보고

HOSO는 K≥2가 필요하고 exact zero-adaptation 선택을 제공하지 않으며, conformal risk
control도 별도 calibration 표본과 exchangeability 가정이 필요하다. 따라서 이런 인접
방법을 인용해 k=1 개인별 무손상을 주장하지 않는다.

## V3 synthetic이 무엇을 반증해야 하나

V2의 “B1/B2/B3 중 두 개” 규칙은 metadata가 직접 유효하지 않아도 AQ family 두 개만으로
promotion될 수 있었다. V3는 모든 이름 붙은 component의 **교집합**을 요구한다.

1. 불변식: label/row leakage 없음, support-query 분리, k0 exact A0, missing exact AQ,
   M의 단일 insertion, probability/bound, 실제로 달라지는 control을 모두 통과한다.
2. AQ viability: participant confusion B1 또는 phase/spatial shift B2 중 적어도 하나에서
   `AQ-A0` 단측 LCB>0, 평균≥0.01, k1 평균≥0이다.
3. Impedance efficacy: B3에서 `AQM-AQ` LCB>0, 평균≥0.01이다.
4. Interface-calibrated impedance efficacy: B4는 support/query를 같은 wet 또는 dry 안에서
   평가하되, 두 interface가 서로 다른 impedance offset·scale·noise를 갖게 한다. Decoder는
   latent state나 생성식을 보지 않고 별도 covariate-only reference에서 robust scale만 fit한다.
   `AQM-AQ` 기준과 함께 correct scale이 pooled/wrong-interface scale보다 나은지도 보고한다.
   Interface-only arm은 exact AQ여야 하며 독립 interface 인과효과를 주장하지 않는다.
   Participant-level source-range stress는 개발 5명/48명, 향후 synthetic 10명/96명으로 정확히
   고정한다. 전체 participant 결과뿐 아니라 stress를 제외한 43명/86명도 동일한 metadata
   efficacy 기준을 독립적으로 통과해야 하므로 OOD→A0 fallback만으로 양성 결과를 만들 수 없다.
   Stress query의 valid channel standardized coordinate는 정확히 `+10`으로 두어 |z|>8 guard가
   구조적으로 발동하게 하고, 그 EEG는 바꾸지 않는다. 이 B4는 지정한 monotone mapping 아래
   구현·기전 assay일 뿐 실제 장치 간 transport robustness 증거로 해석하지 않는다.
5. Pairing mechanism: k=1에는 다른 observed support packet이 없어 정직한 block-pair control을
   만들 수 없다. 같은 packet의 impedance/availability channel rotation은 wiring 구조 진단으로만
   보고 positive pairing/efficacy 주장에는 쓰지 않는다. 다만 N4에서 rotation이 가짜 효과를
   만들지 않는다는 null-equivalence 안전조건은 필수다. 필수 positive mechanism은 k=3이고, k=3/5에서는
   현재 support prefix의 whole-packet derangement를
   사전식 순서로 모두 열거한다(k=3은 2개, k=5는 44개). 각 derangement의 BA와 log score를
   participant 안에서 먼저 평균하며 derangement를 독립 표본으로 세지 않는다. Packet은 바뀌어도
   affinity가 같을 수 있으므로 mean-derangement `|Δg_M|≥0.05`인 unit이 각 B3/B4 potential
   unit의 50% 이상이고 전체 median도 0.05 이상이어야 한다. 부족하면 효과 0이 아니라 assay invalid다.
   support block 하나에서 shuffle은 no-op이므로 k=1 shuffle을 mechanism 근거로 쓰지 않는다.
6. Null/harm: clean-anchor에서는 margin `-1/60` 비열등, context-null의 metadata contrast는
   paired 90% CI가 `[-1/60,+1/60]` 안에 드는 양측 equivalence를 요구한다. B3/B4의 AQM은
   AQ보다 좋아지는 것뿐 아니라 A0 대비 eAUC LCB>`-1/60`, 평균≥0이어야 한다. AQ와 A0 양쪽
   기준의 severe-harm upper rate<0.10, adversarial k3/5 abstention≥0.95를 요구한다.
7. Anti-triviality: B3/B4에서 metadata가 실제 residual을 바꾼 비율이 0.20 이상이어야 한다.
   항상 AQ 또는 A0만 반환하는 후보는 안전해 보여도 통과하지 못한다.

Synthetic PASS는 구현과 메커니즘의 필요조건일 뿐 사람 EEG의 M 효과가 아니다. 실제로 하나
이상의 필수 component가 실패했으므로 exact V3 후보를 종료하며 threshold, DGP 또는 lambda를
같은 결과에 맞춰 바꾸지 않는다. 후속은 새 candidate/scientific revision과 새 미관측 evidence를
먼저 사전등록해야 한다.

## 최종 사람 EEG 질문은 그대로다

Primary eAUC는 k=0/1/3의 calibration curve 면적이다.

`eAUC = Y0/6 + Y1/2 + Y3/3`

최종 claim 순서는 다음과 같다.

1. `ΔQ = eAUC(AQ)-eAUC(A0)`의 단측 95% LCB가 0보다 크고 관측 평균이 0.020 이상이어야 한다.
2. Primary `ΔM = eAUC(AQM)-eAUC(AQ)`의 participant-paired 단측 95% LCB가 0보다 크고
   관측 평균이 0.020 이상이어야 한다.
3. correct-vs-wrong/shuffle mechanism이 사전 기준을 통과해야 한다.
4. AQ k3−k1의 LCB>0이고 평균≥0.020이어야 줄일 calibration value가 존재한다.
5. AQM k1−AQ k3의 LCB가 `-1/60`보다 크고 둘의 BA가 모두 0.50 이상이어야 36 trials에서
   12 trials로 줄였다고 말한다.

마지막 문장은 labeled-trial burden에 관한 것이다. 실제 총 부담 감소를 주장하려면
impedance 측정·전극 setup 시간, elapsed calibration time과 재측정 횟수도 함께 기록해야 한다.

## 데이터는 무엇이 더 필요한가

- BETA와 Dong2023은 AQ의 signal-only calibration 검증에는 좋지만 interface/impedance가
  없어 `M beyond Q`를 검정할 수 없다.
- Choi2019는 30명×2일 cross-session AQ 복제에는 좋지만 역시 직접 M 검정 자료가 아니다.
- 이미 outcome이 공개된 wearable source 39명은 새 구현의 engineering에는 쓸 수 있어도
  독립 promotion evidence가 아니다.
- 미개봉 wearable 60명은 same-study unseen-participant 평가이며 broad device/site OOD가
  아니다. 모든 앞단 gate와 별도 승인 전에는 열지 않는다.
- 가장 직접적인 외부 후보는 16명×3일 randomized wet/dry와 block impedance를 보고한
  Liu 자료다. raw 접근, 분석·파생결과 라이선스, 동의 범위와 Zhu 102명 cohort 중복 여부를
  저자에게 확인해야 한다. 요청 초안은 [data acquisition 문서](data_acquisition_v2.md)에 있다.
- 요청이 실패하면 randomized/counterbalanced wet/dry 또는 device crossover를 새로
  수집해야 한다. block 전 impedance뿐 아니라 가능하면 시간변화 impedance `Z(f,t)`, motion,
  humidity/contact pressure, condition order, setup/재측정 시간을 prospectively 기록한다.

새 dry-contact ear-EEG preprint는 continuous impedance mismatch로 artifact를 적응적으로
줄이는 가능성을 보였지만 N=4, ear-EEG, alpha-band 연구다. 이는 prospective metadata schema의
근거이지 현재 SSVEP 효능이나 기존 wearable 자료에 소급할 수 있는 증거가 아니다.

## 관련 연구가 V3에 주는 실제 영향

- [CSDuDoFN/OS-SSVEP](https://arxiv.org/abs/2311.07932)은 class당 target one-shot과 source
  transfer를 결합해 UCSD/Benchmark/BETA에서 저보정 가능성을 보였다. 그러나 k=0이나
  acquisition M의 Q 초과 가치는 검정하지 않았다.
- [SSVEP-DAN](https://arxiv.org/abs/2311.12666)은 target 2 trials/class부터 waveform
  alignment가 target-only·일부 transfer baseline을 개선할 수 있음을 보였다. V3에서는 k≥3
  protocol-matched comparator이지 M 효과의 증거가 아니다.
- [LST cross-device transfer](https://doi.org/10.1088/1741-2552/abcb6e)는 device/session
  간 transfer 가능성을 보였지만 우리와 같은 participant-level noninferiority savings와
  M-beyond-Q contrast를 하지 않았다.
- [SS-iTRCA](https://doi.org/10.1109/JBHI.2025.3577813)는 EEG similarity로 source subject를
  선택해 적은 target block에서 개선을 보고했지만 최소 2 target blocks에서 시작하고 exact
  source-off 안전 gate가 아니다. k≥3 보조 comparator로 둔다.
- [REFINE](https://arxiv.org/abs/2505.11771)과
  [HOSO](https://arxiv.org/abs/2603.04341)는 frozen base+residual, validation-free adapter라는
  좋은 구조적 비유를 주지만 finite-sample k=1 SSVEP 무손상을 증명하지 않는다.
- [Conformal Risk Control](https://arxiv.org/abs/2208.02814)은 별도 calibration과
  exchangeability 아래 risk control을 제공한다. 현재 한 target block만으로 그 조건이 생기는
  것은 아니다.
- [continuous impedance mismatch ear-EEG](https://arxiv.org/abs/2609.02777)은 향후 동적
  acquisition metadata 수집을 동기화하지만 SSVEP 직접 근거가 아니다.
- [arXiv:2608.11829](https://arxiv.org/abs/2608.11829)의 retained/learned/forgotten 표는
  평균 향상 뒤의 participant transition을 설명하는 보조 비유로만 쓴다. LLM sampling K와
  EEG calibration k는 같은 양이 아니다.

따라서 broad literature search는 현재 포화에 가깝다. V3 synthetic development가 exact 후보를
종료한 뒤 정보가 큰 다음 단계는 논문 수를 더 늘리는 것보다 (1) 새 AQ가 matched-condition에서
실제 utility를 만드는지 별도 evidence로 확인하고, (2) 그 위에 metadata-selective trust를
사전등록하며, (3) 직접 metadata-bearing 독립 cohort의 접근권한을 얻는 것이다.

## 실행·권한 경계

V3 governance는 다음 한 방향으로만 흐른다.

`plan/snapshot → clean implementation commit + 외부 write-once development-bundle receipt → development → selected-method freeze → full tests
→ public-fixture canary → exact future-beacon attempt manifest → 그 manifest에 묶인 owner
authorization → target 도달 → O_EXCL claim → intrinsic live beacon → typed scientific seed
capability → in-memory validation → result 또는 terminal publication → read-only audit`

Artifact loader는 계약에 선언된 exact canonical path만 읽어 immutable path→bytes mapping을
만든다. Glob·directory scan·symlink alias는 거부한다. Pure parser는 그 bytes mapping만 받아
filesystem, 다른 artifact, git, environment, network, RNG 또는 store를 호출하지 않는다.
출력 경로는 trusted root directory FD에서 각 component를 `O_DIRECTORY|O_NOFOLLOW`로 열고
`fstat`한 뒤 마지막 basename을 parent FD 기준 `O_CREAT|O_EXCL|O_NOFOLLOW`로 생성한다.
이 race-safe primitive가 없으면 실패하며, 사전 realpath 검사만으로 권한을 부여하지 않는다.
seed가 reserved인지 알아보기 위해 artifact directory를 탐색하지 않으며 전역 seed cache를
두지 않는다. Historical beacon canary는 같은 state engine과 non-EEG hash probe를 사용하지만
별도 typed `CANARY_*` enum을 쓴다. Fresh audit까지 끝난 `CANARY_FRESH_AUDITED`만 main
lifecycle의 `VERIFIED→CANARY_PASSED` bridge를 연다. 이때 selected freeze, clean commit/tree,
test evidence, fixture, 전체 canary artifact와 trace·seed·probe를 묶은 별도 fresh-process audit
artifact와 typed capability가 필요하다. Canary capability는 scientific executor가
거부한다. Beacon seed는 domain separator, manifest/fixture hash, timestamp, chain/pulse index와
128자리 uppercase outputValue를 strict ASCII+LF로 SHA-256한 full 256-bit big-endian 정수다.
Canary는 이 순수 primitive와 사전 계산한 hash-probe digest까지 검증한다. Scientific result는 전부 메모리에서
검증된 뒤에만 write-once publish한다.

Canary authorization, claim, beacon, result, fresh audit의 exact filenames는 각각
`authorization.json`, `claim.json`, `beacon.json`, `result.json`, `fresh-audit.json`이며 모두
`/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/governance-canary-v1/` 아래에 둔다.
Fresh audit도 예외 없이 `O_EXCL`, regular-file mode `0400`, parent mode `0700`으로 한 번만
publish하며 기존 파일·replacement·realpath mismatch를 거부한다.

Scientific claim은 attempt directory마다 생기지 않는다. V3 candidate 전체에서 단 하나인
`/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/scientific/global-claim.json`을
live network access 전에 `O_EXCL`로 만든다. Attempt ID는 caller가 고르는 값이 아니라 완전히
검증된 attempt-manifest payload SHA-256에 `sha256-`을 붙인 값이다. 따라서 새 attempt ID나
target을 골라 claim을 반복할 수 없다. 이 global claim은 exact attempt manifest와 owner
authorization의 schema·payload SHA·file SHA를 모두 묶는다.

Future attempt의 seed statement도 임의 문자열이 아니다. Candidate, selected scientific ID,
selected-freeze hash, exact target/HTTPS endpoint와 derivation schema를 고정 순서 ASCII+LF로
재구성한 bytes의 base64만 manifest가 받을 수 있다. Statement는 manifest hash를 포함하지 않고,
완성된 manifest payload hash가 다시 실제 beacon seed derivation에 들어가므로 순환이 없다.
Manifest는 selected freeze, development bundle, test evidence, canary result와 fresh audit를
각각 schema·payload SHA·file SHA로 묶는다. Owner authorization도 manifest의 payload SHA와
file SHA를 함께 묶어야 한다. Manifest는 strict RFC3339 UTC 생성시각을 포함하고, signed-at은
그 시각보다 뒤이면서 target보다 엄격히 앞이어야 한다. Filesystem mtime은 이 판단에 쓰지 않는다.

Development-v1의 외부 write-once root는 bundle과 covariate-only context reference 두 파일,
development-v2 root는 bundle 한 파일, development-v3 root는 bundle·context reference·
development start 세 파일, development-v4 root도 bundle·context reference·development start
세 파일만 가진 채 영구 은퇴한다. 어느 old root에도 terminal이나 result를
소급 생성하지 않으며 기존 파일을 수정·삭제·확장하지 않는다. V2에서는 context seed
`20260910` reference replay만 메모리에서 두 번 실행됐고 development seed는 실행되지 않았다.
V3에서는 같은 V1 context bytes를 채택한 뒤 `development-start.json`을 기록했고, 그 단회
receipt가 old development seed `20260909`를 소비한다.

V3 `resume`은 `execute_complete_development`가 반환한 뒤 governance가 frozen
`mappingproxy` result를 synthetic publication validator로 넘기는 transport 경계에서
TypeError로 끝났다. 따라서 seed authority·SeedSequence·DGP·complete grid와 selection이
메모리에서 완료됐다는 것은 control flow에 따른 강한 추론이지만, result/selection 파일이나
caller에게 노출된 metric·outcome은 없다. 이를 과학적 PASS/FAIL 또는 내구성 있는 outcome
attestation으로 취급하거나 재구성하지 않는다. Exact V3 inventory SHA-256은
`59502b324417c2be6c03aa5483532646fb8630af92de1218e56d862ce72cd6e7`, 실패 line
SHA-256은 `5e21a934ee692d13e3717502442c412d58121547eb2d05ecaf3c1bd827abd04c`다.

V4는 seed `3156745110`을 쓰는 replacement였지만 다시 result publication 전에 멈췄다. Exact
bundle/context/start 세 파일만 남았고 inventory SHA-256은
`511de6688fa315830ddf98ed5a78eefde4e0e4ba6f15977c347281403e908dfa`다. `resume`의 exact LF
failure SHA-256은 `5c6a3fa68152b88b3f32745f04afcc52f3c636dd709eb56a937470b45e7ddbcc`다.
Canonical JSON `sort_keys` 뒤 sensitivity endpoint object 순서가 바뀌었는데 core가 이를 exact
key set이 아니라 insertion order로 판정한 것이 원인이다. V4 seed는 start receipt 때문에
소비됐고 DGP/grid/selection 완료는 강한 control-flow inference일 뿐 durable·관측 outcome이
아니다. V4 result를 재구성하거나 seed를 replay하지 않는다.

당시 계속할 수 있는 유일한 attempt는
`/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/development-v5/`이며 bundle
schema는 `cfeg.metadata-calibration-efficiency-v3.development-bundle.v5`다. 새 bundle은
V1/V2/V3/V4의 exact 2/1/3/3-file inventory, 네 recovery amendment, incident identities,
clean A5 source/runtime/tests를 모두 묶는다. Context는 retired-v1 exact bytes를 복사·재검증하고
retired-v3/V4 context와 byte-identical임을 확인한다. V3/V4를 scientific parent로 삼거나
context를 refit하지 않는다.

Replacement development seed `301269949`은 outcome이 아닌 V4 bundle/context/start와 실패
envelope의 immutable pre-result digest만으로 사전 결정했다. 799-byte canonical preimage
SHA-256 `11f503bdca78e8a7d0e853f9a7b484510dd749018c7735bce61ff7965b70b1cb`을 여덟 개
big-endian uint32로 나눈 뒤 forbidden prior/context seeds 밖의 첫 nonzero word(index 0)를 쓴다.
Scientific projection은 retired V4
`5885f39908d8a33b130e0dc923abe560cc4587ebe7c58ed242647ce028f94712`에서 new
`670b61c767ff2a4b02c3a582aa8ac7e46b5632ada2806ffc109e2e79326d05ca`로 바뀌며 허용된
유일한 leaf는 `synthetic.rng.development_root_seed`다. 새 synthetic evidence는 투명한
development/model-selection evidence이지 human confirmatory evidence가 아니다.

Development-v5의 네 canonical 파일은 `development-bundle.json`, `context-reference.json`,
`development-start.json`, `development-result.json`이다. Start receipt는 development
authority/첫 RNG draw 전에 `O_EXCL`로 기록되고, 그 creator process의 private nominal
capability만 같은 호출에서 계속할 수 있다. Fresh process에서 receipt만 있으면
`DEVELOPMENT_ATTEMPT_CONSUMED` terminal이며 replay하지 않는다. Receipt와 result가 함께
있으면 read-only audit만 허용하고 result가 receipt schema·payload SHA·file SHA를 묶는다.

Transport 수정은 governance parser나 global serializer를 느슨하게 만들지 않는다. Frozen
context/result/selected payload는 기존의 제한된 core 경계에서만 ordinary dict/list로
deep-detach하고 LF-종료 canonical bytes가 원문과 정확히 같은지 확인한다. Sensitivity payload의
`endpoint_order` 배열이 유일한 순서 권위이며 endpoint object는 exact unordered key set이어야
하고 검산·hash iteration은 그 배열 순서를 따른다. Initial result publication과 clean-A/B reopen은
실제 canonical serialize/parse round-trip 회귀를 통과해야 한다.
Post-result audit은 development RNG authority나 participant DGP를 다시 실행하지 않는다.
Selected freeze는 repository의
`configs/governance/metadata_calibration_v3_selected_method_freeze.json`, full test evidence는
canary root의 `test-evidence.json`이다.
Scientific execution authorization의 trust root는 repository에 고정한 workspace-owner
RSA-4096 공개키(fingerprint `SHA256:tm6CDH5eVtjTKNqBUwrBYwbq5RhZ48wo1QjP9c+mR+g`)다.
임의 runtime key는 거부한다. 향후 exact attempt manifest가 생긴 뒤 target 전에 대응 개인키로
RSA-PSS/SHA-256 detached 서명을 받아야 하며, 현재 대화의 일반 승인은 그 target-bound 서명을 대신하지 않는다.

## V3 synthetic development 결과와 종료

Development-v5는 commit `8b3ace24eb4160aa61f3e7bf2af7dc1b74905699`, tree
`04895973c140787546d31a72437d8c90982b698f`에서 seed `301269949`로 단회 실행됐다. Canonical
result는 88,992 participant metric rows와 90 invariant rows를 담으며 file/payload SHA-256은
`1af9f65910da1d753a30a957dfb5bbef7366b696f9f32d444bf95dd411afcc8c`/
`341df8f21cadbd599ca2ef9fe15f09f1d354c0b997198f503ba4ec5c1b0760fd`다. Fresh process가
`DEVELOPMENT_NO_GO`, terminal true, selected null을 read-only로 재확인했다.

9개 cell은 모두 같은 7개 필수 component에서 실패했다. B1 최대 AQ eAUC gain은
`.0001736111111111128`이고 LCB는 `-.00011769561139615749`, B2 gain은 0이었다. B3/B4 metadata와
k3 mechanism efficacy도 0이었다. B4 potency는 changed fraction
`.49166666666666664<.50`, median `.04830195627485313<.05`였다. Null/harm/invariant와
abstention이 통과한 사실은 안전성 evidence이지 효능 evidence가 아니다.

따라서 exact V3를 threshold/lambda/DGP 사후조정이나 다른 seed로 재시도하지 않는다. Selected
freeze, canary, scientific lockbox와 external/human EEG는 열리지 않았다. 다음 후보는 먼저
waveform/spatio-temporal AQ의 matched-condition calibration utility를 입증하고, 그 뒤 block별
metadata affinity를 보존하는 trust operator를 새 evidence와 함께 사전등록해야 한다. 예를 들어
`p0 + lambda_Q * sum_b pi_b * a_qb * (p_s,b - p0)`는 검토할 수 있으나 이는 결과 뒤의 가설이며
동결된 후속 설계가 아니다. Continuous log probability/margin은 진단용이고 balanced accuracy와
labeled calibration burden은 confirmatory endpoint로 유지한다. 자세한 표와 주장 한계는
[V3 결과](metadata_calibration_efficiency_v3_results.md)를 따른다.

## 역사 기록: 구현 준비 상태 (2026-09-07 implementation A 직전)

V3 scientific core, governance와 3-command restartable runner를 구현하고 독립 exact-commit
감사를 통과했다. Core 기준 commit은 `a3573876eae97b6aaabcd959d08bec541438e773`, 최종
governance 감사본은 `b98d6398349561fed0d1dadc32f61a3b875b61dc`, 최종 runner 감사본은
`df8fbfc64488e4533fed44d65de7286f4042dfaf`다. Cherry-pick된 메인의 마지막 code commit은
`b2e86a8`이며, 이 문서·상태·연구일지를 포함하는 다음 clean commit이 development bundle이
묶는 implementation A가 된다.

최종 메인에서 V3 집중 6-suite `144 passed`, 전체 repository suite `850 passed`를 재현했다.
Canonical `status`는 1,668 directories·22,247 files의 site-packages 전체 inventory를 import
전에 검사한 뒤 `BUNDLE_PENDING`을 반환했다. Dependency inventory SHA-256은
`5e33f20babd3457f3a5ede1727d1ed04a3fc5ba927dc51e81b0491f4699e00f9`다. V3 변경 파일의
Ruff lint, in-memory compile, shell syntax와 `git diff --check`는 통과했고 governance/runner
subset의 Ruff format도 통과했다. Repository-wide lint는 기존 비-V3 11건, format은 총
112파일을 지적했으며 그중 exact-audited V3 core/synthetic 5파일도 포함된다. 기능 변경과
무관한 대규모 재포맷으로 감사 snapshot을 바꾸지 않기 위해 자동 수정하지 않았다.

이 시점의 권한은 clean implementation A와 그 외부 write-once bundle까지다. Valid bundle
뒤에만 covariate-only reference seed `20260910`과 nonreserved development seed `20260909`를
각각 한 단계씩 실행한다. Future beacon/scientific seed, external·human EEG outcome과 외부
메시지는 여전히 실행할 수 없다.

Implementation A의 첫 bundle 시도는 write-once publication 전에 focused child의 negative
test 3개가 실행 맥락별 오류 문구 차이로 실패해 종료됐다(`141 passed, 3 failed`). 정식
`pytest_focused_v3` capability는 mutation role이 아니므로 production 호출은 모두 거부됐고,
bundle/artifact/seed는 만들어지지 않았다. Ordinary-process test가 process-local capability를
명시적으로 비우고 fresh-child test가 non-fresh governed role의 거부도 확인하도록 고쳤다.
이 correction을 포함한 새 clean A에서 focused test 전체를 다시 관측하기 전에는 retry하지
않는다.

그 새 A의 focused 144개는 통과했지만 두 번째 bundle 시도도 publication 전 timestamp
validation에서 멈췄다. Runner의 `.000Z` 출력과 governance의 strict seconds-only `...SSZ`
schema가 불일치했다. Timestamp 생성은 이제 하나의 helper에서 strict RFC3339 seconds UTC만
입출력하고, prior보다 최소 1초 뒤를 보장하며 최대 연도 overflow를 fail-closed한다. Bundle,
test evidence, authorization과 claim이 모두 이 helper만 쓰고 final governance parser 직접
호환 시험을 포함한 runner 45개가 통과했다. 이때도 artifact와 seed는 생성되지 않았다.

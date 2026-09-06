# Metadata-assisted low-calibration SSVEP — V3 pre-outcome design

상태: **개발 계약 선언 / hash-pinned implementation bundle 전에는 nonreserved synthetic도
미실행 / scientific lockbox·외부 EEG·wearable EEG outcome 미승인**

기준 후보는 `metadata-calibration-efficiency-v3`, 결정 ID는 `DEC-20260906-024`다.
기계 판독 기준원은
`configs/analysis/metadata_calibration_efficiency_v3.yaml`, synthetic 기준원은
`configs/analysis/metadata_calibration_v3_synthetic.yaml`, V2와의 경계는
`configs/governance/metadata_calibration_v3_preoutcome_amendment.json`이다.

## 결론부터 말하면

연구목표는 바꾸지 않는다.

> 처음 보는 사용자가 정답이 붙은 SSVEP calibration EEG를 아주 조금만 제공할 때,
> EEG 자체에서 얻은 품질 단서 `Q` 외에 query 전에 측정한 전극 interface와 impedance
> `M`을 알면 필요한 labeled calibration block을 실제로 줄일 수 있는가?

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
development에서 완전 grid로 검사하고, 모든 gate를 통과하는 조합이
없으면 V3를 종료한다. 통과 조합 중 정해진 tie-break로 하나만 고른 뒤 scientific
lockbox 전에 별도 `selected-method-freeze`로 다시 고정한다. 현재 두 YAML은 scientific
실행계약이 아니라 selection 계약이다. 사전 정의된 selector 결과를 한 번 채우는 것은 V4가
아니지만 operator·DGP·threshold·grid·rank rule을 바꾸면 V4와 새 evidence가 필요하다.

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

Synthetic PASS는 구현과 메커니즘의 필요조건일 뿐 사람 EEG의 M 효과가 아니다. 하나라도
실패하면 정확한 V3 후보를 종료하며 threshold, DGP 또는 lambda를 같은 결과에 맞춰 바꾸지
않는다. 바꾸려면 V4와 새 미관측 evidence가 필요하다.

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

따라서 broad literature search는 현재 포화에 가깝다. 다음 정보가 큰 단계는 논문 수를 더
늘리는 것보다 (1) 이 계약대로 pure-DAG 코드와 canary를 완성하고, (2) nonreserved synthetic
development에서 한 exact 후보를 살리거나 종료하며, (3) 직접 metadata-bearing 독립 cohort의
접근권한을 얻는 것이다.

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

Development bundle은 자신이 묶는 git commit 안에 둘 수 없으므로
`/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/development-v1/` 아래 외부 write-once
receipt로 만들고, 이미 clean한 implementation commit/tree를 묶는다. 현재 허용된 것은
코드·문서·unit test와 이 clean implementation/development bundle을 만드는
작업뿐이다. Root seed 20260909의 nonreserved development도 exact bundle validator가 생기기
전에는 실행하지 않는다. Historical-pulse canary도 state engine과 selected freeze 뒤에만
실행한다. 새 scientific synthetic lockbox, BETA/Dong/Choi outcome, wearable source·held
outcome, 외부 저자에게 실제 메시지 전송은 아직 허용되지 않았다. 새 future beacon도
선택하지 않았다.

Development bundle, context reference, development result는 같은 development root의
`development-bundle.json`, `context-reference.json`, `development-result.json`으로 고정한다.
Selected freeze는 repository의
`configs/governance/metadata_calibration_v3_selected_method_freeze.json`, full test evidence는
canary root의 `test-evidence.json`이다.

Scientific execution authorization의 trust root는 repository에 고정한 workspace-owner
RSA-4096 공개키(fingerprint `SHA256:tm6CDH5eVtjTKNqBUwrBYwbq5RhZ48wo1QjP9c+mR+g`)다.
임의 runtime key는 거부한다. 향후 exact attempt manifest가 생긴 뒤 target 전에 대응 개인키로
RSA-PSS/SHA-256 detached 서명을 받아야 하며, 현재 대화의 일반 승인은 그 target-bound 서명을 대신하지 않는다.

## 구현 준비 상태 (2026-09-07 implementation A 직전)

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

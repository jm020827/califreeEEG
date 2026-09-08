# Metadata가 보정 학습을 바꾸는가 — 문헌·기존 구현 재검토

2026-09-08. 상태: **설계 검토 완료, 단일 후보 제안. 새 구현·사람 효능 실험은 미실행.**
기준 코드 `63238bccfda2245703e35cc6f7d05fef9a07568e`.
기존 종료 실험·원본 결과·held60 경계는 변경하지 않는다.

## 1. 결론부터 쉽게

연구목표는 바꾸지 않는다. **새 사용자가 적은 SSVEP 보정 예시로 쓸 만한 정확도에 도달하게 하는 것**이다.
이번 질문은 다음과 같다.

> EEG만으로 추정한 채널 상태에, 보정 전에 측정한 전극 임피던스를 더하면
> 적은 예시로 학습하는 공간필터를 더 잘 만들 수 있는가?

공간필터는 여러 전극의 신호를 섞어 분류에 쓸 신호를 만드는 가중치다.
예시가 적으면 우연히 반복된 잡음에도 큰 가중치를 줄 수 있다.
제안은 metadata로 답을 직접 예측하는 것이 아니라, **어느 채널의 가중치를 크게 학습하는 데
더 조심해야 하는지**를 정하는 작은 보조 정보를 주자는 것이다.
임피던스가 높으면 무조건 나쁘다는 규칙은 쓰지 않는다. 관계가 없을 때 생기는 우연한 변화·
추가 적합 효과와 실제 pairing 이득을 구분해야 한다.

직전 실험은 FULL/drop-one 모델 중 고르는 방식이었다. 새 후보는 모든 보정 예시를 유지한 채
공간필터를 다시 학습한다. 따라서 기존 후보에 없던 예측도 만들 수 있다.
다만 **V1과 실제 context-template는 이미 metadata를 학습단계에 사용했고,
V2에서도 그런 삽입을 설계했다. V3/V4는 주로 보정 예측의 반영량을 조절했다.**
‘선택에서 학습으로 처음 이동한다’는 설명은 잘못이다. 이번 차이는 아래의 구체적인 삽입 위치다.

## 2. 이미 무엇을 했나

| 이전 방식 | M이 바꾼 부분 | 확인된 범위와 결론 |
|---|---|---|
| [V1](metadata_calibration_efficiency_design.md) | 고정 embedding의 정밀도와 support posterior, query 예측 정밀도 | [실제 source39](metadata_calibration_efficiency_results.md) QM−Q eAUC −0.004843. 올바른 숫자 pairing 이득 미확립 |
| [V2](metadata_calibration_efficiency_v2_design.md) | M affinity로 prototype/pseudocount·파형 평균과 신뢰도 조절 | [V11 인프라 중단](metadata_calibration_efficiency_v2_results.md). 과학적 음성 결과로 세면 안 됨 |
| [V3](metadata_calibration_efficiency_v3_design.md) | Q support posterior의 전체 반영량 조절 | [합성 개발](metadata_calibration_efficiency_v3_results.md) no-go. 실제 사람 효능 결과 아님 |
| [V4 pilot](metadata_calibration_efficiency_v4_pilot.md) | block별 posterior 반영량, 동일 총신뢰도 대조 | [합성 pilot](metadata_calibration_efficiency_v4_pilot_results.md) AQ 미확립. 쉬운 k0의 비용 ceiling 문제도 있음 |
| [Context-template](context_template_source39_v1_results.md) | Q×M 가중치로 실제 support 파형 평균을 변경 | 실제 source39 AQ 미확립, M 증분 미확립. 이것도 이미 학습단계 M |
| [Native subset / known-zero](native_subset_known_zero_source39_results.md) | 만들어진 eTRCA 후보 중 선택 | Q의 개선과 달리 M의 추가 비용 절감 미확립 |
| 이번 제안 | TRCA 고유벡터를 구하는 목적함수의 **채널별 규제 방향** | 설계 가설. 구현·효능·신규성 확립 아님 |

V1의 정밀도 조절과 이번 제안은 넓게 보면 같은 신뢰도 prior 계열이다.
연구공간의 기존 `technique:817de79ac188e467`에도 regularized covariance operator라는 추론이 있었다.
따라서 새 이름을 붙여 이미 검토한 아이디어를 새 발견으로 세지 않는다.
직전 [headroom](native_subset_headroom_cold_r1_results.md)은 후보 선택의 사후 상한이지,
이 새 필터의 상한이나 M 정보의 증거가 아니다.

## 3. 문헌이 실제로 말하는 것

### 전극 임피던스: 사용할 이유는 있지만, 단순 규칙의 근거는 약하다

[Tautan 등, BIODEVICES 2014](https://www.scitepress.org/PublishedPapers/2014/47387/47387.pdf)을
방법·임피던스 분석·한계까지 선택 정독했다. 6명 남성, active front end,
1,024 Hz 전류를 주입한 연속 임피던스 측정, Cz/Pz의 4 Hz SSVEP 등으로 우리 조건과 다르다.
같은 전극 종류 내에서 임피던스와 품질의 뚜렷한 관계를 보고하지 않았다.
하지만 심한 artifact와 포화 신호 제외, 작은 표본, 측정부위·자극의 한계가 있어
‘metadata는 무용하다’의 증거도 아니다. 특히 **전극 종류 간 차이를 같은 종류 내 숫자 효과로
옮겨 해석하면 안 된다.** 위치: PDF pp.3–4 방법, pp.8–10 Figure4·분석·논의.

우리가 사용하는 [Wearable 원논문, Zhu 등 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7916479/)도
참가자별 10 blocks×8 channels 평균 임피던스로 상관을 분석했다.
Wet/dry 각각의 정확도 상관은 유의하지 않았고, wet−dry 차이에서는 FBTRCA의 약한 상관이 있었다.
이는 Q를 통제한 block/channel별 보정학습 효과를 검정한 것이 아니다.
위치: §3.5·Figure9; 이번에는 기존 검증된 읽기 기록을 재사용했다.
Wearable의 실제 amplifier 입력 임피던스·측정 주파수는 여전히 확인되지 않았다.
따라서 M을 물리적 잡음 분산으로 환산하지 않고 **조건부 통계 정보**로만 취급한다.

### Signal processing: 채널별 규제는 이미 있는 좋은 출발점이다

[Lotte & Guan, TBME 2011](https://personal.ntu.edu.sg/ctguan/Publications/2011_Fabien_IEEE_TBME.pdf)은
공간필터 목적함수에 `wᵀKw`라는 벌점을 넣고, 채널별로 다른 벌점을 주는 weighted Tikhonov를 다룬다.
K는 다른 사람의 CSP 필터에서 얻으며, 임피던스가 아니다.
17명 MI 비교에서 isotropic TRCSP와 weighted 방법의 평균은 각각 79.3%, 79.4%였다.
가중형이 무가중형보다 유의하게 좋았다는 증거로 이 작은 차이를 쓰면 안 된다.
저자 post-hoc에서는 CSP 대비 유의한 우위가 TRCSP에만 확인됐다.
위치: PDF p.2 §III.B, pp.4–5 §IV.B.3, p.6 TableII·§V.C.

따라서 가져올 것은 **명확한 학습 삽입 위치와 공정한 regularized baseline**이다.
MI의 분산 구분을 SSVEP 반복 재현성 학습에 옮기는 것은 우리의 가설이지 논문의 결론이 아니다.
‘최초 spatial regularization’이나 ‘최초 low-calibration’을 주장하지 않는다.

### ML: 불확실성 모델이 좋아져도 분류가 좋아지는 것은 아니다

[Faithful Heteroscedastic Regression, AISTATS 2023](https://proceedings.mlr.press/v206/stirn23a.html)은
평균·분산을 함께 학습할 때 평균 예측이 나빠질 수 있음을 다루고, 특정 최적화 방식에서
동등한 mean-only 모델 대비 평균 정확도를 보존하는 성질을 제시한다.
[Seitzer 등, ICLR 2022](https://arxiv.org/abs/2203.09168)도 Gaussian likelihood 최적화의 실패를 분석한다.
이번 읽기 범위는 두 논문의 공식 초록이다. 우리 TRCA에 해당 보장이 적용된다고 말하지 않는다.
설계상 교훈은 Q를 먼저 학습·고정하고, 고정된 예측 잔차에 M의 추가 정보를 시험하자는 것이다.

[Covariate-Powered Empirical Bayes, 2019](https://arxiv.org/abs/1906.01611)는
보조 covariate로 noisy estimate를 개선하는 조건부 추정의 인접 사례다. 공식 초록 수준의 참고이며,
EEG acquisition M의 효과나 우리의 채널 벌점을 정당화하는 직접 증거는 아니다.
LLM이나 큰 hypernetwork를 도입해야 할 근거는 이번 검토에서 얻지 못했다.

## 4. 단일 후보: 동일 총량의 Q / Q+M 채널 규제

Q는 EEG에서 얻은 품질 특징을 뜻한다. 아래 `C`는 그 Q와 다른, TRCA의 support covariance다.
Native 구현은 `S w = λ C w`를 풀어 필터를 얻는다. 제안은 다음 한 곳만 바꾼다.

```text
Q arm:    S w = λ [C + γ τ R_Q ] w
Q+M arm:  S w = λ [C + γ τ R_QM] w

τ = trace(C) / d,  d = 8
R는 양의 대각행렬, trace(R) = d
```

양쪽은 동일한 support, S, C, γ, τ, preprocessing, template 평균, filter-bank와 scoring을 쓴다.
**규제 총량은 같고 채널별 배분만 달라진다.** 동일 trace가 동일 유효 자유도를 보장하지는 않는다.
`C`는 순수 잡음 covariance가 아니며, 8개 임피던스 숫자로 물리적 8×8 잡음 covariance를
복원한다고 주장하지 않는다. 대각 prior는 상관된 공통 잡음까지 모델링하지 못한다.

- `R_Q`: EEG의 채널별 진폭/대역 품질/재현성 특징과 공통 interface·protocol·순서·mask로 추정.
  Q를 단순 identity나 부실한 baseline으로 두지 않는다.
- `R_QM`: 위 Q prior에 **support 전에 측정한 숫자 임피던스**의 작은 잔차만 추가.
  Query M은 사용하지 않아 이번 가설을 ‘보정 학습 지원’에 한정한다.
- M의 증분 계수 부호를 미리 양수로 강제하지 않는다. Q 고정 뒤 M 잔차를 추정하고,
  intercept 또는 Q-only 추가 적합이 이득의 원인인지 별도 대조한다.
- 예를 들어 `r_QM,c ∝ r_Q,c exp(δc)`, `|δc|≤a` 뒤 trace를 정규화한다.
  이 경우 최종 비율의 보장은 `[exp(−2a), exp(2a)]`이지 `[exp(−a), exp(a)]`가 아니다.
  `a`, γ 선택 규칙과 수치 floor는 사람 결과 전에 고정해야 한다.
- 완전한 M 결측은 Q 계산 결과를 그대로 반환한다. 부분 결측의 mask는 양쪽 공통이며,
  채널별 결측 처리·정규화에 의한 다른 채널 영향도 구현 명세에 포함한다.
- `trace(C)=0`, 비유한 값, 고유공간 퇴화는 명시적으로 오류 또는 사전 fallback 처리한다.
  γτ가 0이면 양의 R만으로 positive definiteness가 생기지 않는다.

Pinned toolbox `3344bd199daf78888e364d9db00ae7d8128d2b5f`의
[`_trca_U_2`와 ETRCA.fit](https://github.com/pikipity/SSVEP-Analysis-Toolbox/blob/3344bd199daf78888e364d9db00ae7d8128d2b5f/SSVEPAnalysisToolbox/algorithms/trca.py),
[`eigvec`](https://github.com/pikipity/SSVEP-Analysis-Toolbox/blob/3344bd199daf78888e364d9db00ae7d8128d2b5f/SSVEPAnalysisToolbox/algorithms/utils.py)을 확인했다.
Native S 계산·C의 centering을 그대로 유지해야 한다. 기존 함수를 몰래 고치는 것이 아니라 별도 pure operator로 비교한다.
필터 정규화는 양쪽 모두 해당 regularized denominator를 사용하도록 제안하며,
ensemble correlation에서 class-filter 상대 크기와 부호/centering의 영향을 검증한다.
γ=0에서 native와 일치하는지 확인하는 것은 호환성 검사이지 γ>0의 유효성 증명이 아니다.

### 채널 크기만 바꾸는 것은 왜 피하나

가역적인 공통 채널 변환 D를 support·template·query에 모두 적용한 무규제 TRCA에서는
`S′=DSDᵀ`, `C′=DCDᵀ`, `w′=D⁻ᵀw`가 되어 필터 출력이 상쇄될 수 있다.
이 항등식과 인공 배열 검산은 ‘임피던스 역수로 채널을 곱하면 개선’이라는 직관이 충분하지 않음을 보여준다.
모든 native pipeline의 부호·퇴화·ensemble normalization까지 무조건 byte-identical하다는 정리는 아니다.
또한 k1에서 정규화된 단일 support 가중치는 상쇄된다.
새 후보의 primary budget은 k3/k5이며 k0 anchor는 그대로다. k1의 cross-trial S는 정보를 주지 않으므로
이번 방법을 one-shot 해법으로 소개하지 않는다.

## 5. 어떤 정보를 학습시킬 것인가

첫 설계 후보는 **앞서 얻은 support로 뒤쪽 반복 EEG의 불일치를 예측**하는 것이다.
Source 학습 사람의 support prefix에서 Q와 M을 만들고,
별도의 뒤쪽 동일 class 반복과 support template 사이 차이로 repeat-disagreement target을 만든다.
Target 계산에 쓰는 반복은 입력 Q·정규화·prior 구성과 분리한다.
평가 사람의 뒤쪽 반복은 prior나 모델 선택에 넣지 않는다.

이는 접촉 잡음의 정답이 아니다. 시선, 신경 반응, 위상·잠복기·진폭 변화도 포함한다.
따라서 아래 세 연결을 별도로 보고해야 한다.

```text
숫자 M이 Q 밖의 반복 불일치를 예측
    → 그 prior로 학습한 필터가 실제 분류를 개선
    → 같은 실용 성능에 필요한 target labels가 감소
```

앞 단계가 성공해도 뒤 단계는 실패할 수 있다. 앞 단계의 좋은 fitting score만으로 넘어가지 않는다.
Proxy와 M aggregation을 바꿔가며 가장 좋은 결과를 고르는 것도 금지한다.
Support M은 class-blind한 채널 벡터로 집계하되, 평균이 block 변동을 지우는 위험이 있다.
정확한 proxy 식·시간 분리·집계 방식은 **다음 인공 구현 명세에서 하나로 고정할 미해결 항목**이다.
이 문서는 아직 실행 가능한 preregistration이 아니다.

## 6. 공정한 비교와 성공·중단 기준

| 비교 | 답하려는 질문 |
|---|---|
| 기존 native FULL / isotropic regularization | 그냥 regularization을 추가해서 좋아진 것인가? |
| 충분한 EEG Q prior / 동일 γ·trace Q+M | 수치 M이 Q 밖의 추가 정보를 주는가? |
| 같은 2단계 fitting 예산의 Q-only residual | 추가 학습 기회가 Q의 오차만 고친 것인가? |
| 차원·구조·fit 예산이 같은 SHAM_REFIT | 올바른 M pairing이 필요한가? |
| 고정 학습 후 M permutation / stale / missing | 배포 시 pairing·시점·결측에 의존하는가? |

SHAM_REFIT과 학습 후 permutation은 다른 질문이다. 단순 shuffle를 유효한 conditional
randomization test 또는 완전한 information/capacity matching이라고 부르지 않는다.
Interface·순서·mask는 공통이고, pairing 교란의 층·seed·횟수를 먼저 명시한다.

후속 개발평가를 설계할 경우 기존 source39의 참가자별 3fold(26 fit /13 evaluation)를 유지하는 것이
출발점이다. 내부 모델 선택도 참가자 단위로 분리하고, class/channel/window를 독립 참가자로 세지 않는다.
같은 M packet이 여러 class와 window에 복제되므로 참가자·interface·block 학습 질량을 명시한다.
반복 사용한 source39 결과는 개발 결과다. CI도 독립 확증으로 해석하지 않는다.

- 분류 primary는 사전 고정한 8조건 동등가중 k3의 QM−Q이며, k5·개인별 harm·각 조건을 모두 공개한다.
- 비용은 k3=36labels, k5=60labels를 유지한다. 기존 grid 최초80% 도달과 QM3 대 Q5를 함께 본다.
  동률·미도달·비단조를 보존하며 평균 이득만으로 24labels 절감을 주장하지 않는다.
- 독립 확인에서 보정 절감을 주장하려면 사전 실용효과/비열등폭·정밀도·실제 도달 기준이 필요하다.
  이전 참고폭 `1/60`을 임상적·사용자 검증된 차이로 승격하지 않는다. 현재 검정력도 보장되지 않는다.
- Q가 먼저 A0 평균을 이겨야 M을 검정할 수 있다는 옛 필수 gate는 되살리지 않는다.
  반대로 QM이 Q만 이기고 A0/실용성·harm를 충족하지 못하면 유용한 보정 해법의 확립도 아니다.
- 같은 후보의 paired-M 증분이 불확실하거나 없으면 **이 후보에서 미확립**으로 종료한다.
  논문·proxy·window·subgroup를 다시 골라 양성이 나올 때까지 이어가지 않는다.
  물리 기작이나 M 일반 무용성까지 반증한 것으로 해석하지도 않는다.

## 7. 다음 작업과 데이터 병목

당장 더 많은 외부 데이터가 필수는 아니다. 현재 Wearable 자료는 같은 측정환경의 좁은 개발 질문을
시작할 수 있다. 다른 EEG dataset이 많아져도 같은 EEG와 짝지어진 acquisition M이 없으면
이 질문의 독립 M 반복 표본은 늘지 않는다. 다른 장비·기관 재현에는 외부 paired-M가 별도로 필요하다.

다음은 **사람 데이터 없이 끝낼 유한한 engineering 단계**다.

1. Proxy/특징/집계/γ·잔차 bound/결측/학습 graph를 하나로 명세한다.
2. Pure 배열 operator와 대조군을 구현하고 γ=0 호환성·동일 trace·단위변환·결측·퇴화를 시험한다.
3. 사전 seed의 작은 인공 suite 하나로 확인한다: M이 Q 밖 정보를 갖는 경우, 독립 M,
   Q가 이미 충분한 경우, 접촉 잡음 대신 위상·신경 변화가 불일치의 원인인 경우.
   양성 합성 결과는 만들어 넣은 관계를 읽는 sensitivity일 뿐 실제 EEG 효과가 아니다.
4. 그 뒤에만 source39-only 입력/export·자원·출력·동결·감사 계약을 별도로 만든다.

현재 저장된 후보 correlation/features만으로는 새 공간필터를 학습할 수 없다.
S/C, template, 새 필터로 query 점수를 재구성할 통계와 proxy용 반복 자료가 필요하다.
충분통계 export가 가능한지 먼저 설계하고, 불가능한 부분만 source-only raw 접근 범위를 명시해야 한다.
이는 **새 외부 데이터 요청과 다른 로컬 입력 준비 작업**이며 이번 차례에는 하지 않았다.
Held60·retired S1–S3·옛 종료 runner의 재개는 포함하지 않는다.

## 8. 이번 검토의 증거·한계

`academic-research`의 core/foundation/adjacent/contrary/implementation 5개 discovery를 수행했다.
넓은 검색은 중복·무관 문헌도 반환했고, 특히 empirical-Bayes 검색 결과는 강한 seed가 아니어서
공식 원문으로 별도 확인했다. OpenAlex429, PMC CAPTCHA와 publisher 상세페이지403은 우회하지 않았다.
검색 cutoff는 기존 연구공간의 2026-09-04를 유지했다. 최신 문헌의 완전한 조사가 아니다.
더 검색하기보다 정확한 학습기작·누수·대조군을 정의하는 것이 현재의 중요한 병목이다.

새 공개 PDF 2편의 선택 정독과 그림/표 render를 기록했다. PDF 전용 skill은 없어 일반 추출·렌더 도구를
사용했고 fulltext-complete라고 표기하지 않았다. PDF와 dated card는 연구공간에 보존한다.

| 원문 | PDF SHA-256 | 실제 읽기 범위 |
|---|---|---|
| Tautan2014 | `75722b205a61827cfa4a660b970add7ac1410a06f5a429726762577b829e6a54` | PDF1–4,8–10; Figure4가 있는8쪽 render |
| Lotte2011 | `6cf303ce0db07f2ca2f30054418033bc71499bdbda766cfd0dbd46a4725da2ad` | PDF1–7의 관련 절; TableII의6쪽 render |

기존 구현 이력·대수와 전극 물리 근거를 읽기 전용 subagent로 나눠 검토했다.
Root도 seed20260908의 8차원 인공 S/C로 가역 변환 항등식, 양의 규제, 동일 trace,
정규화 후 잔차 bound, 채널 permutation, γ=0, zero-C를 검산했다.
가역 변환 필터 차이 최대 `6.11e−16`, 양쪽 추가 trace 차이 `3.55e−15`였고,
비등방 규제로 최상위 필터 방향이 달라짐을 확인했다. Native 전체 pipeline 시험이나
분류 효능·신호 잡음비 개선 검증은 아니다. 두 읽기 전용 검토자의 이력·물리 해석 지적도 반영했다.
`coordinate-worktree-changes`에 따라 main만 문서·연구공간을 썼으며 기존29worktrees를 보존했다.
새 환경 설치·human artifact 읽기·M packet·raw EEG·fit·성능 outcome·held 접근·외부 데이터 요청은 0이다.
문헌 PDF 다운로드는 사람 실험 데이터 추가와 구분한다. 새 효능 결과는 없으며,
이번 완료물은 **연구목표를 유지하면서 다음 후보를 왜·어떻게 반증할지 정한 설계 검토**다.

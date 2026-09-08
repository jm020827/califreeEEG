# 분류 목표로 학습하는 Q + metadata 채널 규제 — 새 후보 설계

2026-09-08 · **DESIGN_ONLY: 설계·구현 명세 완료, 새 학습·사람 실험 미실행.**
[고정 선택 JSON](../configs/analysis/task_aligned_trca_shape_v1_design.json)과
[구현 명세](task_aligned_trca_shape_v1_implementation.md)를 함께 읽는다.
이 문서는 종료된 이전 후보를 재개하는 실행 허가가 아니다. 구현 검증과 별도 실행 manifest가 남는다.

## 1. 쉽게 말하면

목표는 그대로다. **새 사용자가 EEG 정답 예시를 적게 제공해도 SSVEP를 잘 분류하게 하는 것**이다.
임피던스는 이를 도울 수 있는 사전 측정 정보이지, 연구목표 자체가 아니다.

기존 후보는 ‘어느 채널의 EEG가 다음 반복에서 덜 일관될까’를 예측하고 그 예측을 분류기에 넣었다.
그 중간 예측이 좋아져도 정답 분류가 좋아진다는 보장은 없다. 실제 이전 후보는 추가 이득을 입증하지 못했다.
새 후보는 **학습용 사람의 실제 분류 오차를 줄이는 방향으로 채널 배분을 학습**한다.

| 비교 | 답하려는 질문 |
|---|---|
| Q → QM | EEG에서 알 수 있는 정보에 숫자 임피던스를 더하면 새 사람에게 도움이 되는가? |
| Q → Q2 | metadata 때문이 아니라 단순히 추가 학습했기 때문인가? |
| QM ↔ SHAM_REFIT | 같은 임피던스 분포만으로 충분한가, 그 사람과의 올바른 짝이 중요한가? |
| FULL/A0 ↔ 새 방법 | 약해진 Q만 이기는 것이 아니라 원래 분류기와 비교해도 쓸 만한가? |
| QM 3블록 ↔ Q 5블록 | 36개 정답 예시로 60개를 준 방법에 가까워지는가? |

**Q는 먼저 학습한 뒤 고정한다. QM은 Q를 다시 학습하지 않고 작은 metadata 잔차만 배운다.**
따라서 비교 대상은 ‘고정 Q 위의 제한된 M 추가 효과’다. 모든 metadata 모델의 가능성이나 최적 성능을 묻지 않는다.

## 2. 이미 아는 사실과 아직 모르는 것

- [이전 실제 후보](metadata_prior_source39_v1_results.md)는 QM3−Q3 약 −0.00668%p,
  k5 차이0, 312개 사람×조건의 관측 보정 단계가 모두 같아 종료했다. 이 결과를 바꾸지 않는다.
- [후속 기하 진단](trca_support_geometry_v1_results.md)은 metadata 이전의 공통 규제만으로도
  필터 방향이 중앙값 약44.4°/42.3° 바뀌었음을 보였다. 이것이 정확도 손실의 원인이라는 증명은 아니다.
- 이번의 학습 목표·규제 상한·정규화 변경은 **새 개발 후보의 설계 판단**이다.
  이전 실패를 한 요인으로 인과 분해한 실험이 아니며, 작은 규제가 성공한다는 증거도 아니다.
- [기존 문헌 검토](metadata_learning_covariance_design_review.md)의 regularized spatial filtering은
  설계 동기다. 이미 V1/context-template도 학습에 M을 넣었으므로 최초 metadata 학습/최초 few-shot을 주장하지 않는다.

## 3. 사람·정답·metadata 경계

기존 개발39명, dry/wet × N=125/188/250/500의8조건을 모두 유지한다. 250Hz에서
0.5/.752/1/2초다. 한 블록은12개 자극의 정답 예시 각1개다. k=3/5만 비교하며 one-shot은 아니다.

| 참가자 역할 | 앞 k블록 EEG·정답·M | block5의 EEG·정답 | blocks6–9 |
|---|---|---|---|
| Inner fitting | 공간필터·특징 생성 | Q와 잔차의 분류 학습에 사용 | 접근 금지 |
| Inner validation | 그 사람의 필터 생성만 | 모든 arm 평가 로그, λ 선택은 Q만, gradient 금지 | 접근 금지 |
| Outer evaluation | 새 사람의 필터 생성만 | **접근 금지** | 전체 outer 모델 동결 뒤 최종48query |

블록 번호는0부터다. 3블록은0–2, 5블록은0–4다. Source block5는 proxy가 아니라
**offline 분류 학습 정답**이다. 이 정답을 사용한 사람은 그 fit의 독립 평가 사람이 아니다.
실제 새 사용자에게는 block5 정답을 요구하지 않으므로 평가 사용자의 보정 비용은36/60개다.
다만 source 학습비용과 실험의 시간 간격을 지우거나 실시간 절감으로 바꾸어 말하지 않는다.

39명을 정렬 ID 순위 modulo3으로26fit/13eval에 나눈다. 각26명 안에서 같은 방식으로
17/18fit, 8/9validation의 inner3fold를 다시 만든다. 한 사람의 모든 인터페이스·시간창·예산이 함께 이동한다.
모든 outer 모델을 동결하기 전 최종 query를 열지 않는다. 다른 fold에서 source였던 사람이라는 이유로
현재 평가 역할의 block5를 사용하지 않는다. Shared artifact가 아니라 **역할 제한 reader/API**로 경계를 구현한다.

현재39명은 이미 반복해 본 개발자료다. Nested CV도 연구자의 과거 결과 노출을 없애지 못한다.
Held60은 계속 범위 밖이며, 양성이 나와도 자동 개봉하지 않는다.

## 4. 입력과 작은 학습기

Q15개는 support에서만 만든다: 상대 채널 power, off-reference power fraction, 반복 불일치,
log k, 관측 M 비율, headband order/period, 채널 onehot8.
공통 missingness와 order는 Q/QM 모두에 준다. **Q 함수는 숫자 M을 인자로 받지 않는다.**
기존 `support_q`는 내부 검증에서 실제 M을 읽으므로 그대로 쓰지 않는다.

M은 앞 k블록의 채널별 `log1p(impedance_kΩ)` 관측 평균을 관측 채널 사이에서 중심화한 값과
관측 population SD의2개다. 0은 유효한 측정값, NaN만 결측이다. Query/future M은 없다.
Q2에는 Q의 고정2개 성분인 log off-reference fraction와 repeat disagreement를 준다.

Fit 참가자만으로 특징별 평균·population SD를 구한다. SD<1e−12이면1을 사용하고 clipping은 하지 않는다.
Q는 모든 fit rows, M/Q2 잔차는 공통 available rows를 사용한다. M 관측이 없는 채널의 잔차는0이다.
모든 rows는 참가자·조건·예산·band·channel의 고정 완전 grid다. Available fit rows가 하나도 없으면
M/Q2 모두 scaler=(0,1), 계수=0인 기록을 남겨 Q로 돌아가며 구조적 정보 부재로 분류한다.
이때도 정해진200step 기록을 남기되 masked residual gradient와 계수는0이다.
Q2도 같은 availability를 적용한다. Numeric M을 없애더라도 Q의 원래 missingness 정보는 유지한다.

한 global Q head는15계수+절편=16개, 각 잔차 head는2계수+절편=3개다.
인터페이스/시간창/예산/band마다 다른 모델을 고르지 않는다. 채널·class에 공유한다.
Class ID를 head 입력으로 주지 않는다. Support 정답은 class 정렬·reference 특징·S/template 생성에 사용하고,
source block5 정답은 분류 loss/inner 선택에만 사용한다. Tanh 덕분에 order/log k 같은 공통 항도 채널별 출력 변화에 영향을 줄 수 있지만,
그 conditioning이 실제 유효하다는 뜻은 아니다. 작은 공유 모델의 underfitting 가능성을 한계로 보고한다.

## 5. 규제 총량과 채널 배분을 분리

채널수 d=8, `a=log(2)/2`로 고정한다. 표준화 입력을 z라 하면

```text
ell_Q[c] = 0.8 a tanh(theta · zQ[c] + b)
ell_res[c] = 0.2 a available[c] tanh(beta · zRes[c] + bRes)
R_Q = 8 softmax(ell_Q)
R_QM/Q2/SHAM = 8 softmax(ell_Q + ell_res)
```

모든 양의 규제 arm은 `sum R=8`, `R>0`, `max R/min R≤2`, `max R≤16/9`다.
ISO는R=1이다. Q도 마지막20%의 logit 범위를 잔차에 남겨둔다. 이는 고정 모델 용량 제약이지
metadata가 좋아질 공간을 보장하는 장치는 아니다. Head0은ISO이지 native FULL이 아니다.

Native support의 각 class/band에서 원래 S와 양정 C를 유지하고

```text
eta = 0.1
tau = eta * lambda_min(C) / (16/9)
P = tau diag(R)
B = C + P
```

로 한다. τ는 실제 R이나 M에 의존하지 않아 ISO/Q/Q2/QM/SHAM 간 공통이다.
`trace P=8 tau`도 같다. `diag R≤(16/9)I`와 `lambda_min(C) I≤C`로부터

```text
0 ≤ P ≤ eta C,  C ≤ B ≤ 1.1 C,
0 ≤ (wᵀ P w)/(wᵀ C w) ≤ 0.1  for every nonzero w
```

를 얻는다(Loewner 행렬 순서). **분모 변화가 모든 방향에서 최대10%라는 대수적 상한**이며
필터 각도10%, 정확도 손실0, metadata 이득을 보장하지 않는다. 이전 trace-gamma=.1과 다른 수식이다.
작은 최소고유값 때문에 지나치게 약한 규제가 될 수 있다. 음성 뒤 eta/범위를 넓혀 재시험하지 않는다.

총량0을 Q가 선택하는 gate는 만들지 않는다. 규제가 항상 양수인 Q/QM에서 채널 배분 효과를 묻고,
원래 분류기를 유지하는 것이 더 나은지는 FULL/A0 비교로 따로 평가한다. 실사용 selector를 학습하는 연구는 아니다.

모든 양의-mass arm을 `wᵀ Cw=1`로 정규화하고, 동일 support의 native FULL 필터와
C-inner-product가 양수가 되게 부호를 정한다. 원 native 전처리·ensemble global Pearson·band weight는 유지한다.
η=0 시험은 새 계산을 흉내 내지 않고 기존 프로젝트 `metadata_trca_prior.fit_trca(gamma=0)`와
`score_trca`를 그대로 호출한다. 이 경로와 프로젝트 reference는 exact, 원 저자 toolbox와의
호환성은 기존처럼 correlation 최대오차1e−9와 argmax exact로 구분한다.
공통 C 정규화는 옛 B-normalization 배율 차이를 제거하지만, 방향 변화의 모든 ensemble 효과를 제거하지는 않는다.

## 6. 분류 학습과 유한한 선택 예산

5개 band를 합친 **원래 signed native score**를 양의 band weight 합으로 나누고,
공통 고정 temperature0.1로 나누어12-class cross-entropy를 계산한다. 양의 공통 배율이므로
argmax는 바꾸지 않는다. Temperature를 학습하거나 Q/QM별로 고르지 않는다.
참가자·8조건·2예산·12개 source query를 동일 가중한다.

1. λ∈{1e−4,1e−3,1e−2} 각각에 대해 inner-fit 사람만으로 scaler와 Q를 학습한다.
2. 해당 Q를 동결하고 같은 inner-fit block5 정답으로 Q2/QM/SHAM 잔차를 각각 학습한다.
   SHAM donor도 그 fit partition 안에서 새로 만든다. Outer26에서 학습한 Q를 재사용하지 않는다.
3. 공통 λ는 **Q의 inner-validation CE만** 평균해 선택한다. 1e−12 이내 동률이면 큰 λ다.
   QM/Q2/SHAM의 validation 성능은 기록하지만 선택에 쓰지 않는다.
4. 선택된 λ로 outer26의 scaler/Q를 처음부터 다시 학습하고 동결한 뒤3개 잔차를 학습한다.
   Outer13의 support로 필터를 만들고, 모든 fold 동결 뒤 최종 query를 한 번 평가한다.

각 fit은 float64, 초기값0, full-batch Adam(lr=.01, β=.9/.999, eps=1e−8), 정확히200steps다.
Loss에 `λ × mean(trainable coefficients²)`를 더하며 절편도 포함한다. 조기 종료·gradient clipping·seed 선택은 없다.
같은 학습 정답을 두 단계에 사용하므로 OOF residual regression/DML이나 어떤 문헌의 무손상 보장을 주장하지 않는다.
Outer 평가가 전체 predictive pipeline을 평가한다. 학습/선택 기회는 세 잔차에 동일하다.
이는 **Q가 선택한 공통 λ에서의 M 증분**이며 M에 최적인 λ나 전체 모델군을 평가하는 것이 아니다.
입력 M의 변화 coverage와 R/필터/score의 실제 변화는 별개다. 채널 공통 logit 잔차는
softmax에서 상쇄될 수 있으므로 전달 단계별 변화량을 남긴다.

최대 `3 outer × (3 inner × 3 λ + 1 final) = 30`개 pipeline,
Q+잔차3개의120head fits, 총24,000 optimizer steps다. 이는3λ라는 명시적 선택 예산이지
양성이 나올 때까지 늘리는 실험 횟수가 아니다. 고정 optimizer가 잘 최적화되지 않으면 그 한계로 종료·보고한다.

## 7. 대조군·반증·판정

FULL, ISO, Q, Q2, QM, SHAM_REFIT, PERMUTED, STALE, MISSING의 k3/5와 A0 k0를 모두 보고한다.
새 Q만 강제 통과시킨 뒤 M을 시험하는 Q>A0 입장 gate는 없다.

SHAM은 각 fit/validation/evaluation partition의 interface/order/**앞 k 전체 결측 패턴** 안에서
정렬 ID의 다음 사람으로 M packet을 바꾼다. Fit 밖 donor를 쓰지 않는다. 같은 Q와 동일3param 잔차를
처음부터 학습한다. PERMUTED는 올바른 M으로 학습한 QM에 평가 donor M만 넣으며 재학습하지 않는다.
STALE은 첫 packet 반복, MISSING은 숫자 M만 제거한 exact Q다. Q 입력은 두 진단에서 다시 만들지 않는다.
Singleton·M 특징 불변 사례도 모두 분모에 남긴다. 단일 sham은 정식 conditional-randomization 검정이 아니다.

Primary는39명의 각8조건 평균 **QM3−Q3**(%p)다. 참가자 단위 기술적95% t interval을 보고한다.
반복 개발/겹치는 training folds 때문에 확증적 유의확률이나 모집단 보장으로 읽지 않는다.
기존 기준을 낮추지 않는다: 평균≥1pp, CI하한>0, QM3−Q2/SHAM의 CI하한>0,
sham의 k3 M특징 변화 coverage≥50%(78개 사람×interface)를 함께 요구한다.

보정량 개발 후보에는 추가로 QM3−Q5의 CI하한>−1pp, QM3평균≥80%,
QM3−FULL3와 QM3−A0의 CI하한 각각>−1pp를 요구한다. 참가자 평균 QM3−Q3가 나빠진 사람 수가
좋아진 사람 수보다 많지 않아야 한다. 마지막 조건은 개발용 harm 점검이지 개인 무손상 보장이 아니다.
모든 참가자의 정수 정답 차이·도움/동률/손해·최대 손실을 별도로 공개한다.

관측 k0/3/5에서80%는48query 중39개 이상 정답이다. A0는 Q/QM 공통 k0다.
둘 다 도달한 cell의 pooled label savings가 양수이고 새도달≥도달상실이어야 한다.
전체312cell에서 둘다도달/새도달/상실/둘다미도달을 분리하며 미도달에 가짜60/84label 비용을 넣지 않는다.
둘다도달0개이면 비용 평균은null, gate실패다. 이는 **관측 grid상의 비용 비교**이지
배포 stopping rule·모집단80% 보장·1/2/4shot의 최소 calibration·wall-clock 절감은 아니다.

종료 순서는 validity failure → 구조적으로 잔차가 전달되지 않음 → metadata increment 미확립 →
분류 increment만 확인 → 개발 보정량 후보다. 정확한 enum은JSON에 둔다.
Argmax가 안 바뀌어도 R/필터/score가 바뀌면 구조적 무작동이 아니라 효용 미확립이다.
수치 오류를 missing-M fallback이나 일부 사람 제외로 숨기지 않는다.
모든 분기는 후보 종료이며 새 eta/feature/window/seed로 자동 재시작하지 않는다.

## 8. 이번 완료와 다음 순서

이번에는 저장된 결과 문서·코드·기존 문헌 카드를 검토하고 설계와 구현 계약만 작성했다.
새 사람 artifact·numeric M·query·held60 접근, 새 학습/성능 측정, 환경 설치, 외부 요청은0이다.
**수식의 타당성 검토는 구현의 gradient·수치·분류 검증을 대신하지 않는다.**

다음은 (1) mask-only Q/역할 제한 API 및 연산자 구현 → (2) 고정 인공 gradient·native·누수 검사 →
(3) 독립 auditor/실행 비용 측정 → (4) 정확한 코드·입력 hash가 있는 별도 source39 실행 계약 순서다.
이 설계 JSON은 executable plan이 아니며 옛 runner에 넘기지 않는다.

`academic-research`는 기존 부정 결과/기하 진단/문헌의 적용 한계를 분리하고,
더 많은 검색 대신 지금 설계를 바꿀 수 있는 고유미분 구현 근거만 좁게 확인하도록 사용했다.
`coordinate-worktree-changes`에 따라 main이 계약·문서·SQLite 단독 writer,3명은 읽기 전용
수학/누수/반증 검토였다. 서로 의존하는 설계 문서라 writing lane을 나누지 않았다.
Base bfe5703/main, 기존32worktree 보존, 신규0, 설계 예산64MiB 대 가용약236GiB다.
실험 결과가 아니므로 새 metadata 효능 claim은 없다.

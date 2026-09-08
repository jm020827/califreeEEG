# Metadata prior — 검증 설계 수리, M-blind engineering v1

2026-09-08, baseline `76649af094e2d54dfcb9928d4766f3d8a83981ff`.
사용자 ‘응 계속하라’에 따른 다음 구현 단계다. [이전 합성 결과](metadata_trca_prior_synthetic_v1_results.md)는
종료 상태로 보존한다. 이것은 이전 실패를 성공으로 바꾸는 재분석이나 새 M 효능 실험이 아니다.

## 이번에 해결할 문제

1. **Q의 약한 적합과 M의 추가 정보를 구분한다.** 같은 ridge를 잔차에 다시 맞추면 같은 EEG 정보로도
   규제가 완화된다. Q2 이득은 새로운 정보의 존재를 뜻하지 않는다.
2. **Proxy의 전체 크기와 채널 간 모양을 구분한다.** 모든 채널에 같은 log offset을 더해 줄어든
   예측 오차는 trace-normalized prior에서 사라질 수 있다. 단, base/residual clipping 경계에서는
   offset이 모양까지 바꿀 수 있으므로 실제 prior의 centered log도 별도로 검사한다.
3. **필터→점수→결정 경로를 직접 검사한다.** 신호 전체를 쉽게 맞히는 simulation만으로는 충분하지 않다.
   결과를 원하는 방향으로 만들기 위해 잡음·seed를 탐색하지 않고, 식으로 구성한 인공 배열로 경로를 시험한다.

## 구현 전 고정한 범위

실제 EEG, 실제 M, 과거 human outcome, held60, 기존 v1 generator/suite의 새 실행은 없다.
기존 operator와 v1 코드/config/result는 변경하지 않는다. 새 기능은 기존 실제 runner에 연결하지 않는다.
M 모델 적합·효능 비교·모델 선택·추가 데이터 요청도 이번 범위가 아니다.

Main 단독 writer가 아래 세 기능을 순차 구현하고 읽기 전용 두 reviewer가 검토한다.
공유 계약과 tightly coupled 작은 helper/test이므로 새 writing lane/worktree를 추가하지 않는다.
기존30worktrees 보존. 가용 약262GiB, 신규 테스트·출력 예산1GiB, 별도 mktemp 경로.
환경 설치·공유 환경 수정·push·cleanup은 없다.

### A. Q-only 학습: 참가자 분리 nested selection

입력은 이미 추출한 support-only `q[participant,budget,band,channel,feature]`,
별도 source-repeat target `y[participant,budget,band,channel]`, 고유 정수 participant IDs다.
외부 평가 참가자의 arrays/labels를 받는 인자는 없다. 실제 data adapter나 feature extractor는 만들지 않는다.
이 인터페이스 자체가 호출자가 잘못 넣은 사람 자료/label 누수를 탐지한다는 뜻은 아니다.

- 표현2개: `local`과 `context=[local, 같은 participant/budget/band의 채널 평균]`.
  원 Q와 평균이 있으면 채널 중심화도 선형으로 표현할 수 있다. 미래/query나 다른 참가자를 평균내지 않는다.
  이는 표현의 한 제한을 줄일 뿐 oracle sufficiency/최적 Q를 보장하지 않는다.
- Mean-loss ridge 후보는 `(1, .1, .01, .001, .0001, 0)`로 고정한다.
  Center/scale은 매 fit의 training rows만 사용한다. Intercept는 unpenalized, 상수 SD는1.
  Alpha0은 최소제곱 minimum-norm 해이며 일반화 우월성을 가정하지 않는다.
- Participant ID 정렬 순위 modulo3로 fold를 만들며, 그 사람의 두 budget·모든 채널을 함께 이동한다.
  각 후보의 validation MSE는 참가자별 평균 후 전체 참가자 동일 가중으로 집계한다.
  최소 loss 선택, 절대1e-12 이내 tie는 local 우선 및 큰 alpha 우선이다.
- 최소6 training participants. 각 outer OOF fold의 Q 표현/alpha 선택에도 그 fold 참가자의
  target/features를 쓰지 않는 inner3-fold selection을 수행한다. OOF 잔차와 모든 split/선택 기록을 남긴다.
  마지막 Q 모델은 전체 source training 안에서 다시3-fold 선택 후 전체 source에 적합한다.
- 이 OOF는 **후속 residual 학습용 training nuisance prediction**이다. 이것을 최종 M 검증으로 쓰지 않는다.
  최종 source-fit Q와 작은 OOF-fit Q의 잔차 분포 차이는 남는다. M/SHAM/Q2 비교를 새로 한다면
  같은 residual 규칙·별도 untouched outer 평가·고정된 실제 비용 endpoint가 필요하다.

### B. Proxy decomposition

마지막 channel축에 대해 `MSE = (mean(pred−target))² + mean(center(pred−target)²)`를 기록한다.
두 항은 각각 common-level error와 channel-shape error다. 이는 항등식이지 물리적 잡음 분해가 아니다.
`CLR(prior)=log(prior)−mean(log(prior))`로 실제 normalized penalty의 모양도 볼 수 있게 한다.
v1 raw MSE endpoint를 소급 대체하지 않으며, 이들 값에서 정확도/보정절감으로의 연결을 주장하지 않는다.

### C. 식으로 구성한 M-blind sensitivity fixture

8sample zero-mean orthonormal Hadamard time vectors `u,v,p,q`,3repeats의 계수 `(1,-1,0)`를 쓴다.
`b²=(.03,.0325)`, `a²=1−2b²/3`. Class0 채널은 `a1*u+b1*c*p`, `a2*v+b2*c*q`;
class1은 u/v만 교환한다. 두 class 모두 `C=3I`, `S=diag(6−6b²)`여서 gamma0도 top tie가 없다.
고정 prior `(1.1,.9)`, gamma `.1`, residual `(-.2,+.2)`를 쓰면 선택되는 채널이 바뀌어야 한다.
Query `[u,u]`, `[v,v]`에서 baseline 예측 `[1,0]`이 residual 적용 후 `[0,1]`로 바뀌는지를 검사한다.
이 query에는 연구용 정답을 부여하지 않는다. 어느 prior가 더 좋다는 결과가 아니라 기계적 전달성 검사다.

필수 검사는 S/C 식 일치1e-12, 동일 penalty trace1e-12, score 변화>1e-8, 정확한 두 prediction switch,
gamma0 prior 불변, uniform delta의 prior 불변1e-12, all-missing exact fallback이다.
실패하면 코드/식 오류를 설명하고 수정한다. 여러 seed/noise/gamma를 탐색해 성공 fixture를 고르지 않는다.
통과해도 실제 SSVEP 난이도나 작은 M residual의 평균 효능을 입증하지 않는다.

보조 score 진단은 baseline top-two margin `m`과 전체 class 점수 변화의 최대값 `epsilon`을 기록한다.
`m > 2*epsilon`이면 baseline 승자의 감소와 경쟁자의 증가를 합쳐도 margin을 닫지 못하므로
선택이 유지된다. 역은 성립하지 않고 동점/등호에서는 보증하지 않는다.
이는 engineering 진단 추가이며 새 효능 성공조건이 아니다. v1에는 score vectors가 저장되지 않아
이 식으로 당시 모든 불변 예측의 원인을 입증했다고 주장할 수 없다.

## 다음 효능 설계의 경계

이번에 새 realistic generator·효능 seed·임계값을 고르지 않는다. 다음 별도 설계에서는
Q-only pilot의 모든 사전 지정 조건을 보고하고 M outcome을 숨긴 채 난이도·결정 민감도를 확인해야 한다.
실제-like 분류 성능의 적정 범위는 아직 미정이다. Synthetic sample 수를 늘리는 것만으로 이를 해결했다고 하지 않는다.
난이도 선택 자료와 최종 M 평가 자료를 분리하고, Q/QM/SHAM에 동일한 학습·선택 기회를 부여해야 한다.
현재 단계가 통과하더라도 source EEG export/held60 개봉을 자동 실행하지 않는다.

## 근거와 판정 원칙

기존 academic-research landscape/claim `0d4807a8824492a5`, gap `6a8254f849b36b2f`를 재사용했다.
새 논문 검색보다 현재 코드의 대수와 누수 검사를 먼저 해결한다. 추가 검색/PDF0이며 최신 문헌을 새로 확인했다는 뜻이 아니다.
아래 식은 로컬 코드에서 도출한 설명이다. 표준화된 Gram `A=Z'Z/n`, `b=Z'(y−mean(y))/n`, lambda>0일 때
`beta_Q=(A+lambda I)^−1 b`, 같은 입력 잔차 refit은 `beta_2=lambda(A+lambda I)^−2 b`다.
Q+Q2의 hat eigenvalue는 `s`에서 `2s−s²`로 바뀐다. 배포 시 ±.2 clipping 뒤에는 이 정확한 등가가 깨진다.
따라서 ‘Q2가 더 좋았다’는 사실만으로 오류나 데이터 누수를 단정하지 않는다.

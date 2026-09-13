# Recorded-event M → one-shot template shrinkage: 생성 기작 계약 v1

2026-09-13. 별도 생성 단계이며 직전256행 입력 관문(fit0)은 이미 종료했다.
원 목표는 외부 acquisition metadata가 Q/common 정보 이상으로 저보정 SSVEP에
도움을 주는가이다. 실제 사람 학습 전에 새 후보 **1개**의 작동 경로를 검증한다.
이번 생성 성공은 실제 강한 Q 우위·일반화·정확도·보정 절감 증거가 아니다.

## 가설과 연산

Support의 기록 marker가 nominal frame-grid에서 벗어나는 규칙성이 support
공간/harmonic template의 반복 오차를 Q 이상으로 예측하면, source prior에 대한
shrinkage를 더 잘 정할 수 있다. M을 물리적 jitter라고 해석하지 않는다.

Timestamp z → Δz → r=Δz−τ round(Δz/τ), τ=1000/60ms. M은 r의 median-centered
MAD와 lag1 상관 두 개. |r|<1e−9ms는 수치적0으로 처리하고, 상수 상관은0이다.
이 허용오차는 float 연산 안정화이며 실제 장비 분해능 추정이 아니다. 평균 주기,
frame-count 열·class/frequency·절대 시각·순서를 M에 넣지 않는다.

PSD trace1 support T와 source prior P에 A=(1−λ)T+λP를 적용한다. k1에서도 T≠P면
행렬과 score가 바뀐다. 반복 표적 T*에 대한 source oracle은
λ*=clip(Re< P−T,T*−T >F / ||P−T||²F,0,1), 분모<=1e−12면0이다.
실제 후보의 T*는 source 참가자의 별도 repetition만 사용할 예정이다. 이번에는
아래에서 직접 구성하는 행렬이며 사람 source를 읽지 않는다.

## 고정 생성 세계와 정확한 예산

- Network/raw/새 participant/held60/외부 요청/유료0. Root 단독 코드/기록 쓰기,
  읽기전용 기작 agent가 사전·사후 검토한다. 기존 실패 결과는 불변.
- 시도1,3worlds×4arms=**12 ridge fits**, train32/eval16 episodes, randomdraw0,
  seed label20260913, sweep/recovery/retry0. Unit tests는 전체12fit 실험을 호출하지
  않는다. 생성 입력/대수 함수만 검사하며 추가 unit ridge fit도0.
- Deadline2026-09-13T12:10:00Z, actualrun<=30wall초/CPU30초/AS2GiB,
  singleBLASthread, stdout/stderr 각각32KiB, report32KiB, 직접 새 artifact1MiB.
  실제12fit 전에 코드/계약을 commit하고 고정 단회 디렉터리에 STARTED/terminal 보존.
  각fit의STARTED/COMPLETE 영수증을 별도 작은 파일에 남겨 중간실패도 횟수를 잃지
  않는다. 모델 scaler/beta/intercept 및 eval prediction을 report에 저장하여 재fit
  없이 독립 수치 검산한다. 이 추가 영수증은 기존32KiB/파일·1MiB전체 상한 안이다.
- T=diag(1,0),P=I/2. λ*=index bit0; T*=(1−λ*)T+λ*P. Q의2개 특징과
  Q2의 추가2개 EEG 특징은 모두0인 **통제된 장난감 세계**다. 실제 강한Q 설계가 아니다.
- Positive: M state=bit0, Null: state=bit1, Schedule-only: state=0.
  16event intervals. Δz_i=τ n_i+r_i; state0 r=0, state1 r=alternating(−2,+2)ms.
  n_i는 양의 정수 [4,4,5,4,3] 반복 또는 [2,3] 반복; schedule-only에서는
  이 두 schedule만 episode별로 바뀐다. Cumulative timestamp를 구성한 뒤 위 실제
  extractor로 M을 얻는다. 잔차를 바로 learner에 주는 shortcut은 사용하지 않는다.
- 동일 변동 r에 integer frame schedule/절대 timestamp offset만 바꿔도 M이 같아야
  한다(허용오차1e−9). Schedule-only는 수치 floor 후 M=0이어야 한다.

## 비교군·표준화·SHAM

Q=2zero columns, Q2=Q+2zero columns, QM=Q+2M, SHAM=Q+동시 교환된2M.
동일 ridge penalty0.1, unpenalized intercept, source/train-only population std,
std<1e−9인 column은 scale1. 예측 λ는[0,1]clip. 단일 multi-output이 아니라
이번 toy1output/oneharmonic으로 총12fits다. Q와Q2 모두 같은공통 prior/scorer 사용.

SHAM donor는8episode group 내부의 고정3-bit cycle:
donor bit0=input bit1, donor bit1=input bit2, donor bit2=input bit0.
Positive/Null 각각 train(λ*,M-state)2×2 count가8씩이고 M vector multiset이
보존되는지 assert한다. Train에서만 교환, eval에는 원래 M. Schedule constant M의
2×2balance는 해당하지 않는다. Metadata 두 차원을 따로 섞지 않는다.

## 평가·성공·중단

Loss는 episodes 평균 ||A−T*||²F (행렬 원소별 평균이 아님). 상수λ=.5의 정확한
balanced baseline은0.125다. Source/eval 상태가 반복되는 결정론적 fixture이므로
독립 일반화/OOD/통계적 유의성 검증으로 해석하지 않는다.

Required: positive QM loss<.01, Q/Q2/SHAM은0.125±1e−10. Null와schedule-only의
모든 arm은0.125±1e−10. 이 값은 설계된 능력 관문이지 실제 효능 중단 기준이 아니다.
같은 learned positive QM으로 M-only forward2회: C=I,H=diag(1,0)인 유효 Gram
query의 s=tr(AH)/tr(AC)=1−λ/2 변화>0.4. T=P 대조2회는 metric/score 동일.
H=C 대조2회는 score1 불변. 추가 forwards6회, 추가 fits0.

불변성/수치/사전기준 실패시 실패로 종료하고 feature/penalty를 구제 튜닝하지 않는다.
성공하면 이 연산을 유지하되 실제 학습으로 자동 효능 승격하지 않는다. 다음 실제
source 계약에는 강한 EEG-Q/Q2, known-class/schedule/eventcount 조건, support-onlyM,
participant-exclusive prior/반복표적, queryDIN 격리, baseline과 실제 획득 비용을
명세한다. Sensor setup 미확인이면 순보정시간 감소를 주장하지 않는다.

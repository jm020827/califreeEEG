# 다음 단계: metadata 추가 학습 전 SSVEP 신호 연결 진단

2026-09-13, v2 종료 후 작성한 **실행 전 초안**. 현재 구현·원자료 실행은 하지 않았다.
v2의 80-fit 예산은 소진했고 결과는 `RETIRE_UNDER_FROZEN_PROTOCOL`로 보존한다.
새 실험을 잘 나올 때까지 반복하는 계획이 아니다. 다음 단계 시작 시 아래 범위를
별도 계약과 코드 hash로 고정한 뒤 단회 진단하고, 결과에 따라 유지·중단을 결정한다.

## 왜 이것부터인가

v2에서 QM뿐 아니라 zero-shot CCA와 두 template baseline 모두 평균 22–27%였다.
이것만으로 버그나 데이터 손상을 단정할 수 없지만, 저보정 개선을 주장하기 전에
기본 신호와 decoder가 연결되는지 확인할 이유가 된다. Metadata 계수와 예측은 실제로
달라졌으므로 “M 입력이 빠져서 생긴 결과”라고 설명하는 것은 맞지 않는다.

Problem signature:
- 표현: 250Hz/256 EEG row, 고정 [1,3)초 창, nominal h1/h2 reference와 실수 PSD.
- 병목: 낮은 검출력이 신호-자극 연결, 고차원 CCA 선택성, 표현 한계 중 무엇과 양립하는가.
- 허용 연산: 개발용 S001a의 event clock 비교와 고정 수치 진단; 새 학습·튜닝 없음.
- 목표: 다음 구현을 바꿀 직접 근거를 얻거나, 근거가 없으면 연결 복원을 보류.
- 자원: 기존 파일 1개, 분석창 15개, source 평가 cohort 재접근 0.
- 피드백: 두 clock의 주파수 차이, reference overlap, spectral energy와 CCA selectivity.
- 실패 모드: 정답 DIN을 decoder에 몰래 넣음; 신호가 없다는 인과 단정; 결과별 parameter rescue.

## 서로 다른 설명과 진단

| 설명 후보 | 고정해서 볼 것 | 해석의 경계 |
|---|---|---|
| nominal Hz와 이벤트 시간축이 안 맞을 수 있음 | 각 main group에서 `f_time=1000/(2 mean Δtimestamp_ms)`, `f_sample=250/(2 mean Δsample)` 및 nominal Hz 차이 | DIN edge가 주기의 절반이라는 source 가정에 의존. 실제 photodiode·neural latency 증명 아님 |
| 시간축 차이가 spectral reference를 약하게 만들 수 있음 | nominal/f_time/f_sample의 고정 quadrature subspace overlap과 해당 EEG projection energy | 세 값만 비교, peak 탐색·최적 주파수 선택 없음. 정답에서 얻은 f는 진단 전용 |
| 256채널/500sample CCA가 비선택적일 수 있음 | 공분산 spectrum/effective rank, nominal CCA 5-score와 margin; 한 개 고정 시간 permutation과 비교 | permutation은 공분산을 보존하지만 spectrum·시간 의존도 바꾼다. phase만 선택적으로 없앤 null이 아님 |
| 실수 PSD가 phase 정보를 버리는 데 한계가 있을 수 있음 | 코드/대수 canary: `B→BR`인 공통 quadrature rotation에서 `BBᵀ` 불변 확인 | 표현의 성질만 증명. 실제 phase 모델이 더 좋다는 효능 근거는 아님 |

위 값은 모두 보고한다. 가장 좋아 보이는 하나만 고르거나 S001 정확도로 ridge·전극·filter·
window를 튜닝하지 않는다. S001은 이미 개발용으로 사용한 사람이지 새 독립 test가 아니다.

## 제안 실행 한도와 정보 경계

- 입력: 기존 `/home/whwovy/data/mamem_i_v1_20260913/development_first.mat` 1개,
  SHA256 `57a72c3fde0ff3bc9aaae10721cd7cb450bda63eb5299a96f41704ea696ad10a` 확인.
  Header와 필요한 `eeg`, `DIN_1`, `samplingRate`를 decode하되 수치 EEG 처리는 기존
  v2 15 main [1,3)초 창의 첫 256 row만 한다. 나머지 EEG 구간/257번째 row 분석 금지.
- M2 추출·regression fit·source prior·classification 학습·새 효능 비교·bootstrap 모두 0.
  S001b, S002–S011 원자료/캐시/정답 재접근 0; held60 0; network/download/request/paid 0.
- Time permutation은 seed 20260914로 500 indices 한 번 생성하여 모든 channel과
  15개 window에 동일하게 적용한다. 반복 seed/검정/최선 null 선택 없음.
- 원자료 reader 1회, 재시도·교체 없음. 2GiB AS/90CPU/120wall, BLAS 1,
  출력 JSON ≤128KiB, 새 extraction 0. 코드는 root 단일 writer, 검토는 read-only.
- Generated unit canary로 clock 변환·reference projection·공분산 보존을 검사한 뒤
  코드·input hash·출력 schema·시작/중단 조건을 먼저 고정한다. 진단 stdout/결과 보존.

## 결과 뒤의 결정

1. Clock/샘플 연결에 확인 가능한 모순이 있으면 source 문서의 단위·row 의미부터 재확인한다.
   주파수·class를 새 데이터에 맞춰 임의 보정하거나 v2를 수정하지 않는다.
2. 공통 reference에서 응답이 보이는데 CCA 선택성이 낮다면, 검증된 montage/기준 구현
   확보 후 별도 baseline 복원 계획을 세운다. 후두 전극 번호를 추측해 적용하지 않는다.
3. 신호 근거가 약하거나 원인이 여전히 식별되지 않으면 이 데이터 경로의 추가 효능 실험을 보류한다.
   Metadata 후보를 더 fit하는 대신 이미 확보한 다른 공개 acquisition-context 경로를
   기존 실패·권한 기록과 대조해 재선별한다. 비공개 자료 요청은 진행하지 않는다.
4. 기본 decoder의 타당성이 확보된 뒤에만 M이 어떤 nuisance parameter를 설명할지
   새 기작 가설을 세운다. 공통 EEG front-end를 모든 Q/Q2/QM/SHAM에 같게 적용하고,
   calibration 시간 측정과 독립 평가 계획을 동결해야 한다. Neural ODE 등 큰 모델을
   추가하는 일은 현 진단을 대체하지 않는다.

S002–S011은 이미 탐색 평가에 사용했으므로 이후 개선 결과를 독립 confirmation이라
부르지 않는다. 공개 잔여 run/다른 dataset을 쓸지도 provenance와 조건 일치부터 검토한다.
같은 사람의 미개봉 c/d/e run도 새 참가자 독립 확인 자료가 되는 것은 아니다.
held60 개봉·사람에게 자료 요청·유료 자원은 이 초안에 포함하지 않으며 별도 승인이다.

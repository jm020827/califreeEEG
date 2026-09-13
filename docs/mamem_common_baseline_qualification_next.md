# 후속: 공통 SSVEP baseline 복원 — 새 M 효능 실험 전에

2026-09-13, S001 단회 신호 진단 후 작성한 **미실행 초안**.
v2 80-fit 종료 판단은 유지한다. 새 gate/Neural ODE/채널·주파수 sweep을 바로 시작하지 않는다.

## 이번 관측이 바꾸는 우선순위

S001에서 nominal↔event reference overlap이 일부 class/harmonic에서 매우 낮았고,
현재 CCA의 ridge effective dimension은 약256이었다. 이것은 다음 두 요소를 공통
baseline 문제로 분리할 이유다. 어느 한 변경이 효과가 있다고 아직 확인한 것은 아니다.

1. 자극 주파수/reference의 정의와 사용 가능한 acquisition 기록의 연결.
2. 짧은 창·고차원 EEG에서의 공간 covariance 정규화와 CCA 선택성.

## 먼저 할 유한 source/implementation 확인

다음 turn 시작 시 범위를 다시 고정한다. 공개 저자/MOABB 구현과 기존 보존 자료에서
MAMEM reference frequency, montage/row mapping, preprocessing, CCA 설정을 확인한다.
제안 예산: 공식 repository tree 조회1회 및 필요한 code/docs 최대8파일·합계512KiB,
별도 공식 문서 최대2건, 총30분; 원 EEG/새fit0. 기존 파일을 먼저 재사용한다.
특정 논문 PDF가 구현 선택을 바꿀 때만 질문을 queue하고 별도 읽기 예산을 먼저 고정한다.
자료가 비공개이거나 확인되지 않으면 그 경로는 보류하며 사람에게 요청하지 않는다.

반드시 확인할 항목:

- DIN-derived label key와 실제 reference Hz를 혼동하지 않는가? Author 코드가 실제로
  어떤 Hz를 classifier에 전달하는가? 그 값은 query 정답에서 온 것 아닌가?
- 전극 번호와 후두 montage 연결에 직접 근거가 있는가? 없으면 posterior 채널을 추측하지 않는다.
- 원 CCA·regularized CCA의 정의, covariance estimator와 전처리·창 길이는 무엇인가?
  자동 shrinkage를 검토하더라도 원 EEG가 그 통계적 가정을 만족한다고 주장하지 않는다.
- Published/MOABB 구현과 우리의 label/window 정책이 다른 부분을 모두 명시한다.

## 그다음 개발 평가의 설계 경계

위 확인 후에만 별도 실행 계약으로 아래 요소의 단일 정의를 확정한다. 이 초안은 실행
권한을 확장하거나 이미 열린10명에 새 parameter sweep을 허용하는 문서가 아니다.

- 초기 개발 대상은 S001a/b만. S001a에서 이용 가능한 support 정보로 설정하고 b의
  15개 query에는 EEG와 모든 후보 reference bank만 제공한다. Query DIN으로 얻은
  주파수를 그 query의 reference로 넣는 진단용 oracle을 deployable decoder로 포장하지 않는다.
- Reference: nominal과 support에서 이용 가능한 event 정보의 고정 변환을 비교할 수는
  있지만, 개인 결과별 최적 frequency/phase를 고르지 않는다. 전체 source에 공통인
  잘못된 nominal 상수를 고치는 효과와 사람별 metadata 추가 가치는 구분한다.
- 공간 처리는 source-backed 단일 baseline으로 사전 고정하며, 이번 개발 비교에서
  별도 공간처리 비교는 하지 않는다. 채널/ridge/filter/window의 격자 탐색은 금지한다.
  새 raw접근·계산·예측 수 예산을 먼저 정한다.
- 먼저 문서와 합성 수치검사로 **단일 공통 baseline**을 고정한다. 그 뒤 최소 비교는
  동일 baseline의 nominal5reference와 S001a 첫support1trial/class에서 계산한
  sample-clock5reference 두 조건이다. Time/sample 선택 sweep을 하지 않으며 모든
  query가 동일한5class bank를 받는다. 이것은 **개발 진단**이고 supervised metadata
  gate 학습이나 독립 효능 확증이 아니다. 공통 baseline을 바꾼 효과는 원 v2와의
  정규화 효과 분리검증으로 주장하지 않는다. 기본 decoder가 회복되지 않으면
  MAMEM 추가 효능 fit은 보류하고 잘 나오는 조건까지 반복 확장하지 않는다.

## 다시 M 학습으로 돌아가기 위한 조건

모든 Q/Q2/QM/SHAM이 동일한 복원된 front-end와 공통 자극 정보를 받아야 한다.
단순한 reference 수정으로 얻는 이득을 학습된 metadata 기여로 세지 않는다. 남은
acquisition-context 차이가 반복 template 오차 또는 adaptation parameter를 왜 설명할지
새 기작 가설·반증 조건을 기록한다. 새 후보 예산과 중단 기준도 결과 전에 고정한다.

S002–S011을 다시 쓰면 development-reuse 탐색임을 표시한다. 같은 사람의 c/d/e도
새 참가자 독립 확인이 아니다. 독립 participant 또는 다른 공개 cohort의 타당성을
확인하기 전에는 확증으로 승격하지 않는다. Calibration trial 수뿐 아니라 실제 수집
시간·setup 비용·query-ready 시간도 측정 가능해야 한다.

held60 개봉·사람 자료 요청·유료 자원 사용은 별도 승인이다. 기존 negative와 이번
진단의 한계를 보존하며, 원래 저보정 SSVEP metadata 학습 연구목표는 바꾸지 않는다.

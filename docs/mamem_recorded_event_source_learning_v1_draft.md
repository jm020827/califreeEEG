# MAMEM recorded-event M: 실제 source 학습 연결 초안

2026-09-13. DRAFT_NOT_EXECUTED. 원 저보정 SSVEP 목표를 유지한다.
[256행 실제 입력](mamem_i_256row_adapter_v1_results.md)과
[12-fit 생성 능력](recorded_event_shrinkage_generated_v1_results.md)을 다음 한 후보의
실제 학습으로 연결하기 위한 초안이다. 아래 숫자는 제안이며 실제파일별 역할/분할/
적격성과 구현·생성 검증을 고정하기 전 사람자료 학습 계약으로 취급하지 않는다.

## 한 후보·한 핵심 가설

같은 class/지원 구간/획득 조건에서 support marker의 frame-grid 잔차 규칙성이
support EEG spatial/harmonic template의 다른 repetition에 대한 오차를 Q 이상으로
예측한다. 그렇다면 source prior와 support template의 혼합량을 개선할 수 있다.
M은 support창의 MAD/lag1 두 특징, k1에서도 행렬을 바꾸는 shrinkage 한 연산이다.
물리적 jitter 복원이나 gyro source-expert routing을 재시도하는 가설이 아니다.

기존 exactv1 PDF p2 자극 설명/p4 Flickering frequencies와 보존 MOABB code
lines94/746을 다시 확인했다. Reference는 **6.66,7.50,8.57,10.00,12.00Hz**,
classID1–5다. MOABB 정수 key는6,7,8,9,11이므로9/11을 referenceHz로 쓰지 않는다.
이는 public code의 vocabulary이지 실제 v1 모든 trial 라벨을 확인한 결과는 아니다.
Session.m은1000/(2×meanΔt), MOABB는 평균과 역수 단계의 정수 나눗셈을 쓴다.
임의 nearest-label이나 고정 순서로 미등록 key를 보충하지 않는다. 두 legacy code의
짧은 group 처리에서 frequency 추가와 range/start 추가가 분리돼 있으므로 새 parser는
group별 label/window를 함께 검증해야 하며 zip으로 조용히 정렬시키지 않는다.
근거 원문은[배포 설명서](https://ndownloader.figshare.com/files/3687771),
[MOABB](https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/ssvep_mamem.py),
[저자 Session.m](https://github.com/MAMEM/eeg-processing-toolbox/blob/5a03abe2a6a874e9adaceea29a52c2fce35d8a03/%2Beegtoolkit/%2Butil/%40Session/Session.m)이며 새 network 없이 기존 보존본을 읽었다.

## 구현을 함께 동결해야 할 부분

1. **파일→trial/label→split.** 확보한 exactv1 filename namespace를 사용한다.
   S001 전체는 개발용으로 제외. 알려진5target의 nominal frequency와 source의
   label grouping 규칙을 기존 원문과 연결하고, code의 floor된 class ID와 실제
   sine/cosine reference frequency를 혼동하지 않는다. 각 class 반복수·session
   chronology·제외 파일 이력은 실제 inventory/schema 검증 없이 가정하지 않는다.
   Query DIN은 label 작성 인터페이스에만 있고 predictor에는 존재하지 않아야 한다.
2. **공통 EEG.** 앞256행, 고정 시간창, window centering/수치 scale 정책을 동일하게
   적용한다. 생성 offset/global-scale 검증을 포함하되 원본 단위/reference를 확인한
   척하지 않는다. Geometry 미검증이므로 좌표/후두부 mapping 주장은 하지 않는다.
   각 nominal class/harmonic의 complex sine/cosine 계수에서 trace-normalized PSD를
   만들고 zero-energy 처리를 고정한다. 모든 baseline에 같은 처리·reference 제공.
3. **Q/Q2.** 실제 Q는 harmonic SNR/reference residual, 공간 covariance 안정성,
   한 trial 내부 split-window template 불일치 등 EEG 신뢰도 정보를 포함해야 한다.
   Q2에는 같은2개 추가 차원을 EEG 안정성 정보로 채운다. 생성 단계의 zero Q를
   실제 후보로 옮기지 않는다. 추출식·regularization·source-only scaler를 동결한다.
4. **Source 목표.** P는 pseudo-target 참가자를 제외한 source prior. T는 그 사람의
   support, T*는 그 사람의 다른 repetition만 사용한다. Target 참가자의 query는
   gate fitting/선택/중단 기준 조정에 쓰지 않는다. Fold 안에서 prior/normalizer/
   oracle 표적 생성 순서를 검사한다. Unknown common phase 변화만으로 PSD를
   바꾸겠다는 가정은 금지하며 실제 반복 오차의 조건부 예측력을 먼저 확인한다.
5. **M/confound/SHAM.** Class/nominal schedule·window/eventcount·공통 보정을 모든
   arm에 제공한다. 실제 timestamp 양자화/edge 때문에 class 정보가 M에 남을 수
   있으므로 source-only 조건화와 class/조건 내 donor mapping을 동결한다. 단순
   전역 M shuffle은 쓰지 않는다. M이 조건 내 상수이거나 donor 변화가 없으면 이를
   기록하고 후보의 정보 가정을 다시 평가한다. Query M은 입력 권한 밖이다.

## 제안하는 유한 전체 실행 범위

후보1/M2/harmonic수와 ridge penalty 고정, sweep0. S001 제외10명이 사전 적격성을
충족할 경우 outerLOSO10 × k={1,2} × Q/Q2/QM/SHAM4 = 최대80multi-output gatefits를
제안한다. Prior/template 추정·source pseudo-episodes·class별 반복 및 평가 횟수도
별도로 세어 실제 manifest에 넣어야 한다. 적격성 실패 후 유리한 참가자를 골라
대체하지 않는다. 목적은 데이터 제공을 다시 요청하는 것이 아니라 확보 자료에서
한 후보를 공정하게 판단하는 것이다.

Zero-shot reference baseline, target-onlyλ=0, source-onlyλ=1도 같은 query/시간창에서
함께 평가한다. Primary는 k1/QM의 Q·Q2·SHAM 대비 차이와 Q의k2를 대체할 수 있는지다.
기존 사전 margin과 participant-level paired 불확실성·harm를 적용한다. 기존 margin
값은 원 계약을 정확히 확인해 새 manifest에 명시해야 하며 결과 뒤 정하지 않는다.

Calibration 비용은 실제 acquired-prefix와 적응/대기/버린 trial 및 sensor setup을
포함한다. Setup 시간이 없으면 손익분기 비용은 보고할 수 있지만 순보정시간 감소가
입증됐다고 쓰지 않는다. Single-stimulus 자료의 결과를 동시 다중표적 online BCI나
ITR로 확대하지 않는다. Zero-shot이 이미 목표를 충족하면 보정 절감 기여도 제한된다.

## 중단

Generated actuation의 실제 경로 연결 실패, 정보권한/label 누출, source 조건 내
M 상수, source 반복 오차의 추가 예측력 미확립, 또는 공정한 외부 평가에서 이득과
보정량 감소 미확립을 각각 구분해 기록한다. 같은 M에 feature/penalty를 끝없이
추가하지 않는다. Held60/외부 요청/유료는 여전히 별도 승인이다.

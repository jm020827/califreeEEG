# S001 신호 진단 결과: reference 불일치와 CCA 해석의 두 단서

2026-09-13. 원 저보정 SSVEP 목표를 유지한다. 직전 MAMEM v2의 80-fit 부정 결과는
그대로다. 이번에는 개발용 S001a 한 파일의 고정15창을 **학습 없이** 진단했다.
결론은 “metadata가 도움이 된다”가 아니라, **추가 metadata 학습 전에 공통 baseline의
주파수 reference와 CCA 정규화를 점검할 근거가 생겼다**는 것이다.

## 1. 이벤트의 timestamp와 sample 표기는 서로 부합하지만 nominal과는 다르다

같은 DIN 이벤트의 간격으로 `1000/(2 mean Δtimestamp_ms)`와
`250/(2 mean Δsample)`를 계산했다. Nominal 값은 기존 inferred class reference이며,
아래는 class별 main3창의 group-average 계산값을 평균한 것이다.

| Nominal reference (Hz) | Timestamp 간격 기준 (Hz) | Sample 간격 기준 (Hz) |
|---:|---:|---:|
| 6.66 | 6.4988 | 6.4983 |
| 7.50 | 7.3463 | 7.3467 |
| 8.57 | 8.3151 | 8.3174 |
| 10.00 | 9.6034 | 9.6047 |
| 12.00 | 11.6144 | 11.6144 |

두 event 계산값의 trial별 차이는 최대 절댓값 0.004771Hz였다.
모든15group에서 `max|Δtimestamp−4Δsample|`는 3ms였다. 이는 평균 시간 척도와
국소 증분이 서로 부합한다는 관찰이다. 두 필드의 생성 clock이 독립인지 확인한 것이
아니고, **실제 화면 주파수·edge 의미·절대 지연이 검증된 것도 아니다**.
두 식 모두 주기당2event라는 같은 가정에 의존한다. Constant latency나 sample
index의0/1-based 차이는 차분에서 상쇄되어 이 검사로 판별할 수 없다.

실행 전에 보존된 저자 `Session.m`의 연속식/필드 소비 코드를 재검토했다.
`f_time`은 그 연속식을 따르며 `f_sample`은 추가 일관성 검사다. 실제 hardware 설정을
확인한 것처럼 해석하지 않는다. 이전 첫72event prefix의3ms 결과를 이번15maingroup의
증거로 대신 쓰지 않았고, 이번 값은 별도 고정 진단에서 새로 얻었다.

## 2. 작은 Hz 차이도 지금의 2초 reference에는 큰 차이를 만들 수 있다

2초 sin/cos reference 부분공간의 겹침을 0–1로 계산했다. 1이면 같은 부분공간이다.
이 값은 정확도가 아니라 고정 reference끼리의 대수적 관계다.

| Nominal class | Nominal ↔ timestamp h1 평균 overlap | h2 평균 overlap |
|---|---:|---:|
| 6.66Hz | 0.702 | 0.197 |
| 7.50Hz | 0.725 | 0.235 |
| 8.57Hz | 0.391 | 0.00279 |
| 10.00Hz | 0.0616 | 0.0361 |
| 12.00Hz | 0.0740 | 0.0418 |

예를 들어10Hz reference와 event로 계산한 약9.60Hz reference는 이 창에서 잘 겹치지
않는다. 주파수 차이는 창 안에서 phase가 계속 어긋나는 차이이므로, 단순한 공통 phase
원점 회전과 다르다. 앞 단계에서 nominal 주파수에 잘못 맞춰 투영했다면, 그 뒤의
공간 PSD 혼합 비율만 바꾸는 것으로 놓친 시간 성분을 복구하기 어렵다는 **기작 가설**을
뒷받침하는 단서다. 이것이 v2의 낮은 정확도를 전부 설명한다는 증거는 아니다.

EEG 투영 에너지의 변화도 일률적이지 않았다. 12Hz class의 h1 평균 relative energy는
nominal 0.002566, timestamp-reference 0.010392였지만, 6.66/7.50Hz class h1은
timestamp-reference에서 더 작았다. 세 reference를 모두 보고했으며, 가장 큰 것을
정답별로 골라 새 classifier 정확도를 만들지 않았다. 이 event-derived reference들은
**진단용**이고 query predictor에 제공하지 않았다.

## 3. CCA의 높은 점수는 높은 분류 성능과 다르다

256채널·500sample에서 기존 CCA를 그대로 계산하고, 동일한 시간 순열 한 개를 모든
채널/창에 공통 적용했다. 이때 공간 covariance는 상대 오차 최대8.42e−16으로 보존됐다.

| 진단값, 15창 평균 | 원래 시간 순서 | 고정 시간 순열 |
|---|---:|---:|
| CCA 최대 score (squared correlation) | 0.8296 | 0.6043 |
| CCA 1위−2위 score gap | 0.02526 | 0.01691 |
| inferred class score−다른 class 최대 score | −0.02183 | −0.04089 |

원 신호의 점수가 더 높아 시간 구조의 영향은 관측된다. 하지만 순열에서도 score가
상당히 높고, 원 신호의 평균 signed class margin도 음수다. 따라서 높은 최대 CCA
score만으로 target을 잘 구분한다고 할 수 없다. 이번 진단은 정답률을 계산하지 않았다.

Covariance entropy effective rank는 평균8.77(7.32–10.43)이었으나, 기존 작은 ridge에서
`sum(λ/(λ+alpha))`는 평균255.92(255.91–255.93)였다. 대부분 에너지가 소수 방향에
몰려 있어도 whitening에는 거의256개 방향이 참여한다는 뜻이다. 작은 방향이 전부
잡음이라고 단정하거나 “잡음을 맞춘 것이 원인”으로 확정하지 않는다. 다만 고차원
CCA의 정규화를 점검할 직접적인 수치 근거다.

순열은 spatial covariance와 window쌍의 Gram을 보존하지만 spectrum·시간 의존성도
바꾼다. 모든 trial 공통구조나 phase만 선택적으로 제거한 null이 아니다. 한 개 순열로
p-value나 일반적 null 분포를 주장하지 않는다.

## 실행·검증 범위

- 계약4ab4829, 수식 review를 반영한 최종 producer/auditor99baf32를 실제 진단 전에 고정.
- 13:34:30.676898→13:34:31.860757UTC, 1.183859초, 1attempt/15창/학습0.
- `eeg`, `DIN_1`, `samplingRate`는 한 loadmat 호출로 전체 decode했다. 수치 EEG 처리만
  15main [1,3)초 창·첫256row에 제한했다. Descriptor, row257, 다른 EEG구간은 사용하지 않았다.
- `M_features_computed:false`는 학습용 MAD/lag1 M2 extractor 미호출을 뜻한다.
  위 timestamp/sample 진단 통계는 실제로 계산했다.
- 합성 full-rank 양성/음성, 독립 Gram projection/eigen-whitened CCA, phase rotation,
  excluded poison, manifest/단회/오류 검사를 통과했다. 사전 관련316tests와 새45tests
  통과 후, 최종 통합 재검사317tests도 통과했다(2.85초). 전체repository suite는 미실행.
- 독립 저장 산술 감사는 최대 오차3.33e−16으로 PASS. 재fit/원자료 재열람0이며,
  원 EEG에서 계산한 에너지·공분산 자체를 별도 재구축한 감사는 아니다.
- 별도 read-only protocol audit도6pins·manifest/start/terminal 연결·15창·63,542bytes·
  stdout/stderr0·기존v2산출물hash 보존을 확인했다. OS-level IO 추적이나 peak-memory
  측정은 하지 않았으므로 단일 decode/자원한도는 고정 코드·receipt 근거로 보고한다.

## 다음 결정 — 원 연구목표를 바꾸지 않는다

이 진단은 기본 decoder의 타당성이 확보됐다고 결론내릴 정도의 검증은 아니다.
**MAMEM에서 새 metadata gate의 효능 fit은 보류**하고, 먼저
[공통 baseline 복원 계획](mamem_common_baseline_qualification_next.md)을 진행한다.
관측한 시간 reference와 정규화 문제를 모든 비교군에 공통으로 반영해야 하며, 그 공통
수정으로 좋아진 성능을 metadata 학습의 신규 기여로 세면 안 된다.

그 뒤에도 연구 질문은 “외부 acquisition context가 EEG-Q와 공통 정보 이상으로
학습에 도움을 주어 실제 보정량을 줄이는가”다. 새 후보는 수정된 강한 공통 baseline,
Q/Q2/SHAM, source-only 학습과 별도 검증 계획을 갖춰야 한다. S001은 개발용이고,
이미 관측한 S002–S011의 재평가는 탐색 재사용이다. held60/외부 요청/유료 사용은
이번에도0이며 별도 승인 경계를 유지한다. v2 후보의 부정 결과는 취소하지 않는다.

Artifacts: [원 진단](reports/mamem_signal_validity_v1_run/diagnostic.json),
[manifest](reports/mamem_signal_validity_v1_run/manifest.json),
[terminal](reports/mamem_signal_validity_v1_run/terminal.json),
[독립 감사·요약](reports/mamem_signal_validity_v1_audit.json).

Diagnostic SHA256 `12051d2c67d0780a0e11312b2aa85d7fcb16b2c33d11bb7f0e21a8217a683f4a`.
새 다운로드/데이터 추출/추가 학습/설치/삭제/push0. 기존45worktrees·8untracked 보존.

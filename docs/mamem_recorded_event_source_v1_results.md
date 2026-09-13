# MAMEM recorded-event metadata source-learning v1 결과

2026-09-13. 연구목표는 저보정 SSVEP에서 외부 acquisition metadata가 EEG-Q와
공통 정보 이상으로 도움이 되는지 검증하는 것이다. 이번 시도의 결론은 **효과 없음이
아니라 실제 효능 평가 미실행**이다. 개발용 파일에서 사전 라벨 해석 규칙을 통과하지
못했다. 디스크 공간은 약300GiB로 충분하며 현재 병목은 용량이 아니다.

## 무엇을 구현했나

사람별 한두 개 보정 trial로 만든 주파수·고조파의 공간 PSD template와 다른 사람의
source prior를 섞는다. 다른 source 사람들의 반복 측정만으로 그 혼합량을 학습한다.
외부 metadata는 보정 trial에 해당하는 recorded-event 시간 간격의 frame-grid
residual MAD/lag1 두 값이며, 물리적 display jitter를 확인한 측정치라고 부르지 않는다.

- Q: EEG에서 계산한 신호 품질 + 공통 정보.
- Q2: 같은 입력에 EEG 특징2개를 더 준 비교군.
- QM: Q에 실제 metadata2개를 추가.
- SHAM: Q에 같은 조건에서 다른 source 사람의 metadata 쌍을 넣어 학습.
- 별도로 zero-shot CCA, target-only, source-only baseline을 유지한다.

모든 모델은 같은 query 구간의 모든5개 후보 주파수를 평가한다. Query DIN은 공통
구간 설정과 사후 정답 평가에만 쓰며 predictor에 label/count/order/M을 넘기지 않는다.
Target 사람은 gate/scaler/T*/prior/donor에서 제외하며 pseudo-target prior에서도 그
사람을 추가 제외한다. One-shot은 뒤의 support trial을 사용하지 않는다.

시간창2초/harmonics1·2/ridgeα=.1/80realfit 상한을 사전에 고정했다.
실제 효능을 보려면 QM이 Q/Q2/SHAM을 모두 이기고, one-shot QM이 two-shot Q 대비
비열등하면서 support-prefix가 작아야 한다. Support-prefix 감소는 실제 query-ready
elapsed 감소와 다르다. 남은 a-run과 a→b 전환시간은 측정·제거하지 않았다.

## 실행과 결과

| 단계 | 실제 실행 | 결론 |
|---|---|---|
| 새 모듈·운영 경계 생성 단위검사 | 134 tests PASS | 구현 불변성 검사 통과 |
| 전체 generated pipeline | 4가상 참가자×2k×4arm=32 fits | 완료, 사람 효능 근거 아님 |
| 기존 두 archive 무결성 재검증 | SHA256 모두 일치 | 기존 데이터 유지 |
| 개발용 S001a 단회 확인 | EEG/DIN/samplingRate decode, parser 진입 | 주파수 guardband에서 중단 |
| S001b/실제 S002–S011 준비 | 실행0 | 대체 파일을 열지 않음 |
| 실제 참가자 효능 학습 | 0/80 fits | metadata 이득·보정 감소 미평가 |

Generated 시도는12:23:34.687035–12:23:38.245495UTC에 완료했다. 32개의 고유
FIT_STARTED/FIT_COMPLETE와 학습 입력·scaler·계수·λ·예측을 보존했다. 그 데이터는
쉬운 통합 fixture여서 learned arm들의 정확도가 모두1.0이다. 이는 M의 이득이나
실제 정확도·일반화 성능을 뜻하지 않는다. 이전 positive/null 생성 기작 검사와도 구별한다.
Generated summary 내부의 고정 `10-subject` limitation 문자열은 real protocol 문구이며,
실제 generated 참가자는 FAKE0–FAKE3의4명이다. 원본 artifact는 수정하지 않는다.

독립 생성 수치 감사는 재fit/production import 없이32모델의 source-only scaler와
normal equation을 검산했다. 최대 residual5.204170427930421e-17, 저장 λ320scalars
재계산 오차0,840개 saved-score의 argmax·정확도·요약 산술 일치다. SHAM120rows의
joint-vector/strata multiset,24pseudo-prior 배제 기록, 모든 fold의 유효 change=1을
확인했다. 원 EEG/C/B/T/P가 해당 JSON에 없으므로 원신호에서 oracle/score를 재구성한
감사는 아니다. 별도 protocol 감사도7개 code/contract pin·32fit·중단 경계를 확인했다.

개발 시도는12:23:58.447074–12:24:06.522627UTC에 종료했다. 실패 원인은
`frequency_outside_unique_guardband`이다. `1000/(2*mean(Δtimestamp))`가 nominal
frequency±최근접 간격의1/4인 사전 허용대 안에 유일하게 들어가야 하는 검사다.
이번 오류 영수증에는 실패 group 번호나 실제 추정 주파수는 없으므로, adaptation인지
main인지 또는 왜 범위 밖이 됐는지는 아직 알 수 없다. 데이터 손상이나 metadata
무효를 의미한다고 해석하지 않는다.

코드 실행 순서상 DIN row2/4 전체 numeric scalar 읽기·단조성·23group 확인까지는
통과했다. 해당 group의 label 해석에서 실패해 support M 계산과 EEG window 수치
특징 추출은 모두0이다. EEG/DIN 전체 변수 decode 자체는 이미 했음을 명시한다.
S001a는 기존 pinned MAT를 재사용했다. 이번 새 MAT 추출 bytes는0이며 S001b나
source 참가자 파일을 대신 열지 않았다. 원본 archive와 모든 실패 기록을 보존했다.

## 판단

이번 bounded efficacy 시도는 schema/label stop 조건으로 **중단·보존**한다.
Guardband를 넓히거나 가까운 class를 강제 배정해 같은 결과를 재실행하지 않는다.
후보는 효능상 성공/실패 어느 쪽으로도 판정할 수 없다. 전체 연구목표는 유지하며,
다음 병목은 [개발용 라벨 진단 계획](mamem_recorded_event_label_diagnostic_next.md)이다.
실제 데이터에 대한 성능평가를 시작하려면 먼저 라벨 의미를 분명히 해야 한다.

Source/newnetwork/PDF0, held60/외부발송/유료0. 기존 부정 결과·보호4문서 hash 불변.
전체 repository suite는 미실행이며 이번 관련134tests만 실행했다.

## 재현·협업 기록

- Base fb47c76/main → 계약089caed/481e687 → parser0911332 → featuresc02faac →
  root engine/runner1d11a80. 최종 실행 manifest가 code SHA들을 묶는다.
- Writer들은 별도2worktree, protocol/numeric/runner 감사는 shared repo 읽기전용.
  main만 계약·통합·실행·SQLite를 썼다. 기존41+신규2=43worktree 보존, cleanup0.
- 사전 감사에서 oracle의 tiny denominator 분기, support sample 순서, query 구간
  설명, offline 비용, SHAM 수치적 변화 기준, worker 단회성·입력 binding을 고쳤다.
- 기존8untracked pytest directory와 사용자 변경을 보존했다. Push/설치/환경 복제0.

Artifacts: [manifest](reports/mamem_recorded_event_source_v1_run/manifest.json),
[generated summary](reports/mamem_recorded_event_source_v1_run/generated_summary.json),
[development terminal](reports/mamem_recorded_event_source_v1_run/development_terminal.json),
[실패 traceback](reports/mamem_recorded_event_source_v1_run/io-child_S001a_stderr.txt).
전체 generated 모델은 manifest의 고정 outside-Git cache에 있고 summary에 SHA가 있다.

주요 해시:

```text
manifest 484e17b679401a9cb61d67b1c7c5d54cb93b89978cb973b84afcb8e3bb50e72c
generated summary 0558dd6d15c1d27e4c51ba0962d20b3c169f81d4c82ef138df7750935dec6450
generated full result b8a18934efa4393074aa833f15ae5ca20df15480eb6359c8af7eb793fe6c9284
development terminal 6ab6a462273eab5995f111ba8b00a313a29d745ba83c6d2dd2df8bce03322b34
development failure 181f8199d534a7714c660fc3999696162e626b673addb26513b40270f1d29039
```

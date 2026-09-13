# MAMEM metadata 학습 v2 결과 — 이 후보는 종료

2026-09-13. 원래 목표인 **외부 acquisition metadata로 저보정 SSVEP를 개선하는가**는
유지한다. 이번 한 후보는 사전 고정한 80회 학습을 모두 마쳤으나 metadata의 추가
정확도 이득과 보정량 감소 기준을 통과하지 못했다. 결과에 맞춘 재학습은 하지 않는다.

## 무엇을 시험했나

한 사람의 calibration EEG가 적을 때, 그 사람의 template와 다른 사람들의 평균
template를 얼마씩 섞을지 학습했다. 가설은 “자극 이벤트 기록의 불규칙성이 개인
template의 재현성을 알려주어, EEG 자체만 볼 때보다 혼합 비율을 잘 고를 수 있다”였다.

M은 calibration 구간 이벤트의 frame-grid residual MAD와 lag-1 두 값이다.
실제 화면 지연·물리적 jitter를 검증한 센서값은 아니다. 비교군 모두 class, calibration
수, event count와 분석창 길이 같은 공통 정보를 받는다.

- Q: EEG에서 계산한 품질 특징으로 혼합 비율을 학습.
- Q2: Q에 EEG 특징 두 개를 더함. 입력 수 증가만으로 생기는 이득을 대조.
- QM: Q에 실제 M 두 개를 더함.
- SHAM: 같은 class·k·event count 안에서 다른 source 사람의 M 쌍으로 교환해 학습.
  평가에는 QM과 같은 실제 support M을 받는다.

S001은 개발용으로 제외했다. S002–S011의 a 파일을 support, b 파일을 query로 쓰는
10명 leave-one-person-out 실험이다. 1-shot은 class마다 1개, 2-shot은 class마다 2개로,
각각 총 5개/10개의 labelled support trial이다. 두 조건의 query는 같은 15개다.
각 target의 데이터는 source gate·scaler·prior·repeat target·SHAM donor에서 제외했다.
Source pseudo-target의 prior는 실제 target과 pseudo-target 모두를 제외한다.

## 정확도

10명 평균, 5-class offline task. 단위는 %. 균형 random-choice의 기대 정확도는 20%이며,
아래 결과가 그것을 통계적으로 유의하게 넘었다는 검정을 추가로 한 것은 아니다.

| 방법 | 1-shot | 2-shot |
|---|---:|---:|
| Q: EEG 품질 | 25.33 | 27.33 |
| Q2: EEG 특징 추가 | 25.33 | 26.00 |
| QM: 실제 metadata 추가 | 25.33 | 26.67 |
| SHAM: 교환한 metadata로 학습 | 25.33 | 26.67 |
| 개인 template만 | 22.00 | 23.33 |
| source template만 | 26.67 | 26.67 |
| calibration 없는 CCA | 25.33 | 25.33 |

Zero-shot의 두 열은 같은 예측을 반복 표시한 것이며, calibration을 두 번 한 것이 아니다.
평균 정확도가 같다는 사실만으로 예측이나 학습 계수까지 같다고 추론하지 않는다.
별도 읽기 전용 수치 검토에서는 QM의 1-shot 예측이 Q와 150개 중 3개, SHAM과도 3개
달랐다. 혼합 계수도 달라졌다. 즉 **M은 학습에 들어갔지만 정확도 개선으로 이어지지 않았다**.

사전 고정 participant paired bootstrap 10,000회, seed 20260913. 아래 단위는
**percentage point(pp)**이며, Q2의 이름과 2-shot의 k=2를 구분한다.

| 사전 비교 | 평균 차이 | 95% bootstrap CI | 5pp 초과 악화 인원 | 통과 |
|---|---:|---:|---:|---|
| QM 1-shot − Q 1-shot | 0.00 | [0.00, 0.00] | 0 | 아니오 |
| QM 1-shot − Q2 1-shot | 0.00 | [−2.00, 2.00] | 1 | 아니오 |
| QM 1-shot − SHAM 1-shot | 0.00 | [0.00, 0.00] | 0 | 아니오 |
| QM 1-shot − Q 2-shot | −2.00 | [−6.67, 2.00] | 3 | 아니오 |

첫 세 비교는 평균 ≥1pp이면서 CI 하한 >0이어야 했다. 마지막 비교는 CI 하한 ≥−2pp와
더 적은 support-prefix samples가 필요했다. 평균 −2pp만으로 비열등성 통과라고 하지 않는다.
QM−Q 및 QM−SHAM의 [0,0]은 관측된 10명의 정확도 차이가 모두 0이어서 생긴
경험적 bootstrap 결과이지, 모집단 효과가 정확히 0임을 증명하는 구간이 아니다.

## 보정 부담은 줄었나

1-shot support를 얻는 데 필요한 기록 시작부터의 prefix는 사람별 약 446–458초다.
2-shot보다 짧은 양은 평균 **9.9568초**(9.872–10.040초)다. Class가 묶여 제시되므로
labelled trial 수가 10→5라고 실제 대기시간까지 반으로 줄지는 않는다.

남은 a 파일 구간과 a→b 사이 시간, 추가 metadata setup 비용을 측정하지 않았으므로
**실제 query-ready 시간 절감은 UNKNOWN**이다. 약 10초는 조건부 setup 손익분기 예산일
뿐이고, 성능 유지 조건도 실패했으므로 입증된 절감량으로 보고하지 않는다.
Zero-shot으로 80%에 도달한 사람은 0/10이다. 다른 방법의 낮은 정확도와 함께 볼 때
이 시스템을 실용적 저보정 BCI로 승격할 근거는 없다.

## 실행과 독립 검산

Frozen producer commit `39807d8`; 실행 결과를 보기 전 감사 구현은 `b86684d`까지 고정했다.

| 단계 | UTC 시작 → 종료 | 실제 파일 | gate fit |
|---|---|---:|---:|
| 생성 통합 preflight | 12:58:59.921640 → 12:59:01.826016 | 0 | 0 |
| S001 개발 연결 | 12:59:19.638851 → 12:59:26.686467 | 2 | 0 |
| S002–S011 실험 | 13:00:10.962826 → 13:01:20.750556 | 20 | 80/80 |

Generated는 parser→features→preflight 연결 확인만 했다. 이전 v1의 생성 32-fit 결과와
개발 라벨 실패/실제 0-fit 기록은 별도로 그대로 보존했다. v2는 23개 group을 정렬한 채
MOABB의 문서화된 double-integer-division key→class 관례를 적용한 label 연결 개정이다.
Nominal 주파수와 DIN에서 추론한 class를 구분하며, 독립 정답을 확보한 것은 아니다.

실제 20개 fold의 1,800 scalar source oracle, 80개 고유 gate fit이 완료됐다.
80은 학습 횟수이지 80명의 독립 표본이 아니다. 평가 단위는 10명이다.
SHAM은 모든 fold에서 조건부 교환이 가능했고, 교환 후 의미 있게 바뀐 비율은
1-shot 93.33–100%, 2-shot 82.22–88.89%였다. PSD fallback과 degenerate oracle은 0이다.

[독립 감사](reports/mamem_recorded_event_source_v2_audit.json)는 production engine import나
재fit/solve 없이 저장된 80모델·800λ·900SHAM row·2,100 score argmax entry·bootstrap·비용을
검산해 `PASS_SAVED_ARITHMETIC_AND_ROLES`를 반환했다. Normal-equation residual 최대
4.996e−15, λ 재계산 오차 0이다. 2,100은 k 간 반복되는 zero-shot도 포함한 저장 entry 수다.
원 EEG→PSD, oracle 생성, query score 자체를 독립 재구축한 검증은 아니다.
관련 단위검사 272개 통과(1.17초), 전체 저장소 테스트는 미실행이다.
별도 검토자가 저장 JSON만 읽어 같은 수치와 판정을 확인했다. 절차 검토자도 code pin 9개,
20개 extraction/IO 완료, 고유 80-fit, 개발용 제외와 preflight-before-fit를 확인했다.
시간 순서는 ledger·코드·파일 mtime 근거이며 외부 서명된 실행 증명은 아니다.

## 해석과 다음 단계

판정: `RETIRE_UNDER_FROZEN_PROTOCOL`. Metadata가 학습 입력으로 연결되었어도, 이번
recorded-event M2 + PSD 혼합 후보에서는 Q와 공통 정보 이상으로 분류를 개선하지 못했다.
이는 이 데이터·표현·학습기·조건에서의 부정 결과다. 모든 acquisition metadata의 무효,
SSVEP 원리의 실패, 또는 데이터 손상을 증명하지 않는다. 10명 탐색과 반복 후보 선택의
불확실성이 있으며, 이 cohort는 이후 독립 confirmatory test로 재사용할 수 없다.

다음은 metadata 후보를 추가 fit하는 일이 아니다. 기본 decoder도 22–27% 수준이므로
**개발용 S001에서 입력 시간축·주파수 reference·기본 검출력부터 점검**해야 한다.
가능한 원인은 nominal reference와 event clock의 차이, 고차원 CCA의 비선택성,
공간 PSD 표현이 전달하지 않는 공통 phase 원점 정보 등이며 현재 어느 것도 실증 원인으로
확정하지 않았다. 실수 PSD가 모든 공간 상대 phase나 query energy를 없앤다는 뜻은 아니다.
[후속 진단 초안](mamem_signal_validity_next.md)에 질문·예산·금지사항을 기록했다.
새 신호 정책을 선택하더라도 v2 결과는 그대로 두고, 다음 효능 실험은 별도 버전·새 예산과
검증 cohort 계획을 먼저 고정한다. held60 개봉·외부 요청·유료 사용은 별도 승인이다.

## 보존 위치

- [실행 manifest](reports/mamem_recorded_event_source_v2_run/manifest.json),
  [실제 terminal](reports/mamem_recorded_event_source_v2_run/real_terminal.json),
  [summary](reports/mamem_recorded_event_source_v2_run/real_summary.json),
  [preflight](reports/mamem_recorded_event_source_v2_run/real_preflight.json).
- 원 MAT·feature cache·full model JSON은 `/home/whwovy/data/mamem_recorded_event_source_v2`에
  보존하고 Git에 넣지 않는다. 이 디렉터리는 약 2.9GiB이며 새 다운로드는 없다.
- Full result SHA256: `12e02744f2dca6ff82cce06d4c0f4b6c5596bda944e81b928b2015b5bd674500`.
- Summary SHA256: `c72456c259835b066afa165bcce1fdb78ae813818318a6be7c361edaf7cac3f9`.
- Manifest SHA256: `6c61c6fc3d3cc081d0dc4b9f65529b8cf3ead21befd4e8254703c8720dedef76`.

Root/main 단일 작성자와 읽기 전용 병렬 감사로 마감했다. 앞서 격리 구현한 parser/runner
worktree와 모든 기존 worktree·untracked 파일은 삭제하지 않았다. 여유 약 296GiB,
held60/외부 사람 요청/유료 자원/추가 학습/삭제/설치/push 모두 0이다.

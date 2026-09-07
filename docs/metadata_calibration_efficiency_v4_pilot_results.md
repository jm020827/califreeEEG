# V4 pilot 001 결과 — 실행 완료, AQ_NOT_ESTABLISHED

기준일: 2026-09-07. **24명 합성 pilot을 단회 완료했고 독립 검산은 PASS지만,
사전 AQ utility 기준은 통과하지 못했다.** 특히 A0가 이미 98% 이상이어서 이번
조건은 calibration 감소를 평가하기에 너무 쉬웠다. 이는 metadata 일반의 무용성이나
사람 EEG에서의 실패를 증명하지 않는다. 이 exact pilot은 종료하며 재시도·사후 tuning은 하지 않는다.

버전 구분: 기존 `development-v5`는 **V3 방법의 개발 실행 #5**로, 이미 0/9 eligible,
`DEVELOPMENT_NO_GO`로 종료됐다. 본 문서는 그 재실행이 아닌 **새 V4 방법 pilot 001**이다.
[사전 계획](metadata_calibration_efficiency_v4_pilot.md)과
[고정 설정](../configs/analysis/metadata_calibration_v4_pilot.json)은 변경하지 않았다.

## 쉽게 보는 결과

12개 표적마다 k번 보정하므로 k=1/3/5는 각각 12/36/60 labeled trials다.
모든 방법은 동일한 미래 query 60개/participant를 사용했다. 아래는 participant 평균
balanced accuracy, 단위 %다. Null은 drift와 **동일한 EEG**, 독립적인 metadata다.

| 조건 | 방법 | 보정 0 | 보정 1 | 보정 3 | 보정 5 |
| --- | --- | ---: | ---: | ---: | ---: |
| Stable | A0: 보정 없는 FBCCA | 98.264 | 98.264 | 98.264 | 98.264 |
| Stable | AQ: EEG-only 파형 fusion | 98.264 | 98.403 | 98.403 | 98.403 |
| Stable | AQM-block | 98.264 | 98.403 | 98.403 | 98.403 |
| Stable | AQM-scalar | 98.264 | 98.403 | 98.403 | 98.403 |
| Stable | Pooled 파형 diagnostic | 98.264* | 78.750 | 97.986 | 99.306 |
| Drift | A0 | 98.542 | 98.542 | 98.542 | 98.542 |
| Drift | AQ | 98.542 | 98.611 | 98.681 | 98.750 |
| Drift | AQM-block | 98.542 | 98.542 | 98.681 | 98.681 |
| Drift | AQM-scalar | 98.542 | 98.542 | 98.681 | 98.681 |
| Drift | Pooled 파형 diagnostic | 98.542* | 67.431 | 91.597 | 96.319 |
| Null | A0 | 98.542 | 98.542 | 98.542 | 98.542 |
| Null | AQ | 98.542 | 98.611 | 98.681 | 98.750 |
| Null | AQM-block | 98.542 | 98.542 | 98.681 | 98.681 |
| Null | AQM-scalar | 98.542 | 98.542 | 98.681 | 98.611 |
| Null | Pooled 파형 diagnostic | 98.542* | 67.431 | 91.597 | 96.319 |

\* Pooled k=0도 사전 규칙상 A0 fallback이다. 학습된 template가 없는 상태에서
pooled model이 독자적으로 얻은 정확도가 아니다. Missing-M은 모든 k에서 AQ와 exact 동일하다.

핵심은 세 가지다.

1. **보정 비용 감소를 이 조건에서 증명할 수 없다.** 사전 목표 80%를 모든 24명이
   모든 family에서 이미 k=0에 달성했다. 줄일 비용이 처음부터 0이다. Pooled를 포함한
   first-crossing 결과도 이 fallback 때문에 모두 0이며, 학습의 효과로 해석하지 않는다.
2. **파형 평균 자체는 학습한다.** Stable pooled의 k1→k3→k5가
   78.75→97.99→99.31%다. 따라서 “파형에 학습할 정보가 전혀 없다”는 설명은 맞지 않는다.
   하지만 k5 diagnostic을 사후 채택하거나 이것을 low-calibration metadata 기여로 세지 않는다.
3. **현재 고정 fusion과 metadata 추가효과는 충분하지 않다.** Stable AQ가 A0보다
   맞힌 query는 전체 1,440개 중 2개 더 많을 뿐이다. Drift k3에서도 AQM과 AQ의 실제
   예측 차이는 0이었다. 단지 평균 BA가 같다는 추론이 아니라 저장된 flips를 검산한 결과다.

## 사전 screen과 paired uncertainty

저보정 eAUC는 `BA0/6 + BA1/2 + BA3/3`이다. 표의 차이와 구간은 모두
**percentage points(pp)**로 표시한다. 24 synthetic participants의 paired t 계산이며,
다중검정을 보정한 confirmatory population inference가 아니다.

| 비교 | 평균 차이, pp | One-sided 95% LCB, pp | 판정 |
| --- | ---: | ---: | --- |
| Stable AQ−A0 eAUC | +0.115741 | −0.021441 | FAIL: 평균 +1 pp 미달, LCB≤0 |
| Stable AQ−A0 k1 | +0.138889 | −0.025729 | 비음수 평균 조건만 PASS |
| Drift M-block−AQ eAUC | −0.034722 | −0.094232 | FAIL |
| Drift correct−deranged M, k3 | +0.138889 | −0.001498 | FAIL: 평균 +1 pp 미달, LCB≤0 |
| Drift block−scalar, k3 | 0 | 0 | FAIL: 추가 재배분 효능 없음 |
| Null M-block−AQ, k3 | 0 | 0 | 90% CI [0,0], 사전 BA equivalence PASS |
| Drift M-block−A0, k3 | +0.138889 | −0.025729 | 비음수 평균 조건만 PASS |

Stable AQ eAUC의 two-sided 95% CI는 [−0.049839, +0.281321] pp다.
Drift M-block−AQ eAUC의 구간은 [−0.106551, +0.037106] pp다.
AQ_stable_utility가 false이므로 최종 상태는 **AQ_NOT_ESTABLISHED**이며
`human_unlock=false`다. 다른 screen을 통과한 것처럼 이 이름을 바꾸지 않는다.

## Metadata가 실제로 한 일과 남은 병목

Drift k3에서 올바른 M의 평균 total trust g는 .642981, block weight turnover는
.047737이었다. Deranged M에서는 각각 .345772와 .140948이었다. 따라서 M 연결을
섞는 조작 자체는 no-op이 아니었다. Scalar는 동일한 g를 가지지만 실제 block
재배분은 정의상 0이다. 그럼에도 correct M-block과 scalar의 k3 BA 차이는 0이었다.

Drift k3 correct log probability는 A0 −1.188199, AQ −1.619709,
M-block −1.444784다. M은 AQ의 confidence 손상을 일부 줄였지만 A0보다 나빴다.
또한 null BA equivalence 통과는 모든 확률이나 개인별 위험이 같다는 뜻이 아니다.

확인된 가장 큰 설계 한계는 **ceiling과 무의미해진 80% cost endpoint**다.
이를 사후에 99% target으로 바꾸거나, noise를 키워 같은 pilot을 다시 돌리지 않는다.

추가 원인 가설은 **score scale 불일치**다. 같은 filter-weight 합으로 나누고 같은
softmax temperature를 쓴다고 CCA²와 flattened waveform correlation²가 같은 확률 척도가
되지는 않는다. Stable pooled k5는 99.31%를 맞히면서도 correct log probability가
−2.290748로, A0의 −1.214896보다 낮다. 순위는 배우지만 고정 fusion의 확률은 약하다는
해석과 양립한다. 다만 이는 결과 뒤 진단이며 온도 변경의 효능을 이 데이터에서 입증한 것은 아니다.

## 다음 연구 루프 — 목표 유지, 같은 pilot은 재실행하지 않음

다음 일은 데이터를 더 뽑아 M-positive 결과를 찾는 것이 아니다.

- 별도 development-only 자료에서 **A0 난이도와 calibration 필요성**을 먼저 확인하는
  계획을 고정한다. 이 단계의 시나리오 선택에는 M−AQ outcome을 쓰지 않는다.
- Signal/noise·위상·공간 변동의 범위를 실측/문헌과 연결하고, 원래 목적상 보정이
  필요한 조건과 강한 EEG-only 비교법을 먼저 마련한다. 손상 복원을 새 목표로 만들지 않는다.
- Pooled waveform처럼 실제 k 증가를 활용하는 AQ와 probability scale 정합성을
  support/development 자료만으로 검토한다. Query 정답으로 fusion을 맞추지 않는다.
- 그 뒤 새로운 미관측 evaluation에서 M−AQ, correct−shuffle, equal-trust block−scalar와
  null을 같은 labeled budget으로 비교한다. 이번 pilot의 seed·threshold·결과는 그대로 보존한다.

현재 결과로 실제 사람 data 또는 기존 held/scientific lockbox를 개봉하지 않았다.
새 연구목표가 아니라 **유효한 난이도와 AQ 비교법을 먼저 확보해야 한다는 설계 수정 과제**다.

## 실행·검산·보존 기록

- Start UTC: `2026-09-07T05:07:44.759060+00:00`.
- Clean source commit: `4a78efe052428b60e5b53c6006fd710941258efb`;
  tree: `c8de54d44b9c1f70430bd47d1fef08d9c5138218`.
- New root seed: `10943930911181219160` (64-bit integer; 부동소수점으로 읽으면 반올림될 수 있음).
- CPU 4 workers, BLAS 각 1 thread; Python 3.10.12, NumPy 1.26.4, SciPy 1.15.3.
  GNU time wall time **2.62 s**. 이는 작은 새 pilot의 시간이지 V3 대비 speedup 측정이 아니다.
- 24 participant × 84 = **2,016 metric rows**, **1,728 prediction records**,
  **6,624 individual permutation metric records**를 독립 검산해 PASS.
- 통합 전후 관련 시험 44 passed; 전체 suite **915 passed**, 68 기존 warnings.
- Artifact root: `/home/whwovy/v4-artifacts/metadata-calibration-efficiency-v4/pilot-001`.
  `start.json`: 1,364 bytes, SHA-256
  `b11b3737c9abd5b47f18d167019568f8cf431c45798596f4d161b61e1baac4d4`.
  `result.json`: 12,321,001 bytes, SHA-256
  `96c5e491f0ba410d343e0fbcf9256e7e762718bb66e4d2c44f8743163b27d8ba`.
  두 파일은 exclusive publication, mode 0400, nlink 1이며 수정하지 않았다.
- Git integration은 main이 단독 수행했다. 별도 구현 worktree/branch는 clean recovery source로
  보존한다. 기존 사용자 변경은 없었고 기존 V3 source/governance/artifacts는 그대로다.

이미 소비된 pilot을 다시 실행하지 말고 아래 **read-only 검산 명령**을 사용한다.
결과 문서가 추가된 뒤의 main에서도 recorded historical source/plan을 검산할 수 있다.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_metadata_calibration_v4_pilot.py \
  --plan /home/whwovy/califreeEEG/configs/analysis/metadata_calibration_v4_pilot.json \
  --result /home/whwovy/v4-artifacts/metadata-calibration-efficiency-v4/pilot-001/result.json
```

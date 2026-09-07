# V4 후속 AQ study001 결과 — 학습 신호는 있으나 primary 조건 집합은 비었음

2026-09-07, [사전 설계](metadata_calibration_v4_aq_study.md)를 수정하지 않고 단회 실행했다.
공식 상태는 **`NO_SOURCE_INFORMATIVE_CELLS`**다. Source A0만으로 미리 정한 조건을
고르는 단계에서 집합 S가 비어 **주효과는 산출하지 않았다**. AQ가 효과 없다는 판정도,
metadata가 효과 있다는 판정도 아니다. 원래 목표인 *외부 acquisition metadata가 EEG-only
방법을 넘어 적은 보정 예제로 SSVEP를 가능하게 하는가*는 유지한다.

## 쉽게 설명하면

이전 pilot은 보정 없이도 약98%여서 개선할 여지가 거의 없었다. 이번에는 관측 시간을
줄이고 잡음을 높이며, EEG 기준법과 개인별 보정 예제로 만든 평균 파형을 결합했다.
두 방법의 점수 크기가 다르므로 **개발용 사람의 합성 데이터에서만 확률 온도**를 맞춘 뒤,
새로운 평가용 합성 참가자에게 그대로 적용했다. 온도를 맞추는 것과 참가자에게 정답이
붙은 EEG를 추가로 받는 것은 다른 종류의 calibration이다.

제가 정한 격자가 거칠었다. 기본 잡음에서 A0는 source 0.5초에51.25%, 1초에98.06%로
뛰었고, 1초의 잡음을2배로 하면44.58%였다. 사전 범위인55~90%에 들어온 stable 조건이
없었다. **범위 밖이라는 이유만으로 '학습 불가능'이라고 할 수는 없다.** 실제0.5초 조건은
학습 신호가 강했다. 다만 그 결과를 본 뒤55%를50%로 낮추거나 이 조건을 primary로
바꾸면 원래 검증과 다른 실험이 되므로 그렇게 하지 않았다.

## 무엇을 관찰했는가

아래는 미리 평가 목록에 넣었던 **독립 evaluation의 cellwise diagnostic**이다.
BA는 balanced accuracy이고 이 데이터는 클래스별 query 수가 같아 일반 정확도와 같다.
`k=1/3/5`는 **각 클래스당**1/3/5개, 총12/36/60개의 labeled support trials다.
새 fusion은 `.5*A0_cal + .5*pooled_cal`; raw fusion은 동일 식에 온도.1을 공통 적용한다.

| 조건 | A0, k0 | Raw fusion, k3 | 새 fusion, k1 | 새 fusion, k3 | 새 fusion, k5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| stable, 0.3초, 잡음×1 | 18.61% | 24.38% | 35.63% | 57.64% | 67.43% |
| stable, 0.3초, 잡음×2 | 9.24% | 10.21% | 13.26% | 16.74% | 20.14% |
| stable, 0.3초, 잡음×3 | 8.54% | 8.75% | 10.69% | 11.53% | 12.92% |
| stable, 0.5초, 잡음×1 | 50.56% | 57.22% | 65.69% | 82.36% | 90.21% |
| stable, 0.5초, 잡음×2 | 16.94% | 17.78% | 19.72% | 24.17% | 30.56% |
| stable, 0.5초, 잡음×3 | 11.53% | 12.08% | 12.50% | 13.82% | 15.42% |
| stable, 1.0초, 잡음×1 | 98.33% | 98.82% | 98.89% | 99.65% | 99.79% |
| stable, 1.0초, 잡음×2 | 45.90% | 46.81% | 47.50% | 51.39% | 55.69% |
| stable, 1.0초, 잡음×3 | 22.15% | 22.50% | 22.78% | 25.28% | 26.53% |
| drift, 0.3초, 잡음×1 | 18.82% | 22.29% | 28.26% | 45.21% | 54.51% |
| drift, 0.3초, 잡음×2 | 9.65% | 9.86% | 11.60% | 14.17% | 17.15% |
| drift, 0.3초, 잡음×3 | 8.54% | 8.47% | 9.79% | 10.63% | 10.76% |
| drift, 0.5초, 잡음×1 | 52.22% | 56.46% | 57.92% | 71.67% | 79.38% |
| drift, 0.5초, 잡음×2 | 17.64% | 18.26% | 18.68% | 22.57% | 26.81% |
| drift, 0.5초, 잡음×3 | 11.94% | 11.94% | 11.81% | 13.06% | 14.03% |
| drift, 1.0초, 잡음×1 | 98.33% | 98.89% | 99.03% | 99.44% | 99.44% |
| drift, 1.0초, 잡음×2 | 47.85% | 48.13% | 47.64% | 50.21% | 52.15% |
| drift, 1.0초, 잡음×3 | 22.85% | 23.40% | 23.40% | 25.28% | 25.42% |

모든18조건을 유지했다. 나머지 standalone/block 방법과 모든 참가자별 count, log probability,
margin, flips, harm, CI는 [전체 결과](/home/whwovy/v4-artifacts/metadata-calibration-efficiency-v4/aq-study-001/result.json)의
`rows` 및 `summary`에 있다. 표는18조건을 독립 표본으로 합친 유의성 검정이 아니다.

0.5초·잡음×1을 설명용으로 자세히 보면:

- **파형 학습 자체가 작동한다.** Stable pooled standalone 정확도는 k1/3/5에52.92/80.63/88.75%다.
  새 fusion은65.69/82.36/90.21%다. 온도 변경은 standalone pooled의 class 순위를 바꾸지 않지만
  결합 시 각 방법이 미치는 영향은 바꾼다.
- **점수 크기를 맞춘 결합의 개선이 독립 평가에도 나타났다.** Stable k3의 calibrated−raw
  fusion 차이는+25.14 percentage points, 참가자24명 기준 diagnostic 95% CI는[22.34,27.94]pp다.
  Calibrated fusion k3−k1은+16.67pp, CI[14.13,19.20]pp다. 이는 계획된 개별 조건 diagnostic이며
  다중 비교 보정된 primary 성공 주장이 아니다.
- **평균80%와 모든 참가자의80%는 다르다.** Stable 새 fusion의 최초80% 도달은 k3에15/24명,
  k5에나머지9명이다. k0/k1에는0명이다. Drift에서는 k3에3명, k5에8명, 미달13명이다.
  Stable k3/k5의 분석 EEG는18/30초지만 실제 cue·휴식·준비 시간을 포함하지 않는다.
- **변동에 취약한 부분은 남는다.** Drift에서 학습 곡선은 개선되지만 stable보다 낮다.
  1초·잡음×2 drift의 새 fusion k1은 A0보다 평균0.21pp 낮고13/24명이 악화된다.
  평균 개선만으로 개인별 안전성·항상 이득을 주장하지 않는다.

## Temperature analogy의 검증 결과와 한계

Source24명 전체18조건에서 고정31-point grid의 standalone NLL만 최소화했다.
평가 결과로 온도나 fusion strength를 고르지 않았다. 선택 온도는 다음과 같다.

| Decoder | Source-fitted T |
| --- | ---: |
| A0, 모든 k 공통 | 0.0681292069 |
| pooled k1 = block k1 | 0.01 |
| pooled k3 | 0.00681292069 |
| pooled k5 | 0.00464158883 |
| block k3 | 0.00316227766 |
| block k5 | 0.00215443469 |

Evaluation18조건 동일 가중의 **기술 통계**로, A0 NLL은 raw2.20648→cal2.16942,
pooled k3는2.43677→2.07767, pooled fusion k3는2.27311→2.01964였다. Pooled fusion k3 BA는
33.125→40.822%로 높아졌다. Primary 집합이 비었으므로 이 전체 평균으로 대체 검정하지 않는다.

Confidence와 accuracy의 분리는 중요하다. Block k3는 NLL2.46083→2.15092로 좋아졌지만
standalone BA는33.819→32.222%로 낮아졌다. A0 cal NLL도 짧고 잡음이 큰 개별 조건에서는
raw보다 나빠졌다. **하나의 global temperature가 모든 조건에서 최적이거나 개선을 보장하지 않는다.**

`academic-research`에서 확인한
[Guo et al., ICML2017](https://proceedings.mlr.press/v70/guo17a.html)은 neural confidence
calibration의 인접 근거이지 EEG 효능을 증명한 논문이 아니다. 이번은 이 아이디어를
EEG decoder score fusion으로 옮긴 **한정된 합성 평가**이며, NLL와 fused BA의 일부 개선은
보였지만 primary utility와 metadata 기여는 확인하지 못했다. 새로운 PDF 전문 검토로
포장하지 않는다. 해당 논문은 official HTML/abstract 범위로 확인했고, search의429 및
PMC browser-verification 접근 제한은 research workspace에 남겼다.
Workspace 기록은 analogy `analogy:0caf76fb624b38c6`(tested), 측정 claim
`claim:88306b92c3627adb`, 다음 검증 gap `gap:088f998efd237544`다.

DGP 자체의 한계도 크다. 참가자/클래스별 phase·topography를 유지하고 trial phase jitter는
0.08rad로 작게 설정했다. 이는 평균 파형 학습에 유리한 구조다. 큰 합성 이득은 이 구조 안의
메커니즘과 구현을 지지할 뿐, 실제 EEG의 변화 폭·장기 drift·전극 물리와 일치한다는 증거가 아니다.

## 판정과 다음 설계 결정

연구목표는 바꾸지 않는다. 바꿀 것은 연구목표가 아니라 **개발 단계의 난이도 탐색 해상도와
EEG-only 비교법의 완성도**다.

1. **이번 study는 종료·보존한다.** S를 다시 고르지 않고 primary eAUC/CI는 null로 유지한다.
   `AQ_READY_FOR_CONTEXT_COMPARATOR_STUDY`로 승격하지 않는다. 같은 seed/자료를 다시 돌려
   결과를 구제하거나 성공할 때까지 independent test를 반복하지 않는다.
2. **다음 개발에서는 A0-only source 지도를 촘촘하게 만든다.** 0.5~1초 사이의 길이와 잡음×1~2
   사이를 별도 사전 고정한 유한 격자로 나누는 것이 우선이다. 현재55~90% 기준을 소급 완화하지 않는다.
   격자와 source-only 선택 규칙을 새로 고정한 후, source에서 조건을 정하고 independent evaluation은
   마지막에 한 번만 연다. 이번 문서는 다음 실행 config나 새로운 RNG draw가 아니다.
   또한 **AQ 학습 효과 측정과 metadata용 난이도 적격성은 분리**한다. 다음 계획에서 AQ는
   사전 지정한 전체 조건의 효과/난이도별 결과를 남기고, A0 적격 집합은 후속 metadata 평가
   조건을 정하는 별도 개발 산출물로 두는 것이 타당하다. 정확한 primary·가중치·종료 규칙은
   다음 outcome 전에 정해야 하며, 통과할 때까지 잡음과 primary를 바꾸는 루프는 하지 않는다.
3. **Metadata 전에 강한 Q 비교법을 갖춘다.** 현재 Q는 labeled support의 class-specific harmonic
   reliability뿐이고 query-support context matching이 없다. Drift 손실이 있다는 이유만으로
   metadata가 유일한 해법이라고 가정하지 않는다. EEG에서 추정한 측정 조건과 외부 측정 context를
   같은 waveform-learning 구조에서 비교해야 한다. Metadata는 synthetic state 정답이나 EEG로
   만든 품질 수치를 이름만 바꾼 것이어서는 안 된다.
4. **최종 검증은 상대적 보정 비용이다.** 같은 query set·목표 성능·보정 budget에서 M+Q가 Q-only보다
   실제로 필요한 labeled trials를 줄이는지 본다. A0보다 좋아진 것, confidence가 좋아진 것,
   drift에서 잃은 성능이 있다는 것은 각각 그 주장과 다르다. 실제 acquisition metadata의
   가용성·누락·신뢰도와 사람 EEG 일반화는 여전히 별도 병목이다.

새 연구목표, 손상 복원, OOD/discovery로 전환하지 않는다. 이번 결과는 유용한 EEG-only
학습 구조를 만드는 데는 진전이지만, metadata 논문의 주장을 입증한 결과는 아니다.

## 실행·검증·재현 자료

- Clean source commit `bd687099af79fac2fa904c7ec60cd5afb4d30e01`, tree
  `4111babd189e627dc8423eca2b8b16ef1adccd0d`; start UTC `2026-09-07T05:57:46.645549+00:00`.
- Source/evaluation 각24명의 독립 합성 참가자,18조건,10방법,4budgets:17,280 metric rows.
  CPU4 workers/BLAS1, elapsed23.47s, max RSS364,308KB. Human data access/unlock=false.
- Source root seed `17469666171393023058`, evaluation root seed `10574746855199264036`.
  Start 이후 source seed를 열고, source-freeze/evaluation-start를 durable publication한 뒤 evaluation
  seed를 열었다. 독립 reviewer의 사전 코드 검토와 fixture ordering test로 순서를 확인했다.
  Hash chain만으로 시간 순서를 외부에 증명한다는 주장은 하지 않는다.
- 새 집중시험27/27, 전체942/942, 기존 warnings68. 별도 독립 검산은3.86s, **PASS**:
  186온도 objectives,17,280 metric rows,180 attainment rows,720 diagnostics와 provenance 확인.
  DGP를 재실행하지 않고 저장된 score로 계산했다. 소수점 산술 동등성을 확인한 것이며 모든
  posterior byte hash를 재생성했다는 주장은 아니다.
- `coordinate-worktree-changes`에 따라 구현자는 별도 worktree의3개 새 파일만 작성하고 main이
  계약·통합·독립 auditor·최종 실행을 맡았다. 기존 V3/V4 코드·설정·artifact는 그대로 보존했고
  dependency/GPU 변경이나 기존 worktree 삭제는 없다.

Artifact root는 `/home/whwovy/v4-artifacts/metadata-calibration-efficiency-v4/aq-study-001`이다.
정확히 여섯 regular files, 각각 mode0400/nlink1이며 총약60.7MB다.

| Artifact | bytes | SHA-256 |
| --- | ---: | --- |
| start.json | 1285 | `800c88c982744c0ddc7ca330a048ac3162e4f7ed9e66819591f62fa63bf46a86` |
| source-scores.npz | 21460662 | `272b94695833a2d4e1fb1f8dd8552e5db1268996a962efc52808ec6d00cdcc19` |
| source-freeze.json | 20615 | `79cedbe5959247ece205a108d750612aa1102b9fc2ac6c3a928903442d517063` |
| evaluation-start.json | 508 | `b7f13782fc6008606d8b1ed7b04bbc54ce9f461da55009360ed164b7f1f19ab8` |
| evaluation-scores.npz | 21460308 | `293d127e565f21074735ee4c167e4ac23dd37264cfc0f3892ef8fb88dd1aa7d7` |
| result.json | 17784831 | `a030164b20a9074c2eb388bc71db310d9faa563dc9d1e3c6fedb66aa4180e5b4` |

Read-only 재검산 명령:

```bash
.venv/bin/python scripts/audit_metadata_calibration_v4_aq_study.py \
  --plan /home/whwovy/califreeEEG/configs/analysis/metadata_calibration_v4_aq_study.json \
  --output /home/whwovy/v4-artifacts/metadata-calibration-efficiency-v4/aq-study-001
```

Runner는 existing start 때문에 재실행을 거부한다. 그 경계를 우회하지 않는다.

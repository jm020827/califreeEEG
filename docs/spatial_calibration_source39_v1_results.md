# Spatial calibration source39 v1 — actual development result

2026-09-07. **AQ_NOT_ESTABLISHED — candidate closed, independent score/provenance audit PASS.**

## 쉽게 설명

보정 EEG와 현재 EEG에서 각각 유용한 채널 조합을 찾는 IT-CCA를 구현해 실제 개발용39명에게 시험했다. 그러나 보정 없는 모델보다 유용하게 좋아지지 않았다.27개 fold×window×budget 혼합비 중24개가 보정 정보를 버리는 lambda0을 선택했다. 보정 정답을 잘못 붙인 대조군과도 의미 있게 구분되지 않았다.

이번에는 metadata를 넣지 않았다. **Metadata 무용성을 입증한 것이 아니라, metadata 효과를 비교할 EEG-only learner가 아직 확보되지 않았다는 결과**다. 목표는 유지한다. k1/3/5는 자극당 보정 횟수이고 전체 labeled trials는12/36/60개다.

| 모든6조건 동일 비중 평균 | k0 | k1 | k3 | k5 |
|---|---:|---:|---:|---:|
| 보정 없는 A0 | 35.620% | 35.620% | 35.620% | 35.620% |
| A0 + IT-CCA, 주방법 AQ | 35.620% | 35.620% | 35.634% | 35.648% |
| IT-CCA 단독 | 해당 없음 | 10.620% | 11.197% | 11.667% |
| unit-TRCA diagnostic | 해당 없음 | 해당 없음 | 17.457% | 23.255% |
| 기존 legacy-TRCA diagnostic | 해당 없음 | 해당 없음 | 25.577% | 31.702% |

우연 정답률은8.333%다. IT-CCA의 class 순위 구분력이 약해 temperature만으로 해결할 수 없다. A0도 조건별12.5–58.0%여서 개선 여지는 충분하다. 새 TRCA 변형은 기존보다 나빴다. 구현 차이를 확인한 사실과 좋은 기준 모델을 재현했다는 주장은 구분해야 한다.

## Prespecified decisions

[Frozen design](spatial_calibration_source39_v1_design.md), [machine contract](../configs/analysis/spatial_calibration_source39_v1.json). All39 participants were previously exposed. Fit26/evaluate13 cross-fitting does not erase exposure; Student-t intervals below are descriptive and do not correct shared-training-fold dependence. All six cells have equal within-person weight; no outcome-selected cell/family.

eAUC=BA0/6+BA1/2+BA3/3; k5 is diagnostic. Differences below are **percentage points**, not relative percentages.

| Screen | Mean | One-sided95 lower bound | Required | Result |
|---|---:|---:|---|---|
| AQ−A0 early-AUC | +0.004748pp | −0.003257pp | mean≥1pp and lower>0 | FAIL |
| Correct−wrong-label AQ early-AUC | +0.001079pp | −0.006110pp | mean≥1pp and lower>0 | FAIL |
| k1 AQ−A0 | 0pp | 0pp | mean≥0 | PASS |
| Dry AQ−A0 early-AUC | +0.004748pp | −0.003257pp | mean≥−0.5pp and lower>−1.667pp | PASS |
| Wet AQ−A0 early-AUC | +0.004748pp | −0.009239pp | same safety bound | PASS |

AQ5−AQ3=+0.014245pp, LCB−0.034260pp: no clear additional learning. Safety reflects near-fallback behavior, not efficacy. Formal terminal: AQ_NOT_ESTABLISHED.

## Every declared condition

Balanced accuracy, percent. N188 is0.752s.

| Interface/window | A0=AQ1 | AQ3 | AQ5 | ITCCA1/3/5 |
|---|---:|---:|---:|---|
| Dry0.5s | 12.521 | 12.564 | 12.479 | 7.949 / 9.444 / 8.675 |
| Wet0.5s | 27.607 | 27.650 | 27.821 | 12.094 / 12.265 / 13.675 |
| Dry0.752s | 29.487 | 29.487 | 29.487 | 8.162 / 9.145 / 9.188 |
| Wet0.752s | 48.590 | 48.590 | 48.590 | 13.333 / 13.205 / 14.615 |
| Dry1s | 37.521 | 37.521 | 37.521 | 8.462 / 9.316 / 8.974 |
| Wet1s | 57.991 | 57.991 | 57.991 | 13.718 / 13.803 / 14.872 |

| Interface/window | unit-TRCA3/5 | legacy-TRCA3/5 |
|---|---|---|
| Dry0.5s | 9.786 / 11.880 | 11.239 / 13.590 |
| Wet0.5s | 24.530 / 33.761 | 38.120 / 47.778 |
| Dry0.752s | 10.385 / 11.709 | 11.496 / 13.803 |
| Wet0.752s | 24.231 / 35.043 | 39.915 / 50.385 |
| Dry1s | 10.171 / 12.179 | 11.325 / 13.889 |
| Wet1s | 25.641 / 34.957 | 41.368 / 50.769 |

TRCA3→5 learns something, especially wet, but neither diagnostic becomes primary retrospectively. The unit variant changed covariance handling, scaling, centering and band aggregation together; its poorer result cannot isolate the responsible ingredient. No SOTA/exact2018 reproduction claim.

## Controls, harm and useful-accuracy attainment

- Lambda0×24; lambda.25×3, all fold2/N125. Their k1/3/5 support temperatures are10/3.1623/1. Longer-window AQ returns exact A0. Interfaces share fits; no metadata-conditioned heads.
- Uniform shrinkage preserves A0 decisions. AQ−uniform all-cell mean correct-log-probability increments at k1/3/5 are only+0.0000035894/+0.0000238347/+0.0000822367. AQ mean log probability is slightly worse than A0. This is not useful label learning.
- Wrong-label control uses all11 cyclic class-label shifts with frozen parameters and averages metrics, not posteriors. Its attainment is not a deployable predictor's cost.
- Per budget14,040 queries (39×6×60): AQ changes4/14/43 A0 decisions at k1/3/5, net0/2/4 correct answers. All39 participant mean BA values are unchanged at k1 despite four individual prediction changes. At k3 help/harm/tie=3/1/35; k5=4/2/33.
- AQ and A0 have identical80% first-attainment profiles. In dry125/wet125/dry188/wet188/dry250/wet250 order,0/3/1/8/4/11 participants already reach80% at k0; the others remain censored beyond5. AQ creates no new80% attainers: calibration saving at useful accuracy is not shown. EEG_seconds means analyzed samples, not cue/rest/whole-session time.

## Post-outcome saved-score diagnosis

[Diagnostic script](../scripts/diagnose_spatial_calibration_source.py) is read-only: no raw EEG reopen, fitting, new condition selection or result mutation. Its2fixture tests pass. Weighted squared canonical-correlation **scores** are not raw amplitudes or unsquared canonical coefficients.

At dry1s/k5, mean correct-template score=.787714927 and mean wrong-template=.784661721; wet1s/k5=.517756896 versus .494901156. Higher dry similarity does not mean better label information. Median participant ITCCA accuracy across cells is8.33–10%; a few wet participants perform well, which does not justify selecting them.

This establishes weak class separation, **not** its physical cause. Summed scores cannot determine whether common interference, filter-edge effects, timing, support reproducibility or another factor caused it. We do not invent a damaged-EEG research objective.

Optional comparison with the closed context study's cached statistics shows exact equality of subject/cell IDs, raw-file hashes, crop hashes, A0 scores and legacy-TRCA scores. Those anchors did not change. This is stored-cache consistency, not independent raw preprocessing reproduction.

## Next bounded loop

1. Close this exact candidate. No same-study retry, outcome-based ridge/lambda tuning, stronger metadata weights or wet-only success selection.
2. Freeze a new **source39-only reference-baseline diagnostic**, retaining all six conditions: per-band correct/wrong-template separation, support-template reproducibility and evidence at known stimulus frequencies. Check dataset class/frequency/epoch conventions against the original specification. These are open hypotheses, not established bugs.
3. Specify one reference-guided standard learner, such as extended CCA, with a faithful baseline check before new efficacy claims. The [original paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140703), pp8–9/Eqs14–15, defines an operator distinct from IT-CCA; it does not guarantee success under our split. Known stimulus references can constrain what common waveform components count, but a fix is not yet demonstrated.
4. Once a useful EEG-only comparator exists, test acquisition-M increment, shuffle/stale/chronology specificity and calibration cost in a separate protocol. Held60 remains unopened. Independent metadata replication data is still needed; the [request draft](independent_impedance_data_request.md) remains unsent. No new dataset or ethics approval was obtained here.

These are next actions, not an implemented/executed second study or held-access authorization. Academic-research grounded the operator distinction in targeted original-PDF reading and recorded claim provenance. Coordinate-worktree-changes separated implementation/auditing and preserved older worktrees/results.

## Reproducibility

Plan SHA`8198c28a36f43465fb2f7c62dfa1e0be2f6415aa11740c1dde906ab0cd157552`.
Execution commit`10b9f36b96ead79e6180fb101343d5164ebd051f`, tree`31af2171b09f46514bac519dd72c7dc367876870`.
Start UTC2026-09-07T08:05:31.691601+00:00. CPU4/BLAS1,19.77s,time-reported maxRSS284628KB. Study RNG none. Raw39 opened after durable start; no metadata/manifest/held/retired/oldrunner access.

Directory `/home/whwovy/spatial-calibration-artifacts/source39-v1`, exact4 regular files, mode0400/nlink1:

| File | Bytes | SHA256 |
|---|---:|---|
| start.json | 1835 | c4302be1564aaf2f0cc4a3ebe8b3cda2cc23bdbb1ed44aa899e4de6f715cda89 |
| scores.npz | 10222711 | bb73f8ac10cd1f257a3df00b7b381d0ded9452a62aa8d633f8d805c35d7c6c26 |
| fold-freezes.json | 653682 | 1eebbbe4515868f93cfb3e3270447e5441e44d5b8357c853b8c8aaf6787c6aed |
| result.json | 11589023 | 796003e50e720c995c6c3e45bbf32b6c804403b87b4d0e0849550afaa7477197 |

Pre-outcome full suite1034PASS/68existingTorch warnings/115.16s; independent CODE-GO. Actual arithmetic/provenance audit3.77s PASS:4770objectives,5382rows,138aggregates,897harm,42attainment. Independent read-only post-outcome reviewer agrees with closeout and diagnosis limits. No raw decoder replay claimed by this auditor.

Read-only verification, **not** study rerun:

```bash
.venv/bin/python scripts/audit_spatial_calibration_source.py \
  --plan /home/whwovy/califreeEEG/configs/analysis/spatial_calibration_source39_v1.json \
  --output /home/whwovy/spatial-calibration-artifacts/source39-v1
.venv/bin/python scripts/diagnose_spatial_calibration_source.py \
  --output /home/whwovy/spatial-calibration-artifacts/source39-v1 \
  --previous-features /home/whwovy/context-template-artifacts/source39-v1/features.npz
```

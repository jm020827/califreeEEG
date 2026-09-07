# Reference-calibration-source39-v1 결과와 후속 결정

2026-09-07. **단회 실행·독립 검산 완료. 두 arm 모두 AQ_NOT_ESTABLISHED.**
전체 DIAGNOSTIC_COMPLETE는 실행 완료만 뜻한다. 목표는 metadata-assisted low-calibration SSVEP 그대로다.
이번에는 metadata를 시험하지 않았다. [실행 전 설계](reference_calibration_source39_v1_design.md)와
[고정 JSON](../configs/analysis/reference_calibration_source39_v1.json)을 변경하지 않는다.

## 쉽게 읽는 결론

개인 EEG의 정답 연결에는 정보가 있다. 실제 정답으로 학습하면 잘못된 정답을 붙였을 때보다 낫다.
그러나 현재 ECCA가 그 정보를 쓰는 방식은 보정 없는 사인파 기준을 크게 악화시킨다.
**정보가 있다는 것과, 추가 측정 비용을 들일 가치가 있다는 것은 다르다.**
Notch는 일부 손실을 줄였으나 이 문제를 해결하지 못했다. 더 많은 labels의 개선도 기준을 회복하지 못한다.
이는 논문의 일반적 실패나 metadata의 무용성을 증명하지 않으며, 이 유한 개발 후보의 실패다.

39명×6조건을 동등 평균한 balanced accuracy(%):

| 전처리 | 사인파 기준 Aref k0 | ECCA k1 | ECCA k3 | ECCA k5 | ITCCA k5 |
|---|---:|---:|---:|---:|---:|
| no_notch | 35.620 | 17.301 | 20.655 | 22.799 | 11.667 |
| causal_notch50 | 35.043 | 22.286 | 23.903 | 25.057 | 22.742 |

k=1/3/5는 class당 예제 수다. 12class이므로 개인별/interface별 labeled trials는12/36/60개다.
Aref는 개인 labeled EEG 없이 현재 query와 공개 stimulus reference만 쓴다.
이번 temperature는 각 standalone posterior의 NLL만 보정한다. 양수 T는 class 순위를 바꾸지 않고,
fusion lambda도 없으므로 실패를 이전의 A0 fallback이나 확률 평탄화 탓으로 설명하지 않는다.

## 사전 고정 판정

아래 차이는 percentage points이며 LCB는 one-sided95%, CI는 descriptive paired95%다.
eAUC는 BA0/6+BA1/2+BA3/3으로 초기 보정 budget을 요약한다. 39명은 과거 결과에 이미 노출됐고
cross-fit은 training을 공유한다. 이 CI를 독립 confirmation이나 반복 탐색을 보정한 유의성으로 부르지 않는다.

| 비교 | no_notch mean [95% CI]; LCB | causal_notch50 mean [95% CI]; LCB |
|---|---|---|
| ECCA−Aref eAUC | −14.148 [−17.513,−10.783]; −16.950 | −10.091 [−12.883,−7.300]; −12.416 |
| ECCA−wrong eAUC | +5.730 [+3.192,+8.268]; +3.616 | +11.326 [+7.531,+15.121]; +8.166 |
| k1 ECCA−Aref | −18.319 [−22.741,−13.897]; −22.002 | −12.756 [−16.288,−9.225]; −15.697 |
| dry ECCA−Aref eAUC | −11.712 [−15.465,−7.959]; −14.837 | −10.546 [−14.540,−6.552]; −13.872 |
| wet ECCA−Aref eAUC | −16.584 [−21.218,−11.949]; −20.443 | −9.637 [−13.042,−6.232]; −12.472 |
| k5−k3 ECCA diagnostic | +2.144 [+0.938,+3.350]; +1.139 | +1.154 [+0.568,+1.739]; +0.666 |

양 arm 모두 supervised-specificity만 PASS, utility/k1/dry/wet safety는 FAIL이다.
원래 utility 기준(mean≥1pp 및 LCB>0)과 safety를 낮추지 않는다.
Notch−no_notch의 ECCA eAUC는+3.479pp [0.431,6.528], Aref는−0.577pp [−2.088,0.934]다.
따라서 calibration-gain interaction은+4.056pp [0.442,7.670]이다.
이는 **음수인 보정 이득이 덜 음수가 된 것**이지 유용한 calibration 입증이 아니다.
Wrong-label eAUC도−2.117pp 변했으므로 specificity 증가 전부를 실제 decoder 향상으로 해석하지 않는다.
ITCCA k1/3/5는 notch에서+3.732/+8.618/+11.075pp 개선됐다.

## 모든 조건의 정확도

각 값은39명 평균 BA(%). N125/188/250은0.5/0.752/1초다.
Wrong은11개의 template-label cyclic shift에서 계산한 metric 평균이지 deployable predictor가 아니다.

| Pipeline | Cell | Aref k0 | ECCA k1 | ECCA k3 | ECCA k5 | Wrong k3 | ITCCA k5 |
|---|---|---:|---:|---:|---:|---:|---:|
| no_notch | dry-n125 | 12.521 | 10.128 | 11.068 | 13.120 | 8.908 | 8.675 |
| no_notch | wet-n125 | 27.607 | 18.932 | 23.718 | 27.350 | 9.538 | 13.675 |
| no_notch | dry-n188 | 29.487 | 12.393 | 13.761 | 14.658 | 10.971 | 9.188 |
| no_notch | wet-n188 | 48.590 | 23.034 | 30.214 | 33.077 | 12.304 | 14.615 |
| no_notch | dry-n250 | 37.521 | 13.376 | 14.744 | 14.786 | 11.445 | 8.974 |
| no_notch | wet-n250 | 57.991 | 25.940 | 30.427 | 33.803 | 13.695 | 14.872 |
| causal_notch50 | dry-n125 | 14.103 | 9.658 | 9.872 | 10.043 | 7.797 | 13.462 |
| causal_notch50 | wet-n125 | 30.427 | 22.692 | 24.274 | 26.239 | 7.537 | 25.470 |
| causal_notch50 | dry-n188 | 24.530 | 11.282 | 12.222 | 13.291 | 7.925 | 14.701 |
| causal_notch50 | wet-n188 | 48.803 | 35.513 | 38.974 | 40.897 | 9.623 | 32.137 |
| causal_notch50 | dry-n250 | 32.949 | 11.496 | 13.291 | 13.419 | 7.218 | 16.026 |
| causal_notch50 | wet-n250 | 59.444 | 43.077 | 44.786 | 46.453 | 11.344 | 34.658 |

No_notch k1은39명 모두 자신의 all6-cell 평균에서 Aref보다 악화됐다.
k3/k5는 help3/harm36, notch k1/k3/k5는 각각 help4/harm35였다. Tie0.
특정 wet/긴 window나 한 대역의 나은 결과로 primary를 교체하지 않는다.

## 비용·최초80% 도달

| Pipeline | Cell | Aref first80 (k0 / >5) | ECCA first80 (k0 / k1 / k3 / k5 / >5) |
|---|---|---|---|
| no_notch | dry-n125 | 0 / 39 | 0 / 0 / 0 / 0 / 39 |
| no_notch | wet-n125 | 3 / 36 | 3 / 0 / 0 / 0 / 36 |
| no_notch | dry-n188 | 1 / 38 | 1 / 0 / 0 / 0 / 38 |
| no_notch | wet-n188 | 8 / 31 | 8 / 0 / 0 / 1 / 30 |
| no_notch | dry-n250 | 4 / 35 | 4 / 0 / 0 / 0 / 35 |
| no_notch | wet-n250 | 11 / 28 | 11 / 0 / 0 / 1 / 27 |
| causal_notch50 | dry-n125 | 0 / 39 | 0 / 0 / 0 / 0 / 39 |
| causal_notch50 | wet-n125 | 3 / 36 | 3 / 0 / 0 / 0 / 36 |
| causal_notch50 | dry-n188 | 1 / 38 | 1 / 0 / 0 / 0 / 38 |
| causal_notch50 | wet-n188 | 7 / 32 | 7 / 1 / 0 / 0 / 31 |
| causal_notch50 | dry-n250 | 3 / 36 | 3 / 0 / 0 / 0 / 36 |
| causal_notch50 | wet-n250 | 14 / 25 | 14 / 1 / 0 / 0 / 24 |

각 행은39명. >5는 관측 budget 내 미달이며 비용을 가상으로 대입하지 않는다.
k0 alias가 있으므로 Aref의 기존 도달자를 ECCA의 새 보정 성과로 세면 안 된다.
k0 도달 이후 k>0에서 성능이 악화되더라도 first-observed k0는 남는다.
새로운 도달은 no_notch wet-n188/n250에서 각각1개의 participant-cell(k5),
notch 같은 두 조건에서 각각1개(k1)뿐이다. 전체 utility/safety 실패를 뒤집는 결과가 아니다.

Label cost=12k; retained EEG seconds=12k*N/250. 별도 available history=12k*0.64초다.
예컨대 k3/1초는36 labeled trials, retained36초, preceding history23.04초다.
두 arm의 prefix 접근 권한은 같고 실제 notch만 history를 활용한다.
Intertrial interval·지시·측정 준비·impedance 측정 시간이 빠져 있으므로 전체 보정 wall-clock 절감 주장은 없다.

## 사전 지정 신호 진단과 사후 점수 분해

[모든7band×6조건×두 arm 표](reference_calibration_source39_v1_band_diagnostics.md)를 보존했다.
독립 진단은588 score rows,84 signal rows,12 line50 rows를 생성한다.
Query의 pre-bank50Hz 투영 에너지 비율은 전체 평균 .407130→.160065,
support는 .425904→.157722였다. Support의 대역별 flattened Pearson 평균은
−.006580→+.483661로 높아졌다. 그러나 query의 정답/평균오답 reference 투영 에너지는
.065284/.060503→.085312/.079803이었다. **높은 반복 상관 자체가 class 구분력은 아니다.**
Notch는 .64초 history와 필터 동작을 함께 바꾸고 upstream downsampling은 재검증하지 않았다.
물리적50Hz가 단독 원인이라거나 저자의 full710-sample filtfilt를 정확히 재현했다고 주장하지 않는다.

다음은 outcome을 본 뒤 추가한 **사후적 fixed-decision attribution**이다. 재학습·성분 제거·재가중·새 decoder 평가는 아니다.
각 arm/budget의14,040개 결정에서 실제 Aref/ECCA 순위를 그대로 비교했다.
이14,040개를 독립 표본 수로 삼지 않는다. Main 계산은1,404 participant×arm×cell×budget 행의 정답 수를
canonical result와 전부 대조했다.

| Pipeline | k | 정답→오답 | 오답→정답 |
|---|---:|---:|---:|
| no_notch | 1 | 3505 | 933 |
| no_notch | 3 | 3133 | 1032 |
| no_notch | 5 | 3013 | 1213 |
| causal_notch50 | 1 | 2639 | 848 |
| causal_notch50 | 3 | 2458 | 894 |
| causal_notch50 | 5 | 2330 | 928 |

k3의 손상 결정에서 정답−실제 ECCA 선택 오답 margin을 네 signed-square 항으로 정확히 분해하면:

아래 손상/복구 성분 기여는 해당 결정 수로 가중한 pooled-decision 평균이다.
참가자·조건마다 그 사건의 발생 수가 다르므로 nonempty participant-cell 평균의 동등 평균과 다르다.
성분별 class 간 SD는 각 query의12class에서 계산한 뒤 query 평균, participant-cell 동등 평균을 취한다.
이 사후적 요약에는 별도 유의성 검정이나 독립 표본 수 주장을 붙이지 않는다.

| Pipeline | r1 기여 | r2 기여 | r3 기여 | r4 기여 |
|---|---:|---:|---:|---:|
| no_notch | +.117649 | −.892674 | +.002979 | +.014844 |
| causal_notch50 | +.087852 | −.005055 | −.309620 | −.046625 |

No_notch k3의 class 간 점수 SD는 r1=.055783, r2=.452293: 개인 template에서 얻은 r2가
기준 항보다 약8.1배 크게 흔들리며 좋은 reference margin을 압도했다.
Notch 후 SD는 r1=.056667,r2=.036141,r3=.139429,r4=.075518이고 주된 손상 기여가 r3로 바뀐다.
같은 성분이 복구 사례에서는 도움이 된다(no_notch r2=+.695867, notch r3=+.201284).
따라서 해당 항을 무조건 삭제해야 한다는 결론은 아니다. 정답−평균오답 margin은 네 항 모두 양수여도
가장 강한 오답이 정답을 이길 수 있다. 이는 현재 점수 합산의 실패 설명이지 새로운 효능·물리적 인과효과 주장이 아니다.
독립 read-only agent의 별도 분해와 main의 새 재현 함수가 일치했다.

## 실행·검증·복구 기록

- Source commit e53d459e3cdbd83b2d2df31b4aebc6a7c9520ac7, tree0d4ce1ed27e2726d8c6b7fcfe177ceae7ae42075.
- Start UTC2026-09-07T09:10:34.381409+00:00. Start 이후에만 allowlisted raw39를 읽었고 모든3fold fit publication 뒤 평가했다.
- CPU4/BLAS1,80.60초, GNU time maxRSS1,462,476KB, exit0. 이 maxRSS는 전체 병렬 process 합계의 연속 peak 측정이 아니다.
- 독립 audit8.43초 PASS:4,185 objectives/7,956 metric rows/204 aggregates/1,326 harm/54 attainment/provenance.
- Pre-outcome 전체1095PASS/68기존Torch warnings/116.46s. Posthoc 함수2tests 추가 후 producer+auditor61PASS/2.92s.
- Wrong aggregate score의 실제 추출은 true-only component cache만으로 독립 재구성되지 않는다. 별도 scalar kernel fixture가 검증했다.
  Auditor의 posterior hash는 형식/alias 검사이며 모든 posterior byte 재현을 주장하지 않는다. Human raw preprocessing replay도 아니다.
- 기존 spatial cache와 sourceIDs/cells/rawhash/no_notch crophash/legacyA0/ITCCA scores가 **exact 동일**했다.
  이는 이전 no_notch 측정과의 계보 검증이며 이전 실패 결과를 소급 수정하지 않는다.
- Artifact root /home/whwovy/reference-calibration-artifacts/source39-v1. Exact4files, 모두0400/nlink1, 총432,050,210bytes<1GiB.
  기존 start가 있으므로 이 runner를 다시 실행하지 않는다. 새 artifact 추가·재fit·조건선택0.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| start.json | 2024 | 1060949abc9e82be3e8170d8ac4f165fccdc091112769c7256f54ce44d10e0ea |
| features.npz | 410527230 | 784a07a76d8c901b819f6ac2654d5d9a248206d5e3bed4ec7eff8413ccc92744 |
| fold-freezes.json | 385190 | e4ad6f866731980c484c51182c2c048603aebe07ddc401b117188c362146fde5 |
| result.json | 21135766 | 661ef36b97cd556721ab98aea844419026b69b713fe3a1906bcbcd7a173e8e43 |

Repository 루트에서 read-only 재검산(실험 runner가 아님):

~~~bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python scripts/audit_reference_calibration_source.py \
  --plan /home/whwovy/califreeEEG/configs/analysis/reference_calibration_source39_v1.json \
  --output /home/whwovy/reference-calibration-artifacts/source39-v1
# 모든 사전 진단: --diagnostics
# 명시적 사후 점수분해: --component-diagnostics
~~~

## 다음 결정 — 목표 유지, 현재 후보 종료

두 arm 모두 종료하며 notch를 자동 채택하거나 metadata/held로 승격하지 않는다.
다음 구현 질문은 **좋은 reference 판단을 보존하면서 support의 class-specific 증거만 반영할 수 있는가**다.
이미 저장된 성분·대역 진단을 새 유한 후보의 근거로만 삼고, fit26-only 신뢰도/규모 보정,
reference fallback 및 real/wrong controls를 실행 전에 별도로 고정해야 한다.
어느 항/대역을 삭제할지, fusion을 어떻게 할지는 이번 결과의 사후 winner로 결정하지 않는다.
이 다음 operator의 계약이나 실행은 아직 없다. 같은 source39의 추가 개발은 독립 확인이 아님을 계속 명시한다.

유용한 EEG-only 기준 뒤에야 같은 learner의 Q 대 acquisition M+Q, missing/shuffle/stale/order,
동일 정확도에서 label cost 비교로 돌아간다. 목표를 손상 EEG 복원이나 discovery/OOD로 바꾸지 않는다.
새 독립 acquisition-context dataset 확보0, 저자 요청 발송0.
[독립 impedance 자료 요청안](independent_impedance_data_request.md)은 미발송이며
paired raw EEG/시점이 명확한 block/channel metadata/사용조건/참가자 중복 확인이 여전히 필요하다.
이번 실행은 metadata/manifest/held60/retired1–3/old runner/seal 접근0이다. 기관 ethics 승인이나 전체 연구 완료를 주장하지 않는다.

academic-research는 Eq14–15/공식 dataset 문서/author preprocessing 차이를 설계에 반영하고
reported definition과 로컬 measurement를 별도 evidence로 기록하게 했다.
coordinate-worktree-changes는 producer만 분리된 worktree에서 작성하고 main의 독립 검산·통합을 유지하게 했다.
원문과 code의 정확한 URL/hash는 실행 전 설계와 research workspace 카드에 있다.

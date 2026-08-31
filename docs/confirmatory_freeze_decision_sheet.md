# Confirmatory freeze 결정표

기준일: 2026-09-01
상태: **physical reveal #2 validator-only exact restart 승인·개발 gate 동결 — confirmatory/lockbox 봉인 유지**

이 문서는 `configs/analysis/wearable_primary.yaml`의 남은 confirmatory `null`과 별개로, physical reveal #2의 개발 진단값을 결과 전에 고정한 승인표다. `DEC-20260901-004`는 validator-only 정정 뒤 같은 개발 실행을 재승인하며 `plan.status=dev_not_frozen`과 lockbox 봉인은 유지한다. 불변 기준원은 `configs/governance/wearable_physical_reveal2_decision.json`이다.

## 현재 증거

- 전체 `wearable_v3`: 102명·24,480행 raw↔processed 전수 감사 완료. 이 감사는 신호 byte를 읽었지만 성능 평가가 아니다.
- S1–S3 첫 outcome-gated **legacy `prompt_adapter_v1`** grid: A0 BA 0.1222, A2 BA 0.1042, Δ −0.0181. 세 participant 모두 음수이나 N=3 최소 단측 exact p=0.125라 `directional-warning / benefit-not-demonstrated / population-inconclusive`다. 새 `physical_hybrid_v1`의 성능 근거가 아니다.
- Wang source-learning: spectral BA 0.8220, pure F0 BA 0.8107, CCA BA 0.7875, strict Chen-2015 FBCCA BA 0.7768. Backbone 학습 가능성의 개발 근거이며 metadata 효과가 아니다.
- 독립 N=60 lockbox용 target-free simulation: 네 core DGP에서 Type-I Wilson upper ≤0.0581, one-sided coverage lower ≥0.9413, SESOI+0.03 power lower ≥0.8769.
- S4–S102는 전체 무결성 감사 외에 모델 학습·선택·성능 계산에 쓰지 않았고, confirmatory checkpoint·prediction·reveal receipt는 없다.

## 승인할 값

| YAML key | 정확한 의미 | 권고안 | owner가 확인할 판단 |
|---|---|---|---|
| `sesoi_balanced_accuracy` | A2−A0가 넘어야 “실질적 개선”으로 부를 BA 절대차 | **운영 효용을 기준으로 0.01 또는 0.02 중 선택** | 1–2 percentage point가 실제 BCI 비용·정확도에 충분한가 |
| `inference.alpha` | 단일 overall primary contrast의 Type-I 기준 | **0.05** | 더 보수적인 0.025가 필요한가 |
| `inference.alternative` | H1 방향 | **`greater`** | A2 개선만을 사전가설로 둘지, 양방향 검정을 요구할지 |
| `success_threshold` | learned/retained/forgotten descriptive cell의 BA pass 기준 | **운영 목표로 별도 승인** | 이 값은 primary p-value에 쓰지 않지만 capability 해석을 바꿈 |
| `clean_a2_mean_minimum_delta` | 세 fold 평균 Full A2−A0 방향 gate | **0.00 동결** | tie는 “개선”이 아니라 관찰 평균이 음수가 아님을 뜻함 |
| `clean_a2_subject_harm_margin` | 개별 observed subject의 clean A2 최대 허용 손실 | **0.03 동결** | 명시적 감사값이며 아래 두 wrong-metadata gate 조합에서는 수학적으로 중복 |
| `shortcut_equivalence_margin` | metadata-only BA가 chance를 넘는 observed upper screen | **0.03 동결** | 통계적 equivalence 주장이 아니며 세 subject 최대 excess를 사용 |
| `pairing_mechanism_margin` | correct A2가 shuffle-train 또는 inference block shuffle보다 커야 할 최소 BA 차 | **0.0125 동결** | 각 240-row fold에서 정답 3개 상당, 모든 fold에서 요구 |
| `counterfactual_mechanism_margin` | correct A2가 wet↔dry wrong metadata보다 커야 할 최소 BA 차 | **0.0125 동결** | 모든 fold에서 정답 3개 상당을 요구 |
| `wrong_metadata_safety_harm_margin` | wrong metadata가 A0보다 낮아질 수 있는 최대 BA 손실 | **0.03 동결** | 각 fold 최대 정답 7개 상당 손실 |
| `minimum_bundle_changed_fraction` | shuffle donor가 external bundle을 실제 바꾼 최소 비율 | **0.95 동결** | inference 228/240, training 456/480 이상 |
| `minimum_condition_flip_fraction` | counterfactual에서 dry↔wet이 실제 뒤집힌 최소 비율 | **1.00 동결** | 240/240 exact flip |
| `confirmatory_control_policy` | controls를 development gate로만 둘지 lockbox에서 한 번 더 할지 | **`development_gate_only` 동결** | reveal #2 결과가 confirmatory 실행을 자동 승인하지 않음 |

`inner_val_ratio: null`은 미결정값이 아니다. Confirmatory training에 validation·early stopping을 두지 않는 고정 계약이다.

## Control estimand 정의

값을 승인할 때 부호를 혼동하지 않도록 다음으로 고정한다.

```text
shortcut excess       = BA(metadata-only control) - chance BA
clean direction       = mean_subject[BA(clean Full A2) - BA(A0)]
clean subject harm    = max_subject[BA(A0) - BA(clean Full A2)]
pairing sensitivity   = BA(correct A2) - BA(block-coherent within-class shuffled)
counterfactual effect = BA(correct A2) - BA(wrong wet/dry metadata)
safety harm           = BA(A0) - BA(wrong wet/dry metadata)
```

Clean 방향은 세 fold 평균이 0 이상일 때 통과한다. 나머지 effect/safety gate는 least-favourable observed fold를 쓴다. BA 비교에는 정확한 3/240 경계가 부동소수 오차로 실패하지 않도록 `1e-12` tolerance를 기록한다. 이는 신뢰구간이나 모집단 gate가 아니라 **N=3 개발 진단**이며, 10개를 모두 통과해도 owner review와 confirmatory 봉인은 자동 해제되지 않는다.

## 승인 뒤 기계적 동결 순서

1. 위 값과 physical development reveal #2의 owner decision ID·승인 시각·중단 규칙을 hash-bound decision receipt와 append-only 연구일지에 남긴다.
2. Physical grid를 exact six-role×3-fold 18 jobs로 동결한다. **각 fold당 Full A2 intervention bundle 하나, 총 세 개**의 missing/shuffle/wet↔dry 개입, block-coherent donor 규칙, metadata-only shortcut control, `missingness_only=invalid_assay`도 같은 계약에 묶는다.
3. Clean commit과 annotated tag `physical-reveal2-freeze-20260901-r1`에서 canonical manifest를 다시 만들고 `prepare`/`status`를 확인한다.
4. 18개 학습을 모두 완료한 뒤 단일 `reveal` 명령이 18개 clean prediction, 3개 Full A2 intervention bundle, subject/intervention aggregate와 10-gate 판정을 모두 private staging에서 완성한다. Canonical manifest로 donor mapping·potency를 재구성하고 전체 file-tree hash를 검증한 뒤 tree와 staging parent를 fsync한다. 그 다음에만 digest를 ledger와 precommit receipt에 기록해 reveal #2를 소비하고, 디렉터리를 atomic rename·parent fsync로 공개한 뒤 별도 final publication receipt를 쓴다. Precommit 뒤 장애가 나면 같은 digest만 복구·공개할 수 있다. 분리된 `predict`·`aggregate` 경로는 허용하지 않는다.
5. 사전 margin으로 mechanism·shortcut·safety를 판정하고, 통과 여부와 관계없이 결과를 연구일지에 남긴다. 이 N=3 결과를 모집단 효과로 해석하지 않는다.
6. Confirmatory 진행을 승인하면 `primary_fairness_hash`를 최종 A0/A2 config에서 재계산하고, 전체 변경을 clean commit과 annotated freeze tag로 고정한다.
7. `source_lock.freeze_tag`와 `source_lock.implementation_contract_sha256`를 채우고 `plan.status=frozen` validator를 통과시킨다.
8. 새 source contract로 canonical 6-job confirmatory manifest를 `--replace-unexecuted-draft`로 한 번 교체한다. 기존 파일이 unauthorized dev draft이고 run/prediction/staging/reveal 산출물이 전혀 없을 때만 허용된다.
9. 39명에서 여섯 fixed-epoch training을 완료한다. 여섯 completion/final/train-metric/split/development-control hash와 A0/A2 pair-wide 초기상태·schema·split fairness가 맞기 전에는 lockbox reveal이 등록되지 않는다.
10. 60명 lockbox의 86,400 prediction rows를 staging에서 6/6 생성한 뒤 한 번에 공개·집계한다.

## 별도 권한 요청

- 현재 compact spectral primary와 네 P0 공개 데이터에는 계정·토큰이 필요 없다.
- Nakanishi exact-12 외부 복제는 저자 서면 재사용 허가가 필요하다.
- REVE secondary는 Hugging Face gated access와 `HF_TOKEN`이 필요하다.
- Dong2023은 CC BY-NC 4.0이므로 상업적 사용은 별도 허가가 필요하다.

이 세 외부 요청은 wearable primary의 필수 진입조건이 아니다. 반면 SESOI·alpha/alternative·operational threshold·multiplicity·fairness와 clean source freeze는 필수다. Physical development control 값과 `development_gate_only` 정책은 이미 동결됐다.

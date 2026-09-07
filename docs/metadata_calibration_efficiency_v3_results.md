# Metadata-assisted low-calibration SSVEP — V3 synthetic development 결과

기준일: 2026-09-07

상태: **`DEVELOPMENT_NO_GO` terminal / 9개 grid cell 중 eligible 0 / 선택 방법 없음 /
scientific lockbox·외부·사람 EEG outcome 미개봉 / exact V3 candidate 종료**

이 문서는 `metadata-calibration-efficiency-v3`의 synthetic development/model-selection 결과다.
사람 EEG 효능이나 모집단 효과를 검정한 결과가 아니며, 아래 사후 진단은 새 선택 기준이 아니다.
설계와 과거 recovery 이력은 [V3 설계](metadata_calibration_efficiency_v3_design.md), 전체 현재
상태는 [연구 프로토콜 상태](research_protocol_status.md)를 따른다.

## 한 문장 결론

동결한 P3 후보는 안전하게 abstain하고 불변식을 지켰지만, Q-only support update도 metadata
increment도 사전 효능 기준을 만들지 못했다. 9개 cell이 모두 같은 7개 필수 component에서
실패했으므로 선택 없이 V3를 종료했다.

이 결과를 baseline ceiling으로만 설명할 수는 없다. B2 strict-FBCCA A0의 평균 balanced
accuracy가 `0.5850694444444444`여서 개선 여지가 있었는데도 AQ eAUC gain은 정확히 0이었다.
현재 P3 posterior/residual 방향이 유효한 class evidence를 만들지 못했고, M은 AQ의 변화를
줄였을 뿐 없는 evidence를 새로 만들지 못했다.

## 실행·무결성 경계

- 실행 snapshot: commit `8b3ace24eb4160aa61f3e7bf2af7dc1b74905699`, tree
  `04895973c140787546d31a72437d8c90982b698f`
- 단회 development seed: `301269949`
- 규모: family당 48 participants, 8 metric families, 9 grid cells
- 결과 행: participant metric 88,992 + invariant 90 = 총 89,082
- Governed `resume`: `DEVELOPMENT_NO_GO` terminal
- 선택: `selected_grid_cell_id=null`; selected-method freeze 없음

Fresh separate process의 읽기 전용 `status`는 exit 0으로 다음 한 줄을 정확히 반환했다.

```json
{"command":"status","durable_phases_advanced":0,"next_command":null,"selected_grid_cell_id":null,"selection_status":"DEVELOPMENT_NO_GO","stage":"DEVELOPMENT_NO_GO","terminal":true}
```

Canonical development-v5 evidence는 다음과 같다.

| Artifact | File SHA-256 | Payload SHA-256 | 추가 관찰 |
|---|---|---|---|
| `development-bundle.json` | `17f20c00b59e71efe9e94f8942170aa943899457f2d038a4055262339ff01742` | `76c0c656d3d3b90c8c1746a0fb43505a319e6475857c726f99d04161f666cff3` | V5 source/runtime/retired-attempt binding |
| `context-reference.json` | `7b78b3092f5824b6e247d97f4b9cd9f9ec22dbfa74a182e2f32c34fb1c8cf3c8` | `c475e9d0ce4f8e436b49c50585eaee37965bba7c052eefdee862bc50f70a9253` | covariate-only reference |
| `development-start.json` | `1e7ab674a81e52c138207c538011f4e13d5ee41d0843cda960abed920886de99` | `f62b1b1f57588ceca0d501e5733d82d8548c6085cffe16166870ac8f7eac1851` | seed 사용 전 write-once receipt |
| `development-result.json` | `1af9f65910da1d753a30a957dfb5bbef7366b696f9f32d444bf95dd411afcc8c` | `341df8f21cadbd599ca2ef9fe15f09f1d354c0b997198f503ba4ec5c1b0760fd` | 201,080,608 bytes, mode `0400`, nlink 1 |

Result의 canonical path는
`/home/whwovy/v3-artifacts/metadata-calibration-efficiency-v3/development-v5/development-result.json`이다.
이 synthetic terminal 뒤 selected freeze, canary, future scientific seed/lockbox, external 또는 human
EEG artifact는 만들거나 열지 않았다.

## 9-cell 판정

모든 cell은 동일한 complete data와 사전 교집합 gate로 평가됐다. 중앙 cell
`p3-nu_4-lambda_0p20`은 아래 진단을 설명하기 위한 예일 뿐 선택된 cell이 아니다.

| Grid cell | `nu` | `lambda_max` | Eligible | Selected | 실패 component 수 |
|---|---:|---:|---:|---:|---:|
| `p3-nu_1-lambda_0p10` | 1 | 0.10 | no | no | 7 |
| `p3-nu_1-lambda_0p20` | 1 | 0.20 | no | no | 7 |
| `p3-nu_1-lambda_0p30` | 1 | 0.30 | no | no | 7 |
| `p3-nu_4-lambda_0p10` | 4 | 0.10 | no | no | 7 |
| `p3-nu_4-lambda_0p20` | 4 | 0.20 | no | no | 7 |
| `p3-nu_4-lambda_0p30` | 4 | 0.30 | no | no | 7 |
| `p3-nu_16-lambda_0p10` | 16 | 0.10 | no | no | 7 |
| `p3-nu_16-lambda_0p20` | 16 | 0.20 | no | no | 7 |
| `p3-nu_16-lambda_0p30` | 16 | 0.30 | no | no | 7 |

각 cell에서 실패한 component는 정확히 다음과 같다.

- `A_Q_viability_B1_or_B2`
- `B3_metadata_efficacy`
- `B4_metadata_efficacy`
- `B4_in_reference_metadata_efficacy`
- `B4_interface_scale_value`
- `k3_pairing_mechanism`
- `pairing_potency`

나머지 named component인 `invariant_suite`, `B4_source_range_stress_safety`,
`B3_B4_deployment_viability`, `N1_null_noninferiority`, `N4_null_equivalence`, `severe_harm`,
`adversarial_abstention`, `helpful_use_anti_triviality`는 모두 통과했다. 이는 fallback과 harm
guard가 작동했다는 뜻이지, metadata efficacy가 있다는 뜻은 아니다.

## 중앙 cell을 쉽게 읽으면

아래는 `p3-nu_4-lambda_0p20`의 대표 mean이다. B1–B3의 nominal condition은 `neutral`,
B4는 participant 안에서 `wet`과 `dry`를 동일 가중한 `equal_condition_composite`다. A0 selector는
`role=A0, control=none, gate=none`; k1 AQ는
`role=A_Q, control=none, gate=deployed, budget=1`; k1 AQM은
`role=A_QM, control=correct, gate=deployed, budget=1`이다.

| Family | Condition | Role/control/gate | Budget | Mean balanced accuracy | Mean correct log probability |
|---|---|---|---:|---:|---:|
| B1 participant-class confusion | `neutral` | A0 / `none` / `none` | any A0 row | `.6993055555555552` | `-1.0035586253248951` |
| B2 phase/spatial shift | `neutral` | A0 / `none` / `none` | any A0 row | `.5850694444444444` | `-1.4251663473440142` |
| B3 impedance-linked shift | `neutral` | A0 / `none` / `none` | any A0 row | `.7791666666666668` | `-.9662263132643306` |
| B4 interface-calibrated shift | `equal_condition_composite` | A0 / `none` / `none` | any A0 row | `.8520833333333333` | `-.8568904556129896` |
| B3 impedance-linked shift | `neutral` | AQ / `none` / `deployed` | 1 | `.7791666666666668` | `-1.012439675229234` |
| B3 impedance-linked shift | `neutral` | AQM / `correct` / `deployed` | 1 | `.7791666666666668` | `-.9851483059647562` |
| B4 interface-calibrated shift | `equal_condition_composite` | AQ / `none` / `deployed` | 1 | `.8520833333333333` | `-.9048708890354323` |
| B4 interface-calibrated shift | `equal_condition_composite` | AQM / `correct` / `deployed` | 1 | `.8520833333333333` | `-.8796512351336911` |

k1에서 AQM은 AQ보다 correct-class probability를 덜 손상시켰지만 두 family 모두 A0보다 log
probability가 나빴고 balanced accuracy는 A0와 정확히 같았다. 즉 M이 confidence를 실제로
바꾸기는 했지만 class decision이나 calibration burden을 개선하지 못했다.

Deployed k3/5에서는 B1–B4 모두 M-free prequential gate가 support를 끄고 exact A0로 돌아갔다.
이것은 안전한 abstention이다. 그러나 gate를 강제로 켠 사후 진단에서도 의미 있는 balanced
accuracy gain은 없고 log probability는 A0보다 나빴다. 따라서 이 no-go는 유익한 효과를 gate가
가린 결과로 해석할 수 없다.

## 왜 필수 효능 gate가 실패했는가

- B1의 cell 전체 최대 AQ eAUC gain은 `.0001736111111111128`, 그 one-sided LCB는
  `-.00011769561139615749`였다. B2 AQ eAUC gain은 정확히 0이었다. 사전 기준 mean `>=.01`과
  LCB `>0`에 미치지 못했다.
- B3/B4의 metadata efficacy와 correct-vs-shuffle mechanism efficacy는 0이었다.
- B4 pairing potency는 changed fraction `.49166666666666664<.50`, median
  `.04830195627485313<.05`로 두 기준에 모두 조금 못 미쳤다. B3 potency 자체는 통과했지만
  B3/B4 교집합인 전체 `pairing_potency`는 실패했다.
- B4의 correct interface scale이 pooled/wrong-interface scale보다 유용해야 한다는 component도
  실패했다.

이 패턴은 baseline ceiling보다 **현재 support representation과 update 방향의 실패**에 가깝다.
P3는 normalized class-score vector와 JSD에 의존하므로 B2의 spatial/phase 정보를 충분히 담지
못한다. 또 현재 AQM의 scalar `g_M`은 A0–AQ 선분 위에서 AQ residual을 한 번에 줄이므로 block별
affinity 차이를 보존하지 못한다. 이 두 설명은 결과 뒤의 원인 가설이지 새로 동결된 방법은 아니다.

## 주장할 수 있는 것과 없는 것

주장할 수 있다.

- 이 exact synthetic DGP와 사전 gate에서 현재 P3/AQM 후보는 선택 자격을 얻지 못했다.
- M은 k1 probability를 바꿨지만 BA나 labeled-calibration 효능을 만들지 못했다.
- k3/5 gate, null·harm·OOD fallback과 anti-triviality 검사는 의도대로 작동했다.
- 0/9 규칙에 따라 scientific lockbox와 사람 EEG를 열지 않은 것은 사전 설계에 부합한다.

주장할 수 없다.

- interface/impedance metadata 일반이 SSVEP에서 무용하다는 결론
- 실제 사람이나 새 장치에서의 효과, 인과효과 또는 OOD 일반화
- 이 synthetic development를 human confirmatory evidence로 해석하는 것
- 중앙 cell이나 다른 cell을 사후 선택해 scientific candidate로 승격하는 것

## 다음 연구 루프

상위 목표—**external pre-query acquisition context M이 EEG-derived Q를 넘어 labeled target
calibration 부담을 줄이는가**—는 유지한다. 하지만 exact V3 candidate는 종료한다. 같은 result를
보고 threshold, lambda, DGP를 바꾸거나 다른 seed로 재시도하지 않는다.

다음 후보는 새 candidate ID, 새 scientific revision, 새 미관측 development evidence와 gate를
먼저 사전등록해야 한다. 순서는 (1) waveform/spatio-temporal 정보를 보존하는 강한 AQ가 matched
condition에서 실제 calibration utility를 만드는지 확인하고, (2) 그 뒤 metadata-selective trust의
순증분을 검정하는 것이 타당하다. Continuous log probability와 margin은 원인 진단에 쓸 수 있지만,
balanced accuracy와 실제 labeled-block/trial burden은 confirmatory endpoint로 남긴다.

한 가지 후속 가설은 block별 affinity를 유지하는
`p0 + lambda_Q * sum_b pi_b * a_qb * (p_s,b - p0)` 형태다. 모든 `a_qb=1`이면 AQ, k=0이면
A0가 되도록 설계할 수 있다. 이는 사후 가설일 뿐 V3 결과가 동결하거나 승인한 후속 V4 설계가
아니며, 새 데이터와 함께 별도로 사전등록·반증해야 할 제안이다.

계산 최적화도 과학 변경과 분리한다. 현재 workload는 Python의 작은 배열, control,
derangement, hash/audit 비중이 커서 단순 CUDA 이전의 이득이 작다. 먼저 whitening/reference 재사용,
batched CCA, nu/cache/lambda 재사용과 participant 단위 deterministic CPU multiprocessing을
프로파일링한다. GPU를 쓰려면 별도의 deterministic numerical/RNG/runtime 계약을 새로 동결해야 한다.

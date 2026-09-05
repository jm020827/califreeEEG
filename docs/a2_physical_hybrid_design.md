# A2 physical-hybrid 설계 계약

기준일: 2026-09-01
상태: **S1–S3 reveal #2 완료 — valid physical assay / diagnostic no-go / confirmatory 차단**

> **2026-09-04 superseded / historical only:** `physical_hybrid_v1`은 retired no-go이며
> 이 문서는 새 metadata-assisted calibration-efficient 후보의 계약이 아니다. 현재 질문과
> 실행 경계는 [새 설계](metadata_calibration_efficiency_design.md)와
> [프로토콜 상태](research_protocol_status.md)를 따른다.

이 문서는 당시 primary A0/A2의 모델 계약을 설명한다. 첫 S1–S3 grid의
`prompt_adapter_v1` 결과는 역사적 개발 증거로 그대로
남기며, 새 `physical_hybrid_v1`의 성능 근거로 재해석하지 않는다.

## 한 문장 설명

A0와 A2는 같은 spectral decoder를 쓰되, A2만 이미 관측된 wet/dry interface와
block impedance를 작은 bounded residual로 사용한다. A0 또는 metadata missing
상태에서는 그 residual이 수학적으로 정확히 꺼진다.

```text
현재 query EEG ── spectral channel token ───────────────────────┐
                                                               │
공통 query QC ── zero-init bounded FiLM ────────────────────────┤
                                                               ├─ classifier
A2 global: electrode type + impedance mean/max ── bounded FiLM ┤
                                                               │
A2 channel: channel별 impedance + availability ── bounded gain ┘

A0: 두 A2 branch의 availability를 0으로 강제
```

쉽게 말하면 global branch는 “이번 측정은 dry인가 wet인가, 접촉 임피던스의 전체
수준은 어떤가”를 알려 준다. Channel branch는 “8개 센서 중 어느 센서의 접촉 상태가
어떤가”를 각 센서의 **신호 token에만** 반영한다. Channel ID embedding은 줄이지
않으므로 센서 정체성과 신호 품질을 섞지 않는다.

## 정확한 입력 권한

| 묶음 | 입력 | A0 | A2 | 의미 |
|---|---|---:|---:|---|
| 공통 signal/structure | 현재 EEG, canonical channel ID/mask, sampling/time grid | 사용 | 사용 | 두 arm 공통 |
| 공통 query QC | 현재 query의 pre-zscore signal std | 사용 | 사용 | target의 다른 trial·batch 통계는 금지 |
| global external | `electrode_type`, block impedance mean/max, availability | null | observed | interface/session setup bundle |
| channel external | canonical slot별 impedance와 availability | null | observed | shared non-monotonic gain |
| 분석 전용 | headband order, first/second period | 금지 | 금지 | confound 분석에만 사용 |
| primary 제외 | reference, cap type | 금지 | 금지 | wearable에서 상수라 효과 식별 불가 |
| 금지 proxy | dataset/hardware/subject/session/trial ID, label/frequency/phase, source file | 금지 | 금지 | shortcut·누출 차단 |

Impedance scalar는 raw per-channel vector의 finite value에서 결정적으로 다시 계산된다.
모델 입력에서는 `log1p(kΩ) / log1p(100)` 후 `[0,2]`로 clip하는 기존 v1 transform을
사용한다. 이 transform은 물리 법칙이 아니라 동결해야 할 모델 입력 규칙이다. 높은
impedance가 언제나 나쁘다는 단조 규칙은 넣지 않는다.

## 모델 계약

1. `physical_hybrid_v1`은 legacy prompt token과 conditioned adapter를 primary에서
   사용하지 않는다. 안정적인 role ID 때문에 `A2_structured_condition_prompt`라는
   artifact 이름은 남지만 실제 architecture는 prompt가 아니다.
2. A0와 A2는 module graph, parameter key/shape/count, 초기 trainable state가 같고
   `external_metadata_mode=null|observed`만 다르다.
3. Global 및 query branch는 bounded residual FiLM이다. FiLM 최종 projection의
   weight와 bias는 0으로 초기화한다.
4. Channel branch는 모든 channel에 같은 scorer를 적용하고
   `gain = 1 + availability × ρ × tanh(score)`를 사용한다. 현재 `ρ=0.25`라 gain은
   항상 `[0.75, 1.25]`다. Score 마지막 층도 0으로 초기화한다.
5. External-null, unknown categorical ID, all-missing field, inactive channel은 학습 뒤에도
   각각 residual 0 또는 gain 1이다. Batch에 stale value가 남아 있어도 model-side null
   mask가 다시 차단한다.
6. 같은 weights라면 A2 all-missing과 그 checkpoint의 external-bypass signal path가
   정확히 같다. **별도로 학습한 A2와 별도로 학습한 A0의 성능이 같다는 뜻은 아니다.**
7. Query QC FiLM은 A0/A2 공통이다. External branch와 분리해 primary contrast가 외부
   metadata access만 나타내게 한다.
8. Spectral/Tiny backbone만 channel gain을 지원한다. Channel 대응이 보장되지 않은
   REVE와 physical primary 조합은 fail-fast한다.
9. Architecture key가 없는 예전 checkpoint/config는 `prompt_adapter_v1` graph를 그대로
   만들므로 첫 grid와 legacy secondary를 재현할 수 있다.

## 무엇을 식별하고 무엇을 식별하지 않는가

Primary가 성공해도 방어 가능한 주장은 좁다.

> 같은 wearable 장비·montage·protocol에서 source에서 이미 본 wet/dry 두
> interface를 사용하고 처음 보는 participant를 strict k=0으로 분류할 때, query 전에
> 관측한 interface/session setup과 block impedance의 global/spatial 표현을 제공하는
> 것이 external-null decoder보다 predictive accuracy를 높인다.

`electrode_type`은 이 asset에서 session과 함께 움직이고, impedance도 interface와 강하게
결합된다. 따라서 wet/dry의 독립 인과효과, impedance 하나의 인과효과, reference/cap
효과, 처음 보는 hardware/site/interface로의 일반화는 이 비교로 주장하지 않는다.

## 두 번째 development assay

첫 prompt grid와 사후적으로 섞지 않는다. 새 질문은 “physical routing이 실제로
metadata pairing을 사용하면서 결측·오입력에 안전하고, metadata만으로 label을
맞히는 shortcut이 없는가”다. 학습 matrix는 다음 여섯 역할이다.

| 역할 | EEG 경로 | Global FiLM | Channel gain | 학습 metadata |
|---|---:|---:|---:|---|
| A0 | 사용 | off/null | off/null | external-null |
| Full A2 | 사용 | on | on | correct |
| Global-only | 사용 | on | off | correct |
| Channel-only | 사용 | off | on | correct |
| Full-shuffle-train | 사용 | on | on | acquisition-block 단위 donor metadata로 학습, clean metadata로 평가 |
| Metadata-only | 차단 | on | on | correct; label/session shortcut 진단 |

자연 missingness pattern은 현재 하나뿐이어서 missingness-only 결과를 학습·해석하지
않고 `invalid_assay`로 기록한다. 전체 개발 grid는 seed 42, 3개 LOSO fold의
`6 roles × 3 folds = 18 jobs`이며 모집단 추론을 금지한다.

Full A2 checkpoint에는 재학습 없이 all/global/channel/electrode/summary/query-QC missing,
block-coherent shuffle, joint wet↔dry swap을 적용하는 manifest-bound 평가 경로가 있다. Shuffle은 한
acquisition block의 12 label을 서로 다른 donor로 찢지 않고, donor block 하나를 선택한
뒤 같은 label/window 행끼리 교환한다. **각 fold당 Full A2 intervention bundle 하나, 총 세 개**의 artifact는 18개 clean
prediction과 같은 hidden staging bundle에서 완전성·hash를 확인한 뒤 한 번에 공개한다.
승인과 수치 기준은 `configs/governance/wearable_physical_reveal2_decision.json`에
hash-bound로 동결한다. Outcome은 prediction·intervention·aggregate·gate를 모두 private
staging에서 완성하고 canonical manifest에서 donor mapping·potency를 재구성한 뒤 tree hash를
검증한다. Tree와 staging parent를 fsync한 다음 그 digest의 reveal-budget precommit을 먼저
기록하고 단일 디렉터리 atomic rename·parent fsync로 공개한 뒤 final receipt를 쓴다.
Precommit 뒤 복구에서는 같은 digest만 허용한다.

S1–S3는 모집단 효과 검정에 너무 작다. 두 번째 reveal을 승인하더라도 다음만 판정한다.

- 세 fold 평균 clean Full A2−A0가 음수가 아닌가
- algebraic neutral·정보 경로·mapping audit가 통과하는가
- correct Full이 shuffled/wrong metadata에 실제로 민감한가
- all-missing/wrong metadata가 사전 허용 harm margin을 넘지 않는가
- global-only와 channel-only 중 어떤 경로가 작동 가능성을 보이는가

정확한 margin, 중단 규칙, output root와 단일 reveal 절차는 `DEC-20260901-004`로
고정했다. Source commit `7bb8afc64c02e45beda7e245370563eac0e021e2`와 annotated tag
`physical-reveal2-freeze-20260901-r1`에서 18개 job과 세 intervention bundle을 완성하고
reveal index 2를 공개했다.

## Reveal #2 판정

| 지표 | 관찰값 | 판정 |
|---|---:|---|
| mean A0 BA | 0.1292 | descriptive |
| mean Full A2 BA | 0.1194 | descriptive |
| mean Full−A0 | −0.0097 | clean directional gate 실패 |
| mean Global-only−A0 | −0.0083 | descriptive |
| mean Channel-only−A0 | +0.0014 | descriptive |
| mean Full−shuffle-train | 약 0 | training pairing gate 실패 |
| metadata-only excess over chance | 0 | shortcut screen 통과 |

Counterfactual flip과 train/inference donor-bundle potency는 모두 fold별 `1.0`이라 assay는
유효하다. Clean observed-subject harm와 wrong-metadata safety screen도 통과했다. 그러나
`clean_a2_directional_mean`, `counterfactual_reliance`, `inference_pairing_shuffle`,
`training_pairing_shuffle` 네 substantive gate가 실패했다. 따라서 결과는
`diagnostic_no_go_one_or_more_substantive_gates_failed`다. N=3이므로 모집단 효과의 부재나
안전성을 주장하지 않지만, 현재 physical routing이 correct metadata 이득·의존성을
보였다고도 주장할 수 없다.

## 현재 구현과 중단 경계

구현 파일은 `src/cfeg/models/physical_conditioning.py`, primary config는
`configs/train/wearable_loso.yaml`, 개발 matrix는
`configs/train/wearable_physical_mechanism.yaml`이다. Block-coherent donor 구현은
`src/cfeg/data/metadata_controls.py`에 있다.

현재 상태는 다음과 같다.

- [완료] owner가 두 번째 S1–S3 assay와 reveal 사용을 승인하고 mechanism/safety margin을 고정
- [완료] metadata provenance, canonical donor reconstruction, six-role/18-job·3-bundle 계약과 atomic publication 검증
- [완료] clean execution commit/tag와 18/18 training, 3/3 intervention, public receipt/ledger 검증
- [완료] potency-valid assay에서 네 substantive gate 실패와 diagnostic no-go 적용
- [금지] 동일 S1–S3 threshold 조정, 재학습, 추가 reveal 또는 결과 기반 architecture 선택
- [차단] 현재 `physical_hybrid_v1`의 39명 training·60명 lockbox prediction
- [새 연구 필요] 독립 development data와 외부 근거로 새 후보·gate·decision contract를 결과 보기 전에 정의

SESOI·alpha/alternative·operational threshold·multiplicity·fairness/source lock을 채우는 것만으로
이 no-go를 해제할 수 없다. 39명 training이나 60명 lockbox prediction은 별도의 새 연구설계가
사전등록되고 독립 개발 근거를 통과하기 전까지 허용되지 않는다.

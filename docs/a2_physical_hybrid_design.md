# A2 physical-hybrid 설계 계약

기준일: 2026-08-30
상태: **DEC-20260831-003으로 S1–S3 reveal #2 승인·기준 동결 / 실행 전 source freeze / confirmatory 미동결**

이 문서는 primary A0/A2의 현재 모델 계약을 설명한다. 연구 질문과 39명
training·60명 lockbox 배정은 바꾸지 않는다. 바뀐 것은 metadata를 모델에 넣는
방법이다. 첫 S1–S3 grid의 `prompt_adapter_v1` 결과는 역사적 개발 증거로 그대로
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
staging에서 완성하고 tree hash를 검증한다. 그 digest의 reveal-budget precommit을 먼저
기록한 뒤 단일 디렉터리 atomic rename·parent fsync로 공개하고 final receipt를 쓴다.
Precommit 뒤 복구에서는 같은 digest만 허용한다.

S1–S3는 모집단 효과 검정에 너무 작다. 두 번째 reveal을 승인하더라도 다음만 판정한다.

- 세 fold 평균 clean Full A2−A0가 음수가 아닌가
- algebraic neutral·정보 경로·mapping audit가 통과하는가
- correct Full이 shuffled/wrong metadata에 실제로 민감한가
- all-missing/wrong metadata가 사전 허용 harm margin을 넘지 않는가
- global-only와 channel-only 중 어떤 경로가 작동 가능성을 보이는가

정확한 margin, 중단 규칙, output root와 단일 reveal 절차는 `DEC-20260831-003`으로
고정했다. 현재 남은 실행 전 gate는 전체 source를 clean commit과 annotated freeze tag로
결합하고 테스트를 다시 통과하는 것이다.

## 현재 구현과 남은 gate

구현 파일은 `src/cfeg/models/physical_conditioning.py`, primary config는
`configs/train/wearable_loso.yaml`, 개발 matrix는
`configs/train/wearable_physical_mechanism.yaml`이다. Block-coherent donor 구현은
`src/cfeg/data/metadata_controls.py`에 있다.

Confirmatory 진입 전 남은 필수 항목은 다음이다.

- [완료] owner가 두 번째 S1–S3 assay와 reveal 사용을 승인하고 mechanism/safety margin을 고정
- metadata 측정 시점·단위·transform과 block alignment provenance를 최종 서명
- 구현된 six-role/18-job+각 fold당 Full A2 intervention bundle 하나(총 세 개), 전역 event-hash ledger, root-wide lock·atomic publish의 outcome-free 검증은 완료
- prediction→checkpoint/config/completion/reveal provenance, main/intervention clean 교차검증과 least-favourable N=3 gate 적용은 완료
- clean commit/tag 및 최종 implementation hash
- SESOI, alpha/alternative, operational threshold, confirmatory control policy 확정
- 승인된 두 번째 개발 결과가 freeze gate를 통과할 때만 39/60 plan을 `frozen`으로 변경

39명 training이나 60명 lockbox prediction은 이 문서 추가만으로 허용되지 않는다.

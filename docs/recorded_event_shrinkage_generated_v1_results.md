# Recorded-event shrinkage v1: 생성 학습 결과

2026-09-13. **One-shot에서 metadata가 template 연산과 query score를 바꾸는 학습
경로는 작동했다. 실제 acquisition metadata의 유용성은 아직 검증하지 않았다.**
[사전 계획](recorded_event_shrinkage_generated_v1_plan.md),
[모델·예측·수치 원문](reports/recorded_event_shrinkage_generated_v1_run/child.json),
[종료 영수증](reports/recorded_event_shrinkage_generated_v1_run/terminal.json).

계획13c65ec, 구현53e77ec를 실행 전에 고정했다. Positive/null/schedule-only
3세계×Q/Q2/QM/SHAM 4개, 총12 ridge fits를 단회 완료했다. 각각의 시작/완료
영수증12개씩, 모델 scaler/beta/intercept와 eval prediction을 저장했다. 재시도0.
이 단계의 실제 EEG/participant/held60/외부요청/유료0. 직전 S001 입력 확인과 별개다.

## 결과

아래는 정확도가 아니라 **반복 template에 대한 episode별 Frobenius 오차의 평균**이다.

| 생성 세계 | Q | Q2 | QM | SHAM |
|---|---:|---:|---:|---:|
| M에 정답 혼합량 정보가 있음 | 0.125 | 0.125 | 0.000000304224 | 0.125 |
| M과 정답 혼합량이 무관 | 0.125 | 0.125 | 0.125 | 0.125 |
| 정수 frame schedule만 다름 | 0.125 | 0.125 | 0.125 | 0.125 |

동일한 one-shot template와 query에서 M만 바꾸자 learned λ는
0.000780→0.999220, score는0.999610→0.500390으로 바뀌었다. Trial 가중치를
정규화해 한 trial에서 효과가 사라지는 경로가 아니다. T=P일 때 metric/score가
같고 H=C인 순수 reference query에서는 score1이 유지되는 상쇄 대조도 통과했다.

Timestamp부터 실제 M extractor까지 연결했다. 같은 잔차에서 정수 frame schedule/
절대 시각만 바꿔도 M이 같았다. SHAM은 두 특징을 함께 교환하고 positive/null의
2×2 교차표를 각칸8개로 유지했다. Metadata vector 절반이 바뀌고 multiset은 같았다.
Schedule-only의 M은 수치 floor 이후 모두0이었다. 총27개 사전 invariant PASS.

## 이것이 증명하지 않는 것

- Q/Q2의 입력을0으로 둔 작은 세계다. **실제 강한 EEG-Q를 이긴 실험이 아니다.**
- Positive는 설계자가 M과 정답 shrinkage의 관계를 넣었다. 이 관계가 MAMEM에
  있다는 증거도, 손상된 EEG가 필요하다는 새 연구목표도 아니다.
- Train32/eval16은 동일 상태 조합이 반복되는 결정론적 fixture다. 독립 일반화,
  OOD 발견, 통계적 유의성, 분류 정확도·보정량 감소를 검증하지 않는다.
- 행렬 혼합의 기작만 구현했다. 실제 EEG→harmonic template, 강한 Q/Q2,
  source 참가자 제외 prior/다른 repetition 표적, trial/class label 경계는 남아 있다.
- 단순 공통 phase shift나 순수 배율 변화는 정규화된 spatial PSD에서 상쇄될 수
  있다. 그러므로 ‘timing metadata니까 반드시 도움이 된다’고 주장하지 않는다.
  실제 source에서 Q/common 정보를 조건화한 뒤 반복-template 오차를 더 예측하는지
  확인해야 한다. 알려지지 않은 물리적 jitter를 가정해 양성을 구제하지 않는다.

## 유지와 다음 행동

**유지:** k1에서도 작동하는 PSD template shrinkage 연산과 두 M의 제한된 후보.
**승격하지 않음:** 유망한 실제 효능 후보/보정 절감 입증. 기존 부정 결과는 불변.
**다음:** [실제 source 학습 연결 초안](mamem_recorded_event_source_learning_v1_draft.md)의
하나의 구현·검증 계약으로 연결한다. 불명확한 row257의 정체나 광학 원인을 끝없이
찾는 것이 선행조건은 아니다. 실제 class/split/strong-Q 및 비용 경계를 먼저 고정한다.

새 algebra/보고서 검사16개는 전체12fit을 재호출하지 않는다. 직전 입력/기존 관련
검사까지 합쳐147tests PASS. 실제 모델의 추가 재학습은 없다. 독립 감사자는 production
함수 import나 solve 없이 저장12모델의192eval prediction과2개 M 개입,6scores,
27invariants를 재계산했다. 최대 prediction차2.22e−16, loss차2.78e−17,
정규방정식 residual0으로 일치했다. 이것은 저장 수치 감사이지 실제 효능 감사는 아니다.
별도의 최종 protocol 감사도 producer/helper/config·53e77ec·24개fit영수증·문서/상태를
확인해 PASS했다. 전체 시도0.115395초, 출력16,341bytes, stdout/stderr0bytes다.

# 같은 시간중심 점수에서 metadata의 저보정 기여를 시험하는 설계

2026-09-09. [고정 JSON](../configs/analysis/task_trca_temporal_v1_design.json),
[구현·소유권 계약](task_trca_temporal_v1_build.md).
**이번 승인 범위는 설계·실제 학습기 연결·생성 배열 검증이다. 사람 실험 실행 계획이 아니다.**

## 쉽게 말하면

질문은 그대로다. 새 사용자가 정답이 붙은 EEG 예시를 적게 제공해도 잘 분류하게 만들 수 있는가?
여기서는 EEG에서 만든 Q 특징에 숫자 임피던스를 추가한 QM이 더 도움이 되는지 묻는다.
Q도 공통 결측 패턴과 측정 순서 정보를 받으므로, 정확히는 **Q+공통 정보 너머 숫자 임피던스의
추가 효과**다. 모든 metadata 대 metadata 없음의 비교나 최초 few-shot 연구가 아니다.

지난 [부호 독립 검증](task_trca_shape_signfree_v1_engineering.md)은 계산 부품만 검증했다.
이번에는 그 부품을 실제15특징 Q학습·동결·metadata 잔차·참가자 분리 선택에 연결한다.
임의의 band/channel별 자유 logits를 학습했던 작은 시험을 실제 학습기 검증으로 바꾸어 부르지 않는다.
기존 사람 후보의 VALIDITY_FAILURE와 이전 부정 결과는 보존하며, 실패 조건을 완화해 재개하지 않는다.

## 무엇을 같게 두고 비교하나

| 방식 | 다른 점 | 점수 방식 |
|---|---|---|
| FULL_NATIVE | 기존 gamma0 필터/템플릿 그대로 | 원 global Pearson |
| FULL_CENTERED | FULL_NATIVE의 정확히 같은 필터/템플릿 | 새 성분별 시간중심 Pearson |
| ISO | 고정 동일량·등방 규제, 학습 없음 | 새 점수 |
| Q | Q15특징으로 채널 규제 모양 학습 | 새 점수 |
| Q2 | 고정 Q 위에 EEG특징2개의 잔차3계수 | 새 점수 |
| QM | 고정 Q 위에 숫자 임피던스특징2개의 잔차3계수 | 새 점수 |
| SHAM_REFIT | 같은 잔차 구조에 다른 사람의 짝지어진 M | 새 점수 |
| PERMUTED / STALE / MISSING | 올바른 QM 학습 뒤 M만 교란/과거값/제거 | 새 점수 |

A0는 보정0개인 원 native reference로 별도 유지한다. FULL_CENTERED에는 새 학습·S/C변경·
필터 재정규화가 없다. 따라서 두 FULL의 차이가 점수 정의 변경 효과를 보여준다.
QM과 Q의 비교에는 같은 새 점수식과 같은 학습 조건을 적용한다. 유리한 FULL만 골라 비교하지 않는다.

## 학습·평가 규칙

- 기존 Q15와 숫자 M2 정의, k-prefix mask-only Q, available-only M/Q2 scaler, 공통 순서/
  결측 정보, 전처리/S/C/template를 유지한다. 점수에서 평균을 제거한다고 support를 새로 중심화하지 않는다.
- global Q16계수 하나를 학습하고 동결한 후, Q2/QM/SHAM 잔차를 각각3계수로 학습한다.
  각 head200steps, 초기값0, float64 full-batch Adam(.01,.9/.999,eps1e−8),
  loss=`CE(newscore/sum(native positive weights)/.1)+lambda*mean(coef²)`다.
- logits80%Q+20%residual, R합8/비율≤2, tau=.1lambda_min(C)/(16/9), C≤B≤1.1C를 유지한다.
  부호 대신 class별 C-normalized F를 쓰고 J=sumF로 점수를 계산한다. Topgap/SPD/분산 검사 유지,
  jitter·anchor threshold변경·예외를missing으로처리·실패사람제외·native fallback은 없다.
- 향후 개발39명은 outer3fold26/13, 안에서inner3fold17/18 대8/9로 분리한다.
  lambda={.0001,.001,.01} 선택은 Q validationCE만 사용하며, 동률1e−12이면 큰lambda다.
  매innerfit마다scaler/Q/잔차/donor를 처음부터 구축한다. 최대30pipelines/120heads/24000updates.
- Target 보정은 앞3/5블록(36/60정답예시)이다. Source 학습에만 별도 block5 정답을 쓴다.
  이것을 사용자 보정비용과 혼동하지 않고 source offline 비용으로 따로 기록한다.
  Outer평가자의block5 접근금지, 전모델동결전blocks6–9 최종query금지, held60범위밖은 그대로다.
- SHAM은각fit/validation/evaluation partition의조건/순서/정확한앞k결측패턴 안 cyclicdonor다.
  Q와scaler는동결한다. Singleton/변화없는M도분모에남기고 정식조건부randomization검정으로부르지않는다.
- 새 TaskCase/TemporalGramStatistics/Pipeline score_schema를 사용한다. 옛 global점수로 학습한
  checkpoint를 새점수 모델로 읽지 않는다. Q계수해시만 같아도 학습목표 동등성을 보장하지 않는다.

## 판정 — 기존 기준 유지, 새 대조군 기준 추가

Primary는39명의8조건 평균 QM3−Q3다. 다음을 모두 요구한다.

1. 평균≥1%p, 참가자 paired기술적95%CI하한>0.
2. QM3−Q2_3 및 QM3−SHAM3의 CI하한>0, sham M변화 coverage≥50%.
3. 보정량 후보에는 QM3−Q5의 CI하한>−1%p 및 QM3평균≥80%.
4. QM3−FULL_NATIVE3와 QM3−A0의 CI하한>−1%p라는 기존 기준 유지.
5. **새로 QM3−FULL_CENTERED3의 CI하한>−1%p도 요구.** 기존 gate를 완화하지 않고,
   같은 점수의 학습 없는 대조군에 대해서도 고정 −1%p 비열등성 margin을 적용한다.
6. Q대QM 참가자손해가이득보다많지않아야하며,둘다80%도달한cell의관측label절감합>0,
   새도달≥도달상실. 관측0/3/5에서48query중39정답 이상이도달;미도달비용은null이다.

R→개별F→J→score→argmax를 구분해 전달 변화를 남긴다. F변화는 효용 증거가 아니며,
argmax가 그대로라는 이유만으로 구조적 무작동이라고 하지 않는다. 원 enum 종료순서를 유지한다.
새 centered FULL gate는 기존 calibration candidate를 classification-only로 낮출 수만 있고
metadata 미확립을 성공으로 올리지 못한다. 모든결과는종료,후속eta/feature/window/seed자동탐색없음.

39명은 반복 노출된 개발자료다. Nested CV나 paired CI가 독립 확증자료로 바꾸지는 않는다.
좋은 결과도 외부 일반화·배포 stopping rule·wall-clock 절감·one-shot을 증명하지 않는다.

## 이번 구현 완료 조건과 이후 경계

고정seed20260909의생성배열로 실제4heads200steps의scalar/batch/CUDA일치,6명1조건nested8000updates,
모델저장/복원·label-free평가·native/centeredFULL·Q/MISSING동일·독립저장점수/선택검산을 확인한다.
두인공시간창N17/23,k3/5는연결검사용이며사람조건선택이아니다. 성공할seed/난이도를찾지않는다.
새source archive/실패진단JSON/실제숫자M/query/held60는 이번에도 읽지 않는다.

이후 실제 실행에는 새 reader/완료경로감사/코드·입력hash·자원한도가있는 별도manifest가 필요하다.
현재 pure API는실제자료의OS권한sandbox가아니며,생성배열통과를사람실험종료로보고하지않는다.
기존학술landscape와검증근거를재사용한다. 이번판단을바꿀불확실성은새문헌수보다
동일학습조건에서의구현정합성이므로 broadsearch/PDF추가없이직접검증한다.

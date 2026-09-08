# C2: 보정 반복 쌍이 공간필터를 가르치는 비중

2026-09-09. [고정 수식·특징 JSON](../configs/analysis/task_trca_pair_s_v1_design.json).
상위 [2후보 프로그램](metadata_learning_program_v1.md)의 두 번째 슬롯이다.
**C1의 새 사람 학습/최종 query를 보기 전에 고정한 가설이며, 구현·효능 결과는 아니다.**

## 무엇이 다른가

C1은 ‘어느 채널의 필터 계수에 규제를 줄 것인가’를 바꾼다.
C2는 ‘보정 반복 두 개가 비슷하다는 증거를 필터 학습에 얼마나 반영할 것인가’를 바꾼다.
앞3/5블록에는 각각3/10개의 서로 다른 블록 쌍이 있다. EEG 품질(Q)로 쌍별 가중치를 학습·고정하고,
숫자 임피던스의 상대적 블록 수준·블록 간 차이(M2)가 더 도움이 되는지 작은 잔차로 시험한다.

이전 context-template는 파형 평균을 가중했다. C2는 **평균 템플릿과 분모 C를 그대로 두고,
반복 쌍이 TRCA numerator S에 기여하는 양만** 바꾼다. 임피던스가 높으면 나쁘다고 강제하지 않는다.
새 삽입 위치이지, 문헌상 최초성이나 실제 효과의 증거는 아니다.

## 고정 연산

블록 i,j의 원 전처리 EEG 행렬을 X_i,X_j라 하면 A_ij=X_iX_jᵀ+X_jX_iᵀ다.
원 native S0,C를 보존하고, n개 쌍의 양의 가중치 p합을1로 유지한다.

`S = S0 + sum_pairs[(n*p−1)*A]`, `B=C`, 템플릿은 균일 평균이다.

구현은 u=exp(logit−maxlogit), d=(u−mean(u))/mean(u), S=S0+sum(d*A)를 쓴다.
균일 logits에서 원 S0를 정확히 돌려주면서도 초기 학습 미분을 끊지 않기 위해서다.
Q16계수→동결→Q2/QM/SHAM 각3계수,80/20logit예산,가중치비≤2,
200stepAdam와Q-only nested lambda 선택을 고정했다. 정확한15개 Q와2개 M 식·가중 scaler·결측은 JSON에 있다.

가중치 총량을 고정해도 trace(S), 필터 각도, 유효 독립 반복 수나 정확도가 보장되지는 않는다.
SPD/topgap/양의분산 검사와 실패 보존을 유지한다. C1의 분모10% bound를 C2에 갖다 붙이지 않는다.

## 대조군과 반증

Q/Q2/QM/SHAM은 같은 연산을 쓰고, MISSING은 고정 Q로 정확히 돌아간다. 부분 결측 pair도
정규화 때문에 최종 가중치가 달라질 수 있으므로 개별 pair exact fallback이라고 하지 않는다.
공통 순서·결측·블록 위치는 Q에도 준다. SHAM은 같은 분할·interface·순서·정확한prefixmask의
다른 사람 packet을 사용하며, 변화 없는 짝도78개 participant-interface coverage 분모에 남긴다.

학습 없는 대조군 **UNIFORM_C2**는 S0/C의 C-normalized projector와 새 점수를 사용한다.
원 native 필터의 상대 크기가 다를 수 있으므로 FULL_CENTERED와 같다고 가정하지 않는다.
원 FULL_NATIVE, 같은필터 FULL_CENTERED, A0를 함께 보존하고,
QM3−UNIFORM_C2_3의CI하한>−1pp 기준을 기존 calibration 기준에 추가한다.

같은 보정량에서 Q/Q2/SHAM을 이기는지와 실제0/36/60labels 관측 비용·80%도달·손해를 모두 평가한다.
Pair가중치→S방향→F→J→score→정답선택 변화를 구분한다. 공통 양수배 S 변화나
가중치 변화만으로 ‘학습에 도움이 됐다’고 결론내리지 않는다.

## 진행 조건

C1에서 모든 calibration 기준을 통과하면 C2는 실행하지 않고 유망 개발 후보의 후속 독립 확인을 준비한다.
C1이 유효하게 미확립/분류 이득만 보이면 위 명세 그대로 C2를 준비한다. C1의 수치실패가 있으면
그 원인과 C2 공유 여부부터 확인하며 실패가 곧 C2의 유리한 근거라고 하지 않는다.
C2 자체 구현/정합성 사전검증이 실패하거나 입력 대응이 없으면 슬롯을 기각하고 효능 미평가로 남긴다.
인간 실행은 최대30pipelines/24000updates 한 번, 이후 결과에 따른 특징·bound·seed 변경은 없다.

Source39는 탐색용이다. 두 후보를 미리 정해도 과거 수많은 실험의 노출 이력이 사라지지는 않는다.
좋은 결과는 추가 검증할 개발 후보이며, metadata 일반 효과나 배포·one-shot·wall-clock 절감의 확증이 아니다.

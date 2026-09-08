# C2 사전 구현 지도 — 아직 비활성

2026-09-09. C1 primary가 진행 중일 때 읽기전용 reviewer가 고정한 C2 수식의 코드 연결만 검토했다.
[수식 계약](task_trca_pair_s_v1_design.md) SHA1561541f…109ee3는 C1 사람 학습 전에 이미 고정됐다.
이 문서는 특징·손실·문턱·후보 예산을 바꾸지 않으며 C2를 활성화하지 않는다.
C1의 사전 terminal/eligibility 판단 뒤에만 구현과 별도 실행 명세를 집행한다.

## 최소 새 경계

| 새 모듈 | 소유할 일 |
|---|---|
| task_trca_pair_s_features | 명세의 Q15/M2, 실제 k별 pair좌표, availability와 fit-only 가중 scaler |
| task_trca_pair_s_operator | 정규화 pair질량→S, B=C의 S미분 projector, scalar/그룹 batch |
| task_trca_pair_s_learning | 새 PairTaskCase/Pipeline, Q동결 후 잔차, 고정200step/nestedQ선택 |
| task_trca_pair_s_evaluation | 평가 label 없는 support상태,10arms, 가변 pair직렬화 |
| task_trca_pair_s_audit | 독립 NumPy/SciPy pair/S/F/score/scaler/donor/선택/판정 |

실제 role-bound reader/실행기/cold는 별도 연결 단계다. 같은 cohort와 scorer라도 C1 실행 manifest를
C2에 재사용하지 않는다. C1의25pinned코드와 완료/실패 산출물은 변경하지 않는다.

## 재사용하면 안 되는 부분

- 기존 bounded_projectors는 S를 detached constant로 요구하고 denominator penalty를 넣는다.
  C2는 이를 호출하지 않고 unchanged leading_projector를 감싼 새 B=C 경로를 쓴다.
- S0/A/C/features/Grams는 constant, S와 H는 gradient를 유지한다. Cholesky(C)는 미리 계산할 수 있다.
  NumPy 변환·detach·torch.tensor(existing_tensor)로 S의 미분을 끊지 않는다.
- Uniform logits: u=exp(logit−max), d=(u−mean(u))/mean(u), S=S0+sum(dA).
  정확히 d=0이지만 gradient는 남아야 한다. Uniform이면 detached S0로 반환하는 branch를 넣지 않는다.
- 학습 시작 시 bias/공통logit방향 gradient가0인 것은 가능하다. 비퇴화 생성예의 전체 CE 경로에서
  contrast gradient와1차 유한차분을 검사하며, 모든 계수의 gradient가 반드시 비영이라고 요구하지 않는다.

## k3/k5와 패딩

**k3 pair는 k5의 첫3개가 아니다.** k3는(0,1),(0,2),(1,2), k5의 세 번째는(0,3)이다.
각 prefix의 실제 k에서 사전식 pair를 새로 만들며 k3 특징을 block3/4에서 만들지 않는다.

계산은 k별 그룹의 native pair폭3/10으로 나누는 것이 단순하다. 한 optimizer step의 CE는
각 그룹 case수/전체case수로 가중하고, regularization은 한 번만 더한다. 그룹평균을 무조건 반씩
섞으면 일부 inner partition이나 이후 표현에서 원래 case별 동일 objective를 바꿀 수 있다.

저장용 패딩은 별도의 표현 문제다. 최종 계약을 실행 전에 고정하되 가능한 최대폭10 배열은 다음과 같다:

- q[P,5,10,15], m[P,10,2], pair_cross[P,5,10,12,8,8]
- pair_valid[P,10], pair_coordinates[P,10,2], m_available[P,10], pair_weights[P,8,5,10]
- 숫자padding0/validfalse/좌표−1. Rawpacket5의 미래블록NaN 규칙은 유지한다.

Padding을 max/sum/mean/bounds/scaler/diagnostics에 넣지 않는다. M이 없는 **실제** pair는
여전히 정규화에 참여한다. 따라서 pair_valid와 m_available를 혼용하지 않는다.

## Scaler·대조군·감사

Q각 pid/interface/window/k/band 단위는 같은 질량, pair는1/n이다. Q2는[2,4]이며 C1의1:3이 아니다.
M은 window/band 중복의 packet/M 일치를 확인한 뒤 pid/interface/k별 한 번만 계산한다.
Available pair질량을 전체에서 정규화하며 각 사람의 available pair수를 제각각 재정규화하지 않는다.
Residual 전체 bias까지 gate하고 Q/scaler/logits를 동결한다. Empty fit도 정해진200steps를 유지한다.
STALE는 원 prefix mask를 보존하는 C2 명세를 사용하며 C1의 단순 첫block 반복을 복사하지 않는다.

독립 감사는 pair좌표/패딩→prefixM2→가중scaler/donor→pairweights→S→SciPyF→literaltemporal score→
UNIFORM_C2와 두 FULL/Q2/SHAM/정수판정/추가uniformNI까지 재구성한다. C1의 r라는 이름이나
8channel coverage축을 그대로 쓰지 않는다. C2 k3 M은[39,2,3,2], coverage분모는78단위다.
저장 Q/A/S0/C부터의 감사는 rawEEG→Q15/A/nativefit/Adam 독립 재현이 아님을 명시한다.

## 활성화 후 작업 분리

Root가 공통 schema/feature/operator/API와 최종 통합을 먼저 소유한다. 서로 독립적인 learner와
audit leaf만 기존 두 writingworktree를 재사용해 격리할 수 있다. 새 세 번째 tree는 예산 밖이다.
공통 계약이 준비되기 전에는 병렬 write하지 않으며, 최종 실제 경로·GPU·SQLite는 계속 root 단독이다.

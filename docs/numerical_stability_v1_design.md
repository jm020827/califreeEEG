# 별도 인공 수치 안정화 연구 v1

2026-09-09. 사용자 승인: 전체 goal의 추가 승인 요청에 대한 “응 승인.”
현재 상태: **사전 계약 고정 — 새 수치 실험0.** 정확한 수식·fixture·판정·예산은
[JSON 계약](../configs/analysis/numerical_stability_v1.json)이 기준이다.

## 범위와 문제 signature

- 표현: 대칭 S와 양의 정부호 B/C의8×8 generalized eigenproblem, leading rank-one
  C-normalized projector, S 또는 B를 통한1차 미분.
- 병목: 기존 두 번의 triangular solve로 생긴 raw H의 작은 비대칭이 고정 검사를
  넘었다. 기존 인공 C2 실패는 감사됐지만 대칭화된 다른 정책의 정확성은 미검증이다.
- 허용 연산: **별도 새 모듈**에서 최대2개 수치 계산 정책을 설계·검증한다. 기존
  C1/C2 코드·설계·실패·threshold는 변경하지 않는다.
- 목적: 원 generalized eigenproblem의 해·부분공간·미분을 신뢰할 수 있는지 확인한다.
  분류 성능·metadata 추가 이득·보정량 감소를 이번 단계의 산출물로 주장하지 않는다.
- 자원: 기존 로컬 환경, CPU1, GPU0, 설치·유료0. 수치 실험/검산 누적 최대7200초,
  테스트 누적 최대1800초, 새 출력1GiB, 새 worktree0(기존2개 재사용 가능).
- 피드백: 원 문제의 잔차/역오차, 독립 고정밀 기준 해, projector 차이,1차 방향미분,
  입력·top-gap 거절의 정확성. Query 정확도나 사람 효능은 피드백으로 사용하지 않는다.
- 실패 모드: 대칭화로 큰 입력 오류를 숨김, conditioning에 따른 해 오차, 작은 top-gap,
  비선행 고유값 중복 때문에 생기는 잘못된 autograd, 기준 계산과 구현의 공통 오류.
- 선택: N1의 명시적 대칭 Cholesky 환원과 N2의 CPU generalized eigensolve/암묵 미분이다.
  정해진 모든 조건을 통과하면 torch-native N1을 우선한다. 실패 뒤 수식·문턱을 바꾸지 않는다.

## 데이터·권한 보호

인공 행렬/인공 support만 새로 생성한다. 과거 C2의 **generated-only** 실패 산출물은
고정 해시와 허용 멤버를 먼저 계약에 기록한 뒤 알려진 회귀 사례로 사용할 수 있다.
그 사례는 새 미노출 검증으로 부르지 않는다. 사람 source/evaluation NPZ, 원 EEG,
숫자 M projection, query/labels, held60은 새로 읽지 않는다. 외부 요청 발송·유료 자원0.

## 조사 계획

`academic-research`의 기존 landscape/frontier를 재사용한다. 이번 좁은 질문의 검색은
foundation/implementation/contrary: 대칭 정부호 generalized eigen reduction, backward
error, leading spectral projector의 미분과 중복 고유값 한계다. Netlib/LAPACK 및
공식 라이브러리 문서·원 논문으로 수식/구현 근거를 확인하고, 새 효능 근거와 구분한다.
광범위 SSVEP 문헌 재검색은 이 수치 단계의 필수 작업이 아니다.

## 협업

Root가 계약·실제 실행·통합·SQLite를 단독 소유한다. 지금 agent들은 공유 저장소의
읽기전용 수식/fixture 검토만 수행한다. 최종 API 고정 후 두 writing lane이 필요하면
기존 temporal-runtime/cold worktree를 재사용하고 신규 파일별 소유권을 정한다.
현재 가용220,224,036KiB, 기존40worktrees를 보존하며 새tree/cleanup0이다.

## 근거와 사전 결정

[LAPACK 환원](https://www.netlib.org/lapack/lug/node54.html)은 Cholesky로 대칭 generalized
문제를 표준 문제로 바꾸는 수식을 설명한다. 그러나 [오차 설명](https://www.netlib.org/lapack/lug/node99.html)은
B가 ill-conditioned일 때 backward stability를 무조건 가정하지 못한다고 명시한다.
따라서 N1의 평균 대칭화는 새 검증 가설이지 LAPACK의 안정성을 그대로 상속한다는 주장이 아니다.
[SciPy1.15.3 eigh](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.linalg.eigh.html)의
고정 `gvd`와 B-normalized eigenvectors가 N2의 forward 기준이다. SciPy는 입력 대칭을
자동 검증하지 않으므로 두 방법 모두 자체 입력 검사를 수행한다.

[Greenbaum–Li–Overton, v2 §3](https://arxiv.org/html/1903.00785v2)의 reduced-resolvent
projector 미분은 목표 고유값이 분리됐을 때 나머지 고유값의 중복을 허용한다. 기존 custom
leading-projector도 이미 그 구조다. N2는 존재하지 않는 lower-root 버그의 수리가 아니라
독립적인 forward/미분 계산 경로다. 이번에는 공식 HTML의 해당 절·공식 문서/소스만 읽었으며
논문 전체 PDF 정독이나 새 SSVEP 효능 근거를 주장하지 않는다.

기존960개 인공 행렬을 알려진 회귀 사례로 보존하고, 새로운192개를 조건수·공통 scale·
top-gap·비선행 중복·B=C/다른B의 고정 grid로 만든다. 새 seed 둘은 구현 고정 뒤 처음 실행하며,
두번째 seed 결과를 보고 재학습/조정하지 않는다. 이는 인공 일반화 검사이지 독립 사람 확증이 아니다.
80자리 기준 해는 **저장된 실제 binary64 행렬**을 푼다. 생성 당시 의도한 고유벡터를 정답으로
대체하지 않는다. 작은 잔차로 틀린 고유벡터가 통과하지 않도록 최고 고유값과 C-metric
projector를 모두 비교하고,209개 사례의 S/B/C/공동 방향미분도 두 고정 step으로 검산한다.

새 모듈은 기존 분류 학습기에 연결하지 않는다. 사람 효능 프로그램에 연결하려면 generated
학습/CPU-CUDA/전체 reader·audit 검증과 새 사람 실험 권한·예산이 여전히 필요하다.

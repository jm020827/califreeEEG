# Task-shape v1: 학습기 구현·인공 검증 완료

2026-09-08. **학습 코어와 인공 검증은 완료했다. 사람 실험 준비 전체나 metadata 효능 검증은 아니다.**
원 [설계](task_aligned_trca_shape_v1_design.md)와 JSON은 바꾸지 않았다. 이전 부정 결과/held60 보호도 유지한다.

## 쉽게 설명하면

‘metadata가 채널별 공간필터 학습을 조금 바꾸게 한다’는 식을 실제 학습 코드로 만들었다.
Q를 먼저 분류 오차로 학습한 뒤 고정하고, Q2/QM/SHAM의 작은 잔차를 각각200번 학습한다.
인공 신호에서는 이 경로의 오차가 줄고 미분이 맞으며, metadata가 없으면 Q와 같은 결과가 나온다.
**실제 사람의 정답률이나 보정량이 좋아졌다는 뜻은 아니다.**

## 구현·확인 범위

| 항목 | 이번 확인 |
|---|---|
| Q/M 분리 | 숫자 M을 인자로 받지 않는 Q, support-prefix M, fit-only 불변 scaler와 partition-local donor |
| 연산자 | 공통 bounded mass, C 정규화/native anchor, repeated-lower-root를 허용하는 top-projector 1차 미분 |
| 점수 | 원 flattened global Pearson과 안정형 Gram의 값/gradient, band별 clip·signed weights |
| 학습 | 고정200-step Q 동결 → 동일3parameter Q2/QM/SHAM; missing/zero residual exactQ |
| 분할 | validation/fit 혼합·중복 ID 거부, source block5와 평가 역할 분리, Q-only λ 선택 |
| CUDA | RTX4090의 고정 logits forward/backward가 CPU와 일치. GPU200step 학습·속도 우위 검증은 아님 |
| Native | 새 η0 wrapper와 실제 pinned 저자 toolbox를 같은 인공 bytes로 두 환경에서 직접 비교 |
| 회귀 | 신규180tests PASS7.34초; 전체2057tests PASS239.89초, 기존Torch warning68개 |

원 설계의 `DESIGN_ONLY` JSON은 역사적 과학 선택과 미승인 사람 실행 경계를 그대로 보존한다.
현재 [작업 기록](task_aligned_trca_shape_v1_build.md)이 구현 상태 overlay다.

## 실제 실행한 인공 검증

### 1. 고정 인공 과제와 CUDA

[전체 학습 receipt](reports/task_trca_shape_engineering_v1.json)는 seed20260908,
인공 case1개(k3,12classes,5bands,8channels,N31)에 대한 CPU4head×200steps다.
Q objective는 **0.0192540423 → 0.0191270994**, 4head 학습은2.2634초였다.
이 값은 정규화를 포함한 학습 목적함수이며 정확도 %p나 metadata 추가 이득이 아니다.

CUDA 고정 logits probe의 CPU 대비 최대오차는 R2.22e−16, filter5.20e−16,
score1.85e−15, gradient1.55e−18이고 argmax는 exact였다.
Torch peak allocated18,180,608bytes/reserved23,068,672bytes다. CUDA context·다른 프로세스 메모리를 포함한
GPU 전체 사용량은 아니다. 이 최초-call probe는CPU.01577초/GPU.09477초였으므로 속도 향상을 주장하지 않는다.
CPU peak RSS1,005,756KiB도 import/전체 프로세스를 포함한다.

원본은 `/home/whwovy/task-trca-shape-engineering-v1-vtPDaw/cpu_cuda.json`,
SHA `d17c8828a56a2b61005bdcd97e0ad992b9a2ad04d6d6000d2d0b40affb8327a1`로0400보존했다.
Repo JSON은 같은 파싱 값의 직렬화본이며 SHA `81d9949613a8e673e7d994d3c69960be8266159efbca76a8450ae8ea12b5b5bb`다.

### 2. 실제 중첩 학습 그래프

단위시험의 fit-spy 구조 검사에 더해 [인공6명 중첩 실행](reports/task_trca_shape_nested_engineering_v1.json)을
실제로 완료했다. 같은 고정seed로6명을 생성하고, 각4fit/2validation의3fold에서3λ,
마지막6명 refit까지 **10pipelines/40heads/8000steps**, CPU80.287초다.
Q의 참가자 가중 validation CE만으로 λ=.0001을 선택했다. 선택된 값은 인공 검증 결과이며
미래 사람 연구의 λ를 이 값으로 미리 고정하지 않는다. 원 nested 선택 규칙은 그대로다.

N17/k3/인터페이스1개인 **engineering grid**다. 실제 개발39명/8조건/k3·5나 최종 query를 실행한 것이 아니다.
전체8000step trace·계수·scaler·분할·donor 기록은
`/home/whwovy/task-trca-shape-engineering-v1-vtPDaw/nested.json`,
SHA `e9cf0af288fbe2e54f8f79d6bcce83d320e7a2db84a7fff3df9eca35c3529dbc`에0400보존했다.
Root가 저장기록의 IDs/scalers/donors/steps와 λ 선택을 별도 식으로 검산했다.
이 검산은 원 score를 재구성하는 독립 artifact auditor의 완성을 뜻하지 않는다.

### 3. 실제 저자 코드와 연결

[Native bridge](reports/task_trca_shape_native_bridge_v1.json)는 seed66473의 centered/offset×k3/5,
4가지 인공 입력을 base64 float64 bytes로 부모→Python3.9 native 자식의 stdin에 전달했다.
양쪽 SHA가 정확히 일치하며 실제 ETRCA.fit/predict와 새 η0 wrapper의 correlation 최대오차≤9.992e−16,
argmax exact다. Toolbox revision3344bd199daf78888e364d9db00ae7d8128d2b5f, native NumPy1.23.4,
project NumPy1.26.4다. 같은 check를 통합에서 두 번 실행했고4case 수치는 동일했다.
NumPy monkeypatch/환경 설치/원 toolbox 변경/전처리·dataset 구성은 없었다.

## 발견해 고친 구현 문제

- 공개 predictor가 fit/validation을 섞으면 SHAM donor가 역할 경계를 넘을 수 있었다.
  현재는 혼합 역할·validation/fit ID 중복·불완전 fit donor partition을 먼저 거부한다.
- Builder의 float64 변환이 복소수·문자열·bool 입력을 숨길 수 있어 원 dtype을 먼저 검증한다.
- Native band weights를 암묵적으로 균등값으로 두지 않고 필수 인자로 바꿨다.
- Q 계수는 bytes-backed 불변 배열로 보존한다. Read-only reviewer가 수정과 직접 대응 tests를 재확인했다.
- 초기 operator의 near-tie 시험1개는 부동소수점 경계 fixture를9e−11로 명확화했다.
  명세 top-gap1e−10이나 과학 설정을 완화한 수정이 아니다.

## 남은 작업과 다음 실행 경계

1. 실제 source39 archive의 **역할 제한 reader**: 현 메모리 callback 검사를 실제 decode 범위에 연결한다.
   평가 참가자의 block5, 학습/동결 전 query, query M을 읽지 않아야 한다.
2. 저장 계수·통계에서 score/argmax/집계/판정까지 독립 재구성하는 auditor와
   failure-preserving start→freeze→eval cold CLI를 구현한다.
3. Native preprocessing·실제 weight·입력/코드 SHA, 디스크/RAM/시간 상한을 묶은 별도 실행 manifest를 작성한다.
   현 engineering adapter는 사람 실행 권한이 아니며 final query 기능도 거절한다.

위 연결과 검증이 끝나기 전 실제 EEG 학습/평가를 시작하지 않는다. 새로운 데이터 요청은 이 구현 단계의
필수조건이 아니다. 개발39명 양성이 나오더라도 반복 노출 개발 결과이며 독립 확증은 별도다.

## 재현·통합

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/check_task_trca_shape_engineering.py --cuda
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/check_task_trca_shape_engineering.py --nested
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/check_task_trca_shape_native_bridge.py
```

새 output receipt는 `--output`에 기존에 없는 파일을 명시하면 된다. 기존 결과를 덮어쓰지 않는다.
이것은 인공 engineering CLI이며 사람 실험 runner가 아니다.
Native bridge는 문서화한 별도3.9환경이 있어야 한다.

Base15f853c → root integration88c0109 → operator262f931(원4a79bdf) → features44e1936(원86cef5a)
→ guard/bridge5ff7702 → 추가 인공 nestedCLI3e0ffbd 순서로 통합했다.
전체tests는5ff7702에서 시작했고 그 실행 중 변경은 tests가 import하지 않는 nested engineeringCLI뿐이다.
Core/tests는 불변이며 추가CLI는3e0ffbd에서 위 실제 nested 경로로 별도 검증했다.
Concurrent writing은 분리한2worktree에서만 했고 main은 공통 계약/통합/최종검증을 소유했다.
두 checkout 합계약20.4MiB, 기존32+신규2=34worktrees 보존. 텍스트 충돌0, 위 의미상 경계 문제2건은 수정했다.
환경 변경·push·cleanup·사람 데이터 요청0이다.

`academic-research`에는 구현 근거와 ‘실제 metadata 효과는 미확립’이라는 한계를 분리해 기록한다.
고정 식의 검증이 현재 불확실성이었으므로 이번 새 논문 검색/PDF는0이다.

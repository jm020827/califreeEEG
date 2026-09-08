# Task-aligned TRCA shape v1 — 구현 계약과 검증 순서

2026-09-08 · **구현 전 명세. 아래 파일/API는 별도 표시가 없으면 아직 존재하지 않는 제안 경로다.**
[설계](task_aligned_trca_shape_v1_design.md)와 [JSON](../configs/analysis/task_aligned_trca_shape_v1_design.json)이
과학 선택의 권위다. 이번 변경에 learner/reader/runner 구현이나 새 EEG 실행은 없다.

## 1. 재사용 경계와 모듈 계약

| 제안 모듈 | 입력 → 출력 | 허용 재사용 / 금지 |
|---|---|---|
| `task_trca_shape_features.py` | support[k,12,5,8,N], mask[k,8], order/interface/frequencies → Q[5,8,15]; 별도 M[k,8]→[8,2],available[8] | `m_features`, feature 수식만 재사용. Q에 numeric packet 전달 금지 |
| `task_trca_shape_operator.py` | 고정 S/C[class,band,8,8], native anchor, R[band,8] → C-normalized filters[class,band,8] | 원 `trca_matrices` 유지, 새 bound/projector 구현. 원 operator 수정 금지 |
| `task_trca_shape_learning.py` | 역할 제한 source features/score statistics, fit IDs, λ → scaler/Q 및3residual state | `participant_folds`만 재사용 가능. Proxy MSE `select_q/crossfit_q` 재사용 금지 |
| `task_trca_shape_inputs.py` | allowlisted IDs + 역할 + block set → 허용 prefix/source block5/query projection만 | 옛 `load_native/subject_features`의 full-x/모든 사람 proxy 계산 금지 |
| `run_task_trca_shape.py` | 별도 executable manifest + 검증 receipt → start/freeze/eval/result | 현재 DESIGN_ONLY JSON은 무조건 거절. 옛 runner 변경/재개 금지 |
| 독립 `audit_task_trca_shape.py` | pinned config, fit state, score statistics/predictions → audit receipt | producer의 predictor/aggregation을 import해서 같은 버그를 검산하지 않음 |

구현 시 namespace는 `src/cfeg/analysis/`, runner/auditor는 `scripts/` 아래다.
기존 `native_support_prefix.py`는0–4만 읽는 support helper이므로 block5/query를 읽게 몰래 확장하지 않는다.
새 reader는 role token과 실제 I/O decode 경계를 시험해야 한다. 응용 수준 guard이며 OS 접근 격리가 아니다.
전체 archive를 무결성 hash하는 것과 허용되지 않은 배열을 해독하는 것을 구분해 receipt에 기록한다.

Q 수식의 numerical reference는 기존 `support_q`에 실제M 대신 mask로 만든 인공0/NaN packet을 준 결과다.
새 API가 mask를 넘겨도 숫자 M 값이 달라지면 Q가 바뀌지 않음을 dependency/poison 시험으로 확인한다.
Q2의 standardized2특징은 별도 fit-only scaler를 쓰고 available 행만 fitting한다.
M scaler는 correct/sham의 fit 내 permutation이 관측 rows의 multiset을 보존하므로 공통이다.
Validation/evaluation의 scaler 재적합은 금지한다.

## 2. 미분 가능한 leading projector

C/S/native anchor/Q 특징은 support에서 한 번 계산한 상수다. 학습 gradient는 shape head에서
R→B→필터→원 점수→CE로만 흐른다. C 비양정/비유한·비대칭 오차 초과는 실패다.
비대칭 허용치는 `1e-12*max(1, ||A||F)`로 검사한 뒤 부동소수점 오차만 `(A+Aᵀ)/2`로 맞춘다.
Jitter/pinv/floor로 C를 바꾸지 않는다.

```text
L = cholesky(B)
H = L^-1 S L^-T
H = (H + Hᵀ)/2              # 위 대칭 오차 검사를 통과한 뒤
lambda, V = eigh(H)           # custom autograd forward에서만 호출
v = V[:, -1]
E = v vᵀ                     # sign-independent top projector
```

Top gap은 `lambda[-1]-lambda[-2] > 1e-10*max(1,max(abs(lambda)))`를 요구한다.
하위 고유값끼리의 중복은 허용한다. Upstream `G=dLoss/dE`에 대해

```text
h = sum_{j<top} v_j * (v_jᵀ (G+Gᵀ) v)/(lambda_top-lambda_j)
dLoss/dH = (h vᵀ + v hᵀ)/2
```

인 leading-projector custom backward를 구현한다. 분모는 top-versus-rest gap만 사용한다.
Cholesky/triangular solve의 backward는 float64 기본 연산을 쓴다.
이 식은 **검증해야 할 구현 명세**이지 현재 저장소의 gradcheck 통과 코드가 아니다.

원 native FULL support anchor `w0`를 C norm1로 만들고 stop-gradient 상수로 보관한다.

```text
anchor_h = L^-1 C w0
w_raw = L^-T E anchor_h
w = w_raw / sqrt(w_rawᵀ C w_raw)
```

이렇게 하면 임의의 eigh sign 대신 native anchor와 양의 C-inner-product인 방향을 얻는다.
계산 전에 candidate generalized direction과 w0의 abs C-cosine>1e−6를 검사한다.
거의 직교인 anchor/top tie/nonfinite gradient는 전체 attempt의 `VALIDITY_FAILURE`다.
무작위 sign, 특정 channel 양수 규칙, per-row centering으로 회피하지 않는다.
η=0은 이 과정을 거치지 않고 프로젝트 `metadata_trca_prior.fit_trca(gamma=0)`와
`score_trca`를 그대로 호출해 그 reference와 exact 일치를 보존한다.
원 저자 toolbox와 프로젝트 경로는 별도로 correlation 최대오차1e−9, argmax exact를 검사하며
서로 다른 행렬곱 순서의 byte-exact 일치를 주장하지 않는다.

### 구현 근거의 출처와 한계

[PyTorch v2.2.2 공식 소스](https://github.com/pytorch/pytorch/blob/v2.2.2/torch/linalg/__init__.py)의
`linalg.eigh` warning은 임의의 eigenvector sign과 근접/중복 고유값의 gradient 불안정을 명시한다.
설치된 Torch2.2.2+cu121의 `torch/linalg/__init__.py` lines620–650을 확인했다.
Local SHA256=`d0a40f59375463726f9dbe5b758505966b3a6c0fe71eda600bfded3ecbd3fc7a`, 확인일2026-09-08,
원격 revision=v2.2.2. 현재 stable docs URL은 본문 없는 redirect여서 근거로 쓰지 않았다.
공식 소스는 일반 API 경고의 근거다. **위 custom projector 식이나 우리 EEG 성능을 검증한 출처가 아니다.**
이번에는 새 PDF 정독/광범위 논문 검색을 수행하지 않았다.

## 3. Native 점수와 효율화 경계

원 preprocessing은 각 N에 대해 독립 적용한다. 원래 crop/filterbank/detrend/z-score/35sample discard,
signed linear band sum을 유지한다. 새 projected row centering이나 squared correlation을 넣지 않는다.

한 band에서 W=[w1,…,w12], query X[8,N], candidate template T[8,N]라 하자.
Native ensemble은 WᵀX와 WᵀT를 각각12N개로 펼친 다음 **전체 평균 하나**를 빼 Pearson을 구한다.
따라서 구현 시 Gram 통계로 줄이더라도 다음 보정이 남아야 한다.

```text
J = W Wᵀ; u = W ones(12); m = 12 N
sx = uᵀ X ones(N); st = uᵀ T ones(N)
dot = trace(J X Tᵀ) - sx*st/m
xx  = trace(J X Xᵀ) - sx*sx/m
tt  = trace(J T Tᵀ) - st*st/m
rho = dot / sqrt(xx*tt)
```

Trace의 cross-term orientation은 J가 대칭이므로 동등하지만 구현/gradient 검산으로 확인한다.
안정형 계산 뒤 xx≤0/tt≤0 또는 비유한이면 실패한다. Tiny negative를0/eps로 치환하지 않는다.
유한·양의 분산을 확인한 뒤 최종 correlation만 원 reference처럼 [−1,1]로 clip한다.
독립 class-filter sign 변경은 u를 바꾸므로 시간 평균이0이 아닐 때 이 점수를 바꿀 수 있다.
Template와 query를 같이 투영했다는 이유만으로 sign-invariant라고 단정하지 않는다.

큰 수끼리의 뺄셈을 줄이는 동치형도 사용할 수 있다. `muX=X ones(N)/N`,
`muT=T ones(N)/N`, `Kcenter=(X-muX)(T-muT)ᵀ`,
`Hcenter=(W-u ones(12)ᵀ/12)(W-u ones(12)ᵀ/12)ᵀ`이면
`dot=trace(J Kcenter)+N muXᵀ Hcenter muT`다. xx/tt에도 같은 식을 적용한다.
평균 항을 다시 더하므로 per-row-centered Pearson으로 바뀌지 않는다.
안정형과 원식/flattened 구현의 값·gradient를 모두 검사한다.

Gram cache는 원 score와 값/gradient가 일치할 때만 허용하는 **계산 최적화**다.
CPU float64 기준을 먼저 통과한 뒤 batch8×8 eigensolve와 Gram contraction을 CUDA로 옮길 수 있다.
고정 순서 full-gradient accumulation은 microbatch마다 optimizer step을 하면 안 된다.
GPU가 빠르거나 메모리에 들어간다는 측정은 아직 없다. 합성 benchmark로 peak RAM/VRAM/time을 측정하고
합의된 실행 예산에 못 맞추면 engineering 단계에서 멈춘다. 환경 설치/교체로 자동 확장하지 않는다.

## 4. 필수 인공 검증 — 사람 자료 이전

| 검사군 | 최소 반증 시험과 통과 조건 |
|---|---|
| Bound | 극단 logit/불균형 C/여러 SPD에서 R>0, trace8, ratio≤2, 모든 arm τ 동일; `C^-1/2 P C^-1/2` 최대고유값≤.1+1e−10 |
| Native/normalization | η0 프로젝트 reference score/argmax exact, 원 저자 toolbox correlation≤1e−9/argmax exact; 양의 arm C norm오차≤1e−10; 원 S/C 입력 불변 |
| Projector | unique top + repeated lower eigenspectrum 포함 `torch.autograd.gradcheck(eps=1e−6, atol=1e−5, rtol=1e−3)`; step-halved finite difference도 기록 |
| Gauge/score | nonzero temporal means, eigenvector sign 반전, 고정 native anchor; 입력/anchor/채널정체성의 공동순열; Gram/flattened score max abs≤1e−10와 gradient 일치. 순열 뒤 별도 eigensolver가 새로 고른 anchor sign의 불변성을 가정하지 않음 |
| Failure | top tie/near tie, anchor 거의 직교, singular/indefinite C, NaN/Inf, loss/gradient 비유한을 명시 실패로 보존; sample drop 없음 |
| Learner | 사전에 정한 한 deterministic artificial task에서 유한 loss/gradient와 loss 감소를 확인. Seed/난이도를 M효능이 나올 때까지 선택하지 않음 |
| Null/controls | residual 계수0→exact Q; all numeric M denied→exact Q; Q2/QM/SHAM3params와 동일200steps; no-M행 residual0 |
| Leak | eval block5/query poison, fit 밖 scaler, donor, 한 사람의 window 분리, query M 전달을 거부; λ 선택은 Q validation CE만 |
| Runtime | CPU reference와 CUDA의 R/filter/score/gradient tolerances 및 argmax 일치(명시 tie fixture 별도), 시간·peak memory 기록 |
| Cold CLI | DESIGN_ONLY config 거절, 중복 attempt/변경 hash 거절, start 먼저 기록, 실패 원인 보존; 재학습 없이 auditor 점검 |

CPU/GPU 수치 tolerance는 operator/score1e−9, gradient absolute1e−7+relative1e−5를 초기 engineering
계약으로 사용한다. 사람 데이터를 보고 완화하지 않는다. 다른 eigenbasis sign을 score 실패의 면제 사유로 쓰지 않는다.
Artificial optimization 실패는 학습 구현/고정 최적화 미검증이지 ‘M이 무효’인 결과가 아니다.

## 5. 산출물과 독립 감사

별도 실행 계약은 아래 항목을 채우기 전 `READY`가 될 수 없다. 현재 경로/hash/resource는 미정이며
이를 임의의 placeholder로 실행 가능하게 만들지 않는다.

- **start**: 사용자 실행 범위, clean code/config/environment/input hash, IDs/folds, device/threads,
  역할별 허용 blocks, 시간/디스크 상한, 실패·복구 경계. 실패도 삭제하지 않는다.
- **fit**: partition IDs, scaler statistics, λ별 Q validation CE, donor maps/coverage,
  모든 head의 초기/최종 계수·200step loss/gradient norm·finite flags. Q frozen hash를 세 잔차에 연결한다.
- **freeze**: 모든 outer 모델·scaler·native anchor/feature 계약 hash와 query 미개봉 증빙.
- **evaluation**: all312conditions×2budgets×48query의 arm별 raw scores/argmax/정수 correct;
  R/필터/score 변화량 및 정답이 바뀐 query, margin, NLL. 둘 이상의 budget을 독립 표본으로 세지 않는다.
- **result/audit**: 참가자 단위 paired endpoint와 모든 cell, CI/decision 조건별 boolean,
  관측0/3/5 도달 전이, 도움/동률/손해, 무효·미도달 null, 실제 runtime/RAM/VRAM/label costs.
  Auditor가 저장 계수와 허용 통계에서 score/정수 집계/통계/판정 및 fold/donor를 독립 재계산한다.

SHAM은 exact missingness stratum 안이므로 donor row multiset/availability 보존을 검증한다.
STALE은 첫 packet에 없던 채널을 실제값0으로 꾸미지 않고 unavailable residual0으로 처리한다.
구조적 무작동과 작은 score 변화/불변 argmax를 별도 기록한다.

## 6. 실행 순서와 작성 소유권

현재 base bfe5703/main에서 root만 설계/config/docs/연구공간을 작성한다.
수학, 누수, 회의적 검토3명은 shared repo read-only다. 문서·계약이 서로 의존하므로 concurrent writing은 없다.
기존32worktrees·환경·원 결과를 보존한다. 이후 구현을 병렬화할 때는 다시 worktree/disk gate를 수행한다.

제안 통합 순서는 **계약 → features/operator+인공시험 → nested learner → role reader/cold CLI →
독립 auditor → 전체 회귀·고정 합성 runtime → 별도 사람 실행 계약**이다.
원 `metadata_trca_prior.py` SHA256
`71f7f54e91b9afaf0f571b75da64d4adb7d36d72dc1fe6c2e0d8afab2425baed`와 기존 negative result/geometry는 유지한다.
새 core가 없으므로 과거1877tests PASS를 이번 새 학습기 검증으로 표시하지 않는다.

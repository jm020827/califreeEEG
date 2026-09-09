# N1 metadata generated efficacy v1 — prospective design

2026-09-10 사용자 “응 시작하자”는 직전 결과보고의 인공 양성·음성 대조 검증 제안 승인으로
적용한다. 사람자료 재실행·held60·외부 요청·유료·GPU·새 패키지 설치는 제외한다.
원36 science/runtime 파일과 이전 결과는 보존한다. 등록 생성·학습·oracle 평가 전에 아래
기작·기준·자원과 실행 계약을 완성해 별도 config/commit으로 고정한다. 현재 등록 실행0이다.

## Problem signature

- Representation: 생성된 SSVEP-like support `[k,12,5,8,N]`, 별도 source supervision 및 query
  파형, pre-query acquisition 상태를 나타내는 channel-wise 숫자 packet. Q15는 원 생성자로
  실제 support에서 계산하며 Q/S/C/Gram을 임의로 대체하지 않는다.
- Bottleneck: 지난 N1-R은 M을 학습하고 R/F/J/점수를 바꿨지만 k3 예측을 바꾸지 못했다.
  제한된 M 경로의 능력과 실제 M의 정보 부족은 아직 분리되지 않았다.
- Allowed operation: 원 N1 학습·평가 함수를 사용하는 새 generated-only orchestration,
  동일 EEG·공통 mask/order를 공유하는 연결된 M과 연결이 끊긴 M 대조, 같은 제한 범위의
  사전 고정 oracle. 실제 사람 배열·옛 모델·query 점수 재접근은 하지 않는다.
- Objective: 원 제한된 Q→동결→M 잔차 학습이 사전 명시한 인공 기작에서 실제 예측 선택을
  개선할 수 있는지 검사한다. 사람 metadata 효능·보정량 절감의 증거가 아니다.
- Resource constraint: 인공 기작1개, 연결/비연결2regimes, 각regime 학습1회, 추가 기작/seed
  탐색0. 정확한 사례수·optimizer budget·wall time·출력 한도는 실행 전 고정할 질문이다.
- Feedback: 생성 유효성 → 동일 제한의 oracle headroom → 학습 QM−Q/Q2/SHAM와 음성대조
  → 독립 저장 산출물 검산. 미적격 양성대조는 learner 실패나 M 무용성으로 판정하지 않는다.
- Failure modes: Q에 M/정답 누수, 임의 Q/score 구성, unreachable oracle, 수치 실패,
  분류 ceiling/미미한 margin, 정보 없는 대조의 우연 이득, 사후 생성기 조절, 실제 EEG로의 과잉 일반화.

## 기존 결과에서 넘어오는 근거

[source39 recovery 결과](task_trca_n1_transport_recovery_r1_results.md)의 학습 경로는 작동했으나
고정 정확도·비용 이득이 없었다. 기존 generated24k 시험은 실행 가능성/검산 coverage이며,
metadata가 유용하도록 명시한 효능 양성대조가 아니었다. 본 단계는 이 증거 공백만 다룬다.

현재는 root가 설계·공통 API·근거 DB를 단독 작성하고 두 agent가 읽기 전용으로 기작/API를
검토한다. 계약이 명확해지기 전에는 병렬 writer를 시작하지 않는다. 기존40worktrees와
4개 main untracked 디렉터리 보존. 가용164,316,156KiB에서 기존 runtime/cold 두 tree를
재사용할 수 있으나, 최종 소유권/증분 예산을 고정한 뒤에만 전환한다.

## Frozen question and scope

이하와 `configs/n1_metadata_generated_efficacy_v1.json`을 등록 생성 전에 commit한다.
구현 오류는 등록 실행 전에 unit/contract test로 수정할 수 있지만, 등록 seed의 효능 결과를
본 뒤 생성기·강도·seed·기준·학습 절차를 바꾸지 않는다. 등록 실행은 한 번이다.

원 저보정 SSVEP 연구목표는 유지한다. 본 단계는 사람자료에서 나타난 부정 결과를 설명하기
위한 좁은 **constructive capacity control**이다. k3만 사용하며 k5, calibration-cost,
A0, 사람 metadata 효과, 자연적인 acquisition-noise 모형의 타당성을 검증하지 않는다.
Q15와 native S/C는 실제 생성 파형에서 원 함수로 계산한다. 원 N1 학습·scorer는 수정하지 않는다.

## Data-generating mechanism (DGP)

N=256, Fs=250Hz, k=3, 12classes, 5bands, 8channels, interface=order=0,
weights=[1,1,1,1,1]. class y의 Fourier bin은 m=y+6이고 f=250m/256Hz이다.
각 group/band/class의 위상 φ를 독립 uniform[0,2π)로 정하며
`s[b,y,t]=sqrt(2)*sin(2π*(y+6)*t/256+φ[b,y])`; 각 행은 평균0, 제곱노름256이다.

각 group/band에서 constant와 `{h*m: h=1..5,m=6..17}`의 sin/cos를 모두 포함하는
orthonormal Fourier basis U를 만든다. G~Normal(0,1)[256,24]를 U 밖으로 두 번 투영한
뒤 reduced QR, R 대각 부호 정규화, sqrt(256) 배율을 적용한다. 24개 열을 3block×8channel
noise E로 재배열한다. `support[r,y,b,c,t]=s[b,y,t]+0.2*E[b,r,c,t]`.
따라서 분석적으로 S=6N11ᵀ, C=3N(11ᵀ+0.04I), C condition≈201이다.
원 native 계산과 이 식의 절대 오차≤1e-8 및 Fourier/noise 직교성을 검산한다.

한 group은 두 counterfactual members이다. z는 무작위 4-of-8 bad-channel mask이고
member1은 그 complement이다. support는 두 member 및 두 M regime에서 동일하다.
각 split/group별 source δ[12]와 query δ[4,12]를 서로 독립 uniform[-0.01,0.01]로
정하고 두 member와 5bands가 공유한다. source/query class y의 good channel은
`(1+δ)*s_y`, bad channel은 `(1−δ)*s_((y+1)%12)`이다. 추가 query noise는 없다.
metadata는 class와 독립인 channel context이며 query label을 포함하지 않는다.
단, bad channel이 정확한 decoy를 갖는 의도적으로 유리한 낮은-margin 구성이다.
이는 일반적 EEG 손상이나 생리학적 impedance 모형이 아니다.

연결 regime(`coupled`)의 numeric packet은 3개 동일 행 `expm1(1+2*z_member)`이다.
연결이 끊긴 regime(`null`)은 독립 4-of-8 mask v를 같은 공식으로 packet화하고 두
member가 같은 v를 공유한다. 관측 mask는 모두 true, common order/interface/packet
분포는 같다. M1은 channel-centered log-mean, M2=0. STALE=QM도 예상된다.
null은 pair 대칭으로 gradient가 상쇄될 수 있는 강한 구조적 음성 대조이다. null CI=[0,0]
또는 Q/Q2/null-QM zero gradient도 유효하며 일반적인 false-positive 보정력으로 해석하지 않는다.

RNG: NumPy PCG64(SeedSequence([20260916,split,group,purpose,(optional band)]));
split0=fit,1=evaluation; purpose1=z,2=v,3=φ,4=G(+band),5=sourceδ,6=queryδ.
z/v는 `choice(8,4,replace=False)`로 생성한다. 모든 RNG stream은 별도이며 score/label
성능에 따른 재추출은 없다. fit8groups=16records, evaluation32groups=64records.
ID는 fit base11000/eval base21000에 `member*G+group+1`을 더한다. 즉 member-major다.
기존 sorted-cyclic donor는 반드시 다른 group으로 간다. group을 독립 사람으로 부르지 않는다.

## Methods and freeze barrier

각 regime에서 원 `make_task_case`와 `fit_pipeline(...,0.001,backend="batch")`를 사용한다.
고정 λ=.001, head당200updates, Q→동결→Q2/QM/SHAM_REFIT, regime당800updates,
총1600updates. nested model selection이 아닌 사전 고정 단일 λ capability test이다.
두 모델을 모두 저장·hash 동결한 다음 evaluation query archive를 최초 decode한다.
생성기는 query를 앞서 만들지만 learner는 그것을 받지 않는다. 이것은 프로그램 역할 분리이지
OS sandbox/암호학적 non-access 증명이 아니다. evaluation scorer에는 label을 전달하지 않는다.

원 10arms FULL_NATIVE/FULL_CENTERED/ISO/Q/Q2/QM/SHAM_REFIT/PERMUTED/STALE/MISSING
및 reachable oracle을 모두 저장한다. oracle은 학습된 Q와 M scaler를 그대로 쓰고 residual
coefficients=[2,0,0]을 사전 고정한다. `_head`의 .2*A*tanh, 동일 shape_prior/N1/temporal
scorer를 쓰며 oracle을 학습한 HeadFit처럼 기록하지 않는다. oracle 계수 탐색0이다.
이상적인 균일 Q와 S/C에서 oracle의 경계는 δ≈−.00355, 기대 이득≈17.8pp라는 수식상
예상은 가능하지만, 이것은 관측 결과가 아니다. 실제 oracle 적격 여부는 아래 기준으로 판정한다.

## Estimands, criteria, stop

정확도는 각 group의 두 member×48query를 합한 비율이다. 32개 group delta의 평균(pp),
표본SD/sqrt(32), t31의 양측95% interval을 저장한다. 이는 **고정 latent/지원파형/donor/모델에
조건부인 δ 변동의 descriptive approximate interval**이다. donor가 다른 group의 M을 소비하므로
전체 생성법 아래 outputs를 무조건 iid라고 주장하지 않는다. formal unconditional coverage,
동시95%CI, 다중검정 보정, training-seed variance, 일반적 type-I calibration을 주장하지 않는다.
donor edge를 끊는 bootstrap은 하지 않는다. thresholds는 engineering capability gate이다.

1. 유효성/자원/독립검산 실패: EFFICACY_NOT_EVALUATED. 수치 비교 abs tolerance1e-8,
   score tolerance1e-8, model Q hash 두 regime 동일, 전체 finite, 전 raw/Q/S/C bridge 검증.
2. Q 평균 정확도가 [10,90]% 밖이거나 coupled oracle−Q mean<5pp 또는 interval low≤0:
   POSITIVE_CONTROL_NOT_QUALIFIED. learner 보편적 무능력으로 판정하지 않는다.
3. null QM−Q interval이 [-1,+1]pp 내부가 아님: NEGATIVE_CONTROL_NOT_QUALIFIED.
4. coupled QM−Q, QM−Q2, QM−SHAM_REFIT 각각 mean≥1pp 및 low>0,
   paired interaction `(QM−Q)_coupled−(QM−Q)_null`도 mean≥1pp/low>0,
   coupled QM nonzero coefficient/gradient와 R/score/argmax actuation 모두 필요하다.
   만족하면 GENERATED_M_CAPACITY_PASS, 아니면 LEARNER_CAPACITY_NOT_ESTABLISHED.
5. 모든 결과와 control failures를 보존한다. 재생성/다른 seed/δ폭/β/λ/예산 확대0.
   어떤 결과도 사람자료·held60 재실행을 자동 승인하지 않는다.

## Resource and coordination contract

등록 primary1회(생성+2fits+평가), wall≤900s; 독립 audit1회 wall≤600s.
CPU1/float64, RSS≤8GiB, process address-space≤16GiB; output≤4GiB.
전체 새 test wall≤1800s(각 lane≤300s, full suite≤900s), test artifacts≤2GiB.
새 code/checkouts 포함 증분≤8GiB. timeout은 SIGINT→10s→SIGKILL, 재시도0.
runtime 중 오류면 소비된 등록 attempt를 보존하고 종료한다. 실패 시 updates의 exact partial
count가 없는 경우 completed lower bound와 charged budget을 구분한다.

| Lane | Owned paths | Runtime | Integration |
|---|---|---|---|
| root | design/config, new producer/launcher/tests, final reports/state/research DB | main, primary/audit orchestration | contracts→fixture→producer→auditor→E2E |
| task_shape_reader | `src/cfeg/analysis/n1_metadata_generated_efficacy_fixture.py`, corresponding test | existing temporal-runtime tree, new branch; toy component tests≤300s | first leaf |
| goal_scope_review | `src/cfeg/analysis/n1_metadata_generated_efficacy_audit.py`, corresponding test | existing temporal-cold tree, new branch; toy component tests≤300s | second leaf |

원36 runtime/science 파일 수정0, dependency/lockfile 변경0, 신규 worktree0.
기존 branch와 untracked test folders는 보존하며 명시적 새 branch로 기존 두 slot을 전환한다.
shared `.venv/bin/python`은 읽기/실행만 하고 설치하지 않는다. agent별 PYTHONPATH는 자기 tree의
src, test temp는 자기 tree `.pytest-generated-efficacy-{fixture,audit}`다. root만 DB writer.
root가 diff/path/semantic 계약 검토 및 최종 실제 연결 검증을 소유한다. cleanup하지 않는다.

## Frozen new-module API and artifact contract

Fixture: `generate_group(config, split:int, group:int)->dict` 는 support[3,12,5,8,N],
source[2,12,5,8,N], query[2,48,5,8,N], packet_coupled/null[2,3,8],
bad[2,8], null_mask[8], phases[5,12], source_delta[12], query_delta[4,12]를 반환한다.
`write_fixture(folder:Path,config:dict)->dict` 는 아래 artifacts를 exclusive-create하며 manifest를
반환한다. overwrite/symlink/pickle 금지. registered group generation은 primary만 호출한다.

- fit.npz: ids[16], groups[16](0-based), members[16], support[16,3,12,5,8,256],
  source[16,12,5,8,256], packet_coupled/null[16,3,8], bad[16,8], null_mask[8,8],
  phases[8,5,12], source_delta[8,12].
- support.npz: 동일 eval64 IDs/groups/members/support/packets/bad,
  null_mask[32,8], phases[32,5,12]; source/source_delta는 없음.
- query.npz: ids[64], groups[64], members[64], query[64,48,5,8,256], query_delta[32,4,12].
- fixture.json: schema=`n1-metadata-generated-efficacy-fixture-v1`, config, artifacts
  `{basename:{sha256,bytes}}` (three npz only), counts `{fit_groups:8,evaluation_groups:32}`.
- models.json: `{schema:"n1-metadata-generated-efficacy-models-v1",pipelines:{coupled:original_record,null:original_record},oracle:{coefficients:[2,0,0],trained:false}}`.
- freeze.json: schema, models_sha256, query_sha256, completed_updates=1600,
  event=`both_models_frozen_before_query_decode`, UTC. events.jsonl records generation,
  each fit start/end, model freeze, query decode, evaluation end in strict order.
- fit_features.npz: q[2,16,5,8,15], m[2,16,8,2], s/c[2,16,5,12,8,8].
- evaluation.npz: scores[2,64,10,48,12], r[2,64,8,5,8],
  projectors[2,64,8,5,12,8,8], oracle_scores[2,64,48,12],
  oracle_r[2,64,5,8], oracle_projectors[2,64,5,12,8,8].
  Regime order coupled,null; arm order existing evaluation.ARMS.
- result.json: schema, terminal, config, arms, group_accuracy_percent[2,32,11]
  (oracle last), contrasts `{coupled_QM_minus_Q, coupled_QM_minus_Q2,
  coupled_QM_minus_SHAM_REFIT, coupled_ORACLE_minus_Q, null_QM_minus_Q,
  interaction}` each `{mean_pp,mcse_pp,low_pp,high_pp,n_groups:32}`;
  checks, actuation, pipeline_q_hashes, updates_completed=1600, artifacts hash inventory.
  Per-regime actuation keys max_abs_R/F/J/score, argmax_changed and gradient/coeff maxima.

Auditor: `audit(folder:Path,config:dict)->dict` reads only regular local artifacts, checks all hashes,
RNG z/v/phase/δ and raw formula/orthogonality/geometry, raw→Q/M/scaler/S/C bridge, donor cross-group
assignment, two-Q identity, traces/budget/freeze order. Recomputes all ten arms and oracle using independent
NumPy/SciPy eigensolve/temporal math (can use unchanged feature constructors, disclose shared component),
matches saved scores/R/F/argmax and independently recomputes all reported deltas/terminal. No Adam replay
or independent sourcecode-provenance security claim. Returns JSON schema, status PASS/FAIL, errors,
checks, max differences and recomputed summary; does not write results itself. Root invokes once and writes
audit.json. A score/count/model/raw mutation must be detected by tests, not registered data perturbation.

## Literature use and coverage

Morris, White & Crowther (2019), *Using simulation studies to evaluate statistical methods*,
[official Wiley article](https://onlinelibrary.wiley.com/doi/full/10.1002/sim.8086), DOI10.1002/sim.8086,
informs the separation of aims, DGP, estimands, methods and performance (ADEMP), and the need to report
simulation uncertainty. Official abstract and selected HTML planning/seed/Monte Carlo sections were
inspected; no PDF or complete-paper deep read is claimed. Our conditional interval limitations above are
our design qualifications, not a claim that this paper certifies them.

Structured search `search:1af09279c601ae7d` (2026-09-09UTC) queried the exact methods-paper title;
5 records ingested, only the matching DOI is used here. Semantic Scholar exhausted429 retries and
OpenAlex429 appeared in console: this limited foundation lookup is not an exhaustive latest-literature
review. Broader research cutoff2026-09-04 is unchanged. No new physiology/real-data efficacy claim rests
on this lookup.

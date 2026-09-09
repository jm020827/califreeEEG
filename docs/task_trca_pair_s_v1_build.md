# C2 engineering eligibility and leaf contract

2026-09-09; root-owned prospective implementation contract. Scientific design SHA
`1561541f541d3cbf9a3b2b5a35e78a409edc4099cff3bddf6b343c0d37109ee3` and program SHA
`78ca14d5c155841b727ef8b35b958188639dd7677d1e41ca9e8ffa9bc3bc3a10` remain unchanged.

## Eligibility decision, before C2 implementation

C1 is CLOSED / VALIDITY_FAILURE / efficacy NOT_EVALUATED. Its frozen PERMUTED R path
reproduces the symmetry rejection at one saved prefix (ratio 1.05653); ISO and C2
UNIFORM at B=C pass that prefix (ratios .762435/.532553). Two independent code-only
reviewers found no frozen-prior formula discrepancy. This narrows a C1 R-dependent
failure; it neither demonstrates a shared C2 defect nor rules numerical instability out.
Authorize **generated-only C2 engineering** under its previously frozen science and
unchanged numerical guards. Human C2 remains gated by its own completed preflight,
recorded eligibility and a distinct manifest. No C1 repair/restart, omitted control,
pre-validator H symmetrization, threshold relaxation, backend escape or seed search.
If the required C2 checks expose a defect requiring numerical-policy change, close C2
NOT_EVALUATED; do not spend its human attempt. The original research question remains open.

## Ownership and resource gate

Main base at inspection: ac0c6f27a3a8c2d5bdae300a8506b195b881f932, main. Main owns existing
uncommitted terminal reports and all shared contracts, integration, case/evaluation schema,
GPU and research SQLite. Available disk 233,917,264 KiB; no dependency installation.
Reuse the two already budgeted trees, preserving their old branch heads and untracked
test outputs. Each checkout <=64 MiB plus <=512 MiB test output; root venv read-only,
own src via PYTHONPATH, CPU1, no human data, no GPU for leaf lanes. No new tree or cleanup.

| Lane | Owned new paths only | Dependency | Validation | Integration |
|---|---|---|---|---|
| Features, temporal-runtime tree | src/cfeg/analysis/task_trca_pair_s_features.py; tests/test_task_trca_pair_s_features.py | Frozen JSON, immutable FeatureScaler/donor_map | Literal Q/M/scaler formulas, masks, k3/k5, channel equivariance | first |
| Operator, temporal-cold tree | src/cfeg/analysis/task_trca_pair_s_operator.py; tests/test_task_trca_pair_s_operator.py | Unchanged leading_projector and temporal scorer | Pair bounds, S gradient, SciPy/literal score, failures | second |
| Main | This contract; terminal records; subsequent generated integration | Both leaves after review | Cross-lane and runtime verification | final |
| Shared read-only reviewer | No writes | Designs and code only | Independent semantic/eligibility review | advisory |

Each writer may commit only its two owned files. Return commit/diff/tests/assumptions and
remaining gaps. No edits to any C1/old pinned code/config. Dependent learner, source archive
and cold implementation remain sequential until leaf and case contracts are integrated.

## Features public API (NumPy, no I/O)

- `pair_coordinates(k) -> int64[n,2]`, only k3/5, lexicographic, read-only copy.
- `support_q_mask(support, mask, interface, order, frequencies) -> float64[5,n,15]`.
  Exact frozen Q formulas; numeric M is not an argument. Native support [k,12,5,8,N].
- `pair_cross_products(support) -> float64[5,n,12,8,8]`. No individual trial centering.
- `metadata_features(packet) -> (float64[n,2], bool[n])`, packet [k,8], exact M2.
- `stale_packet(packet) -> float64[k,8]`: block0 values with original prefix mask retained.
- `fit_weighted_scaler(values, participant_ids, fit_ids, masses, availability=None)`
  -> immutable old `FeatureScaler`. Values [rows,d], IDs/masses [rows], availability [rows].
  Masses finite strictly positive (all flattened rows are real pairs, no padding); available
  fitting mass globally normalized, weighted population
  mean/SD; SD<1e-12 ->1; empty available mean0/scale1. Excluded participant/unavailable
  numeric rows may contain NaN and cannot affect fit. Caller assigns each Q/Q2 pair 1/n,
  and deduplicates M by pid/interface/k before flattening; verify duplicate packet equality.
- Export immutable `FeatureScaler`, `donor_map` from old feature module, Q_DIMENSIONS=15,
  M_DIMENSIONS=2, Q2_INDICES=(2,4). Never reuse old unweighted fit_scaler or old M formulas.

## Operator public API (torch float64, no I/O)

- `pair_distribution(logits) -> (p,d)` with final axis n=3/10; u=exp(logits-max),
  p=u/sum(u), d=(u-mean(u))/mean(u). Validate finite float64, same fixed logit envelope
  abs(logit)<=log(2)/2 (roundoff tolerance only), positive normalized bounds. No padding.
- `weighted_numerator(s0, pair_cross, logits) -> S`, s0 [...,b,12,8,8], pair_cross
  [...,b,n,12,8,8], logits [...,b,n]. Constants s0/A, differentiable logits and S.
  S0 + sum(d*A), exact S0 at uniform logits with gradient intact. S trace not normalized.
- `c_projectors(s,c) -> F`, matrices matching [...,8,8], differentiable symmetric S,
  constant SPD C. Exactly B=C; original two triangular solves then unchanged
  `leading_projector(H)`; no pre-check H symmetrization. Original first-order/root-gap/
  symmetry guards. Backtransform K, symmetrize K as original, F=K/tr(C@K).
  Normalization must be finite and strictly positive. C trace near1 is recorded as a
  diagnostic/test invariant, not a new runtime gate or C1 [1/1.1,1] ridge acceptance band.
- `pair_projectors(s0,c,pair_cross,logits) -> F` composition only.
- Temporal Gram constructor/scorer reused unchanged; no native-anchor or old R wrapper.

Native pair widths remain separate for computation. k3 pairs are not first3 of k5.
Later grouped full-batch loss weights groups by number of cases, one regularizer/update.
No learned C1 checkpoint or C1 selected lambda can initialize C2.

## Generated gate recipe fixed before execution

Leaf unit fixtures are algebraic checks, not seed/parameter tuning. Root integration uses
one deterministic seed 20260909 and records the first outcome. Required original guard
stress test uses generated native support k3/5, 12 classes,5 bands,8 channels,N125;
test target C condition levels [1,1000,100000,120000]. These are a declared engineering
ladder spanning the approximately117362 maximum already diagnosed, not an efficacy sweep.
Use NumPy default_rng(20260909), draw shared[12,5,8,125] standard normal first, then
noise[5,12,5,8,125] standard normal; support=shared+noise (both amplitude1), k3 is its first3
blocks, k5 all5. For each k/class/band independently, diagonalize native C=U diag(lambda) U^T
in ascending eigenvalue order. For target condition t use all8 geometric eigenvalues
v_j=t^(j/7), j=0..7, mu=v*trace(C)/sum(v). Apply the symmetric orientation-preserving
congruence T=U diag(sqrt(mu/lambda)) U^T to every block of that class/band. Do not rotate
into a diagonal C basis. Rebuild
native S0/C/A from transformed support. Keep templates uniform. Uniform and fixed linearly
spaced bounded logits (−log2/2 to +log2/2) are evaluated with the original scalar and batch
operator, no learned head or query/outcome needed for this stress gate. Nonuniform logits
increase with lexicographic pair index, shared across bands/classes; no reverse-direction
search. Condition1 uses the same construction, not an alternate easier pipeline.
Record every level,
first failing matrix and the actual condition/symmetry/root-gap diagnostics. This gate has
CPU1/60seconds/64MiB fresh output budget and zero optimizer updates; run every predeclared
row even after a rejection solely to report this fixed coverage, never select a passing level.
Do not change
seed, spectrum, guard or math in response to failure. This is a numerical coverage check,
not proof of human stability or biological realism; generated matrices never count as people.

If this gate passes, additional required work is actual Q/Q2/QM/SHAM fixed200step CPU/CUDA
parity, nested selection, new-schema role reader and independent persisted-artifact audit,
followed by full regression and a separate human manifest. Passing leaf tests alone cannot
activate human C2 or complete the research goal.

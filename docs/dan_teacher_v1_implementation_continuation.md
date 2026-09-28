# DAN teacher v1 implementation continuation — 2026-09-28

User `ㄱㄱ` authorizes the previously frozen final candidate, not a new candidate,
extra tuning, held60 reveal, external contact or automatic expansion of budgets.
Main is the sole writer. No branch integration, push, worktree or agent is needed
for this sequential implementation. Existing untracked test directories are untouched.

Plan/config hashes remain unchanged. Generated call1 used120 optimizer updates,
3.84 seconds and1/6 calls. This continuation reserves call2 for365 updates and
180 seconds: five arms x three bands x11 updates=165; one full-shape CUDA
two-stage profile at50 pretrain epochs/10 fine epochs=200. Those shortened
epochs belong only to generated fixtures; human epochs remain500/150.

Implementation details fixed before this call:

- Independent SciPy filter-bank/TRCA/CCA equations, available535sample prefix only.
- Time-centering each trial for covariance/correlation; trace ridge scaled by
  covariance trace/channel count; TRCA vectors unit norm with deterministic sign.
- Source order5people/6blocks/class-major. Pretrain fitfirst4,validatefifth;
  fine trainfirst4blocks,validate2. Per-fit CPU randperm seed is adaptation seed
  for pretrain and adaptation seed+1000+source_position for fine; paired arms
  receive identical schedules. Band seeds differ byband index in the fixture.
- New Adam optimizer per fit; every fine stage restarts from the same selected
  pretrain state. Select minimum source-validation MSE, earlier on exact ties.
- Persistable source-validation traces and six selected states per band/cell.
  Model checkpoint selection never receives target-query arrays/labels.
- Generator checks include direct pairwise covariance, independent scalar
  decoder correlations, independent NumPy saved-model forward, matched starts/
  schedules, source-only selection, and complete transformed-source augmentation.
- Partial resource extrapolation includes checkpoint cloning and in-memory save,
  but excludes disk, full traces, filter bank/eTRCA, cohort scheduling and audit.
  It is not whole-run human-entry qualification.

Repair round1 covers initial style correction and any documented numerical test
repairs in this integration stage. No frozen scientific setting may be changed.
The role-limited reader, immutable all-model freeze, one-shot cohort runner and
independent saved-result auditor still need implementation and qualification.

## After call2, before call3

Call2 passed6checks using365updates/3.93wallseconds. The first full-shape CUDA
alignment took0.61445seconds for200updates, and its conservative extrapolation
was107852seconds, above the86400second training ceiling. That timed path includes
the first CUDA fit in the process; startup is not a per-cell training cost.

Reserve exactly one cost-separation measurement (call3,208updates,120seconds):
one8-update short two-stage warm fit plus one200-update measured fit. The warm
cost is counted once, then use the same predeclared2x/12.5 ratio for recurring
costs, not a fastest-of-many result. No accuracy is measured or selected here.
If this corrected training-path estimate still exceeds86400seconds, do not
launch human fitting or silently reduce epochs/sources/controls. Report the
resource-limited stop. If it fits, remaining nontraining/runtime costs must
still be qualified before human entry. Old cold measurement remains preserved.

## After call3, before call4

The one-time warm cost was0.72772seconds; steady200updates0.23636seconds. The same
conservative extrapolation is41509.53seconds(11.53h), below24h. This clears only
the learning-path projection, not the whole-run gate. Call3 used208updates and
2.65wallseconds. Total use is693/1200updates,10.42/1200seconds,3/6calls.

Call4 reserves zero optimizer updates/120seconds for role and freeze guards:
sorted-ID mod3 folds, firstfive source pool, firstfour-only channel scalers,
within-target-role/acquisition-order SHAM derangement with seedsham_seed+fold,
identical donor acrossk/interfaces, copied paid support packets, and all-model
registration before a single final query capability. Singleton SHAM strata fail
closed instead of falling back to true metadata. The gate checks registration
ordering only; durable artifact hashing, reader and auditor remain runner duties.

## Before persisted qualification call5 — 2026-09-28

User `시작할까? 시작.` continues this programme. No scientific config change.
Remaining budget before call5:2calls/507updates/1188.88seconds. Reserve360updates
and600wallseconds for call5: one generated target/interface/seed, all3k/5arms/
3bands, pretrain1epoch/fine1epoch. Test source pool still contains5people and
6blocks/12classes; source role/scalers remain first4. Human retains39targets/
2interfaces/2seeds/500+150epochs. Band seed is base_seed+band for every arm.

The generated test writes45alignment artifacts/270selected states and18decoders,
freezes all hashes, scores19rows including CCA, and independently audits912trial
decisions. Deliberately altered teacher and cost copies must fail; the original
generated artifacts stay unchanged. Reader test mocks all39raw loads/hashes and
the single4680row projected metadata read. It is not a real extraction attempt.

Raw MAT decoding necessarily includes query samples in extraction memory. Only
first6 source-support blocks are preprocessed; raw query prefix6:10 is cached
in a separate vault. Fitting receives SupportArrays only. Query preprocessing
and labels/scoring occur after the complete global artifact freeze and single
reveal. This clarifies physical container access versus learning access.

All cached arrays use float64; neural source/support/teacher arithmetic uses
float32; independent decoder/covariance/scalers use float64. Saved audit checks
source role/scalers, SHAM donor mapping, teacher weights, selected-state MSE and
lineage, decoder eigensystem optimality, independently expressed scores and
argmax/counts, descriptive intervals/cost/gates. It does not retrain or reproduce
raw preprocessing/every training update. Numerical thresholds are set in code
before this test; they must not be widened after human fitting.

Whole-resource microprofile usesfull8channels/375samples,largest35repeat eTRCA,
48queries,CPU NumPy checkpoint forward and distinct-state disk write/hash/read.
Projection uses2x scaling plus300audit seconds and3600whole-run margin. It must
fit24h training/30h total/1h audit/12GiB output; runtime guards still enforce caps.
Exact integer correct counts drive group means to avoid tiny signed-zero
floating artifacts at strict positive-gain boundaries. Scientific endpoints unchanged.

Implementation round2 includes new integration code and pre-test static cleanup.
There is no human entry until all tests, resource screen and exact artifact hashes
are bound in a PASS qualification. Actual run has one exclusive repo-level start
receipt and automatic saved audit/terminal receipt, with no retry/resume option.

## Before final generated call6

Call5 passed4tests/360updates/6.15wallseconds. Full saved CPU audit45cells/
270selected states/912decisions passed with max score difference3.064e-14 and
zero argmax differences; deliberate teacher/cost corruption was rejected.
Full-shape projected total47165.41sec/audit798.62sec/output6,621,114,304B fit caps.
Actual human IO remains0; reader was mocked. Remaining1call/147updates.

Final call6 reserves72updates/300seconds: CUDA at8channels×375samples, k2/3/5,
3bands, pretrain1/fine1, QM weights,9cells/54selected states. Independently check
saved selected MSE/source transformations/decoder lineage and144query decisions;
also cold-recheck the original persisted45CPUcells without refitting. No method
selection or efficacy threshold is involved. Use identical numerical tolerances.

Final implementation round3 adds pre-start disk/GPU-busy checks and binds this
last test in the qualification artifact set. No further implementation repairs
or generated calls are permitted after call6. If it fails, stop before human IO.

# MAMEM source-learning v2 — source-backed label compatibility, same M candidate

2026-09-13, root4dbe325 after source v1 closeoutc4d6084. Original low-calibration
SSVEP acquisition-M goal unchanged. v1 real fits0/80 and its guardband failure remain
frozen. DIN-only diagnostic1 completed: failures2,15,17,18,19; all23analysiswindows
contained. Only development S001a has been inspected; S002–S011 DIN/EEG/M/label stats0.

## Why this revision, before any source data

The quarter-spacing nominal guardband was OUR choice, not a documented author/MOABB
bound. Source Session.m at revision5a03abe2a6a874e9adaceea29a52c2fce35d8a03 passes
1000/(2meanΔt) to Trial, and the newly retrieved same-revision Trial.m (blobde32cd15f00c8a5b5c515bdd361a966ad38800ff)
stores it unchanged. No hidden nominal-label normalization is implemented there.
The retained MOABB mamem_event (blob eef0203f06c86f18f6db53f7cc67cbd4a6af685b,
decodedSHA0923958d1fc7a5e2e20db7139dce730d2663f7a0fd67e74f006752b63a1a42f4)
L76/87/89–96 defines an explicit double-integer-division key→class convention.

v2 adopts that SINGLE documented compatibility convention, not a band sweep or a
nearest/order assignment. This is an inferred nominal target label, NOT independent
physical-frequency or ground-truth verification. We do not assert the generator
frequency equals the measured event frequency, correct display latency, or call M
physical jitter. Single-target offline run ordering/generalization limits remain.

## Only label API change

New pure module `src/cfeg/mamem_events_v2.py`; tests `tests/test_mamem_events_v2.py`.
Exports same `FREQUENCIES`, `marker_features`, and
`parse_main_trials(din,total_samples,include_metadata=True)` record schema asv1.
May import v1 pure `_scalar`, `_require`, `marker_features`, `FREQUENCIES`; do not editv1.

For EACH aligned group, require>=2 strictlyincreasing finite timestamp events.
`period_integer_ms = floor(sum(diff(timestamp_ms))/number_of_intervals)`;
`key = floor(1000/(2*period_integer_ms))` using exact integer division once periodint.
Require period_integer_ms>=1. Mapping zero-basedclass `{6:0,7:1,8:2,9:3,11:4}`;
nominal references `[6.66,7.50,8.57,10.00,12.00]` separately. Unknown key STOP.
Do NOT use floor(f_est), nearest label, tolerance band, mean/median swap or ordering.
Exactly50ms and exact1000/24ms mayproduce unsupported keys10/12: tests must show STOP,
not introduce special cases. This convention is intentionally not universally valid.

All v1 sample bounds/count/23groups/8adaptation/15main/3perclass-contiguous checks
remain and happen before M extraction; no skipped shortgroups or frequency/range zip.
Use Python sample1−1 indexing from author MATLAB semantics, not MOABB stim-row writing.
No stim row is written or fed into EEG. Query returns metadataNone and nevercallsM.

## Unchanged science and control budgets

All roles, signal features, PSD normalization, source-exclusive oracle/priors,
Q/Q2/QM/SHAM inputs, alpha, 2harmonics/2sec, k1/2, bootstrap/benefit/harm/calibration
criteria and offline-cost caveats in `mamem_recorded_event_source_v1_contract.md`
remain. Reuse frozen `mamem_signal_v1.py` and `mamem_shrinkage_v1.py` unchanged.
No hyperparameter/architecture/feature policy sweeps and no new efficacy candidate.

Fresh v2 round deadline2026-09-13T14:00:00Z. This does NOT extend any old attempt.
At most80actualmultioutputfits, generatedintegration0fits, developmentS0010fits.
Generated phase must run parser→features→source fold preflight on4FAKEpeople×a/b,
verify exclusions/sham eligibility/counts; do NOT fit gates again or summarizeaccuracy.
Use canonical generated event periods `[75.5,66.5,58.5,52.5,43.5]`ms (keys6/7/8/9/11),
tiny person-varying timing residuals; synthetic EEG nominal sinusoids are integration
canaries, not measured timing or a positive efficacy world. No label-policy selection.

Single S001a/b development attempt then single S002–S011a/b preparation/fit attempt.
Same22pinnedexistingarchive members, each<=512MiB,newextraction<=4GiB, featurecache<=1GiB,
8GiBfree+remainingextraction+1GiBcache+1GiBworktree reserve. Newnetwork/PDF0.
Raw/IO worker2GiBAS/90CPU/120wall,fitworker4GiBAS/600CPU/600wall, BLAS1.
Perphase/model STARTED/COMPLETE, exclusive child claims/activeparent, nooverwrite/retry.
All source file eligibility and20foldpreflights sealed before any efficacy fit.
If schema/unknownkey/info/numerics fails: preserve and STOP; no file/subject replacement,
no second labelpolicy, no source-cohort-informed fixes. Budget80 is a cap, not a promise.

## Coordination and runner API

Root owns this shared contract, source provenance/reports/SQLite, integration and
actual execution. Parser agent owns only events_v2/test in its isolated worktree.
Runner agent owns only `scripts/analysis/run_mamem_source_v2.py` and
`tests/test_mamem_source_runner_v2.py` in its own isolated worktree. No other writes.
Read and adapt v1 runner carefully; do not modify/restart v1 or DINdiagnostic.
CLI freeze/execute(generated,development,real)/io-child/fit-child asv1. Fresh paths:
- `/home/whwovy/data/mamem_recorded_event_source_v2`
- root `docs/reports/mamem_recorded_event_source_v2_run`
Freeze manifest must assert exact pin keyset incl newv2contract/parser/runner and
unchanged v1features/engine/helper hashes. Sourcearchive/inventory/S001a pins samev1.
Guard input/trial/cache exact paths and role binding; no broad extraction.
Parent must validate child report status/scope/IDs/counts and hashes before COMPLETE,
not solely exit0; outputsize64KiB perstdout/stderr, directreports/code<=2MiB.
Fullmodels/cache outsideGit, exact real10people×2k×4arms=80 unique fits required for
efficacy COMPLETE. Generatedpreflightcounts8folds/240scalaroracles/0fits.

Generatedtests/unit smoke only in agent worktrees; no raw/network/SQLite/actualfit.
Use existing root `.venv/bin/python` read-only, separate pytest tmp/noinstalls/symlinks.
Budget each writingtree<=256MiB+testtmp, nocleanup. Integration parser→runner→root
combined tests→freeze→generated→development→real. Root may stop beforeanylaterstage.

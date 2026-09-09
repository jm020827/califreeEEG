# Numerical stability v1 — frozen interfaces and ownership

Science contract: configs/analysis/numerical_stability_v1.json SHA
`04d0968d478c795e8a0860c561fcfb881b9d510b5777c1a7ff45e65764b6fbe8`.
No human numerical input or old numerical-policy edit is authorized.

| Lane | Owned new files | Execution | Integration |
|---|---|---|---|
| operator, goal_scope_review | src/cfeg/analysis/numerical_stability_operator.py; tests/test_numerical_stability_operator.py | isolated temporal-runtime tree, toy tests only CPU1 <=300s/64MiB | first |
| audit, metadata_loop_history | scripts/audit_numerical_stability.py; tests/test_numerical_stability_audit.py | isolated temporal-cold tree, toy tests only CPU1 <=300s/64MiB | after producer archive interface |
| root | contracts, scripts/run_numerical_stability.py, tests/test_numerical_stability_runner.py, reports, SQLite | sole registered run/audit; root toy <=300s/64MiB, fulltests <=900s/600MiB; numerical outputs <=128MiB | final integration and verification |

Reuse existing two trees by new branches from this contract commit; keep old branches and untracked
test outputs. New checkouts about22MiB total; incremental test/output reservations total920MiB plus
checkout remain below1GiB. No new environment or worktree; main .venv is shared read-only, each
lane sets PYTHONPATH to its own src and isolated temporary output. Report every test invocation's
elapsed time and directory; root sums all reported test/numerical times, including failed checks.

## Operator API

`projector(s: Tensor, b: Tensor, c: Tensor, method: str) -> Tensor`
returns differentiable C-normalized F of the same shape. Two exact method names from contract.
Float64 CPU only, no broadcasting, no I/O, no human/code configuration changes. Accept independent
S/B/C parameters; when B and C are the same Tensor autograd contributions must sum. First-order
only. Check all input symmetry/finite/SPD before any correction; validate top gap without clipping;
F must be finite and original-pair Fro residual <=1e-12. No casewise method fallback.

`diagnostics(s,b,c,f) -> dict[str, Tensor]` returns detached batch-shaped scalars:
`lambda`, `residual`, `normalization_error`, `symmetry_error`, `raw_h_skew`.
Lambda=tr(SF)/tr(BF). Relative skew uses Fro/max(Fro,tiny). Raw H is calculated from input-symmetrized
S/B by the two Torch solves, BEFORE H symmetrization; it is diagnostic, not a rejection criterion.
Diagnostics are not automatically evidence of correct top eigenpair; independent audit checks that.
Existing `_LeadingProjector` may be imported unchanged for N1. N2 saves gvd roots/vectors and uses
the frozen VJP, never differentiates LAPACK's arbitrary eigenvector bases or eigenvalue outputs.

Required toy coverage: closed-form2D rotation embedded8D, exact repeated lower roots, exact repeated
B roots, asymmetric/nonfinite/nonSPD inputs, top ties and threshold sides, dense noncommuting G,
S/B/C and tied B=C gradients, scalar/batch F AND VJP parity, joint powers-of-two scale relation.
Do not use seeds20260910/11/12 or the saved legacy NPZ in toy tests.

## Producer archive

CLI `--output FRESH_ABSOLUTE_DIRECTORY`, fixed repo contract only, no input path CLI. Validate legacy
result/NPZ exact path/hash/bytes/mode before decoding only24 allowed C/S members. No raw EEG/M files.
Every input/output directory and file must be unaliased; fail on existing output. CPU1, noGPU.
Before fixture generation write immutable `start.json`: schema, design SHA/path, code_pins mapping,
revision, versions, time, fixed budget and exclusions. Runtime files are pinned after integration and
before execution. Count elapsed walltime/CPU and exact bytes; enforce timeout/output caps.

Literal fixture conventions supplement the mathematical contract: `sym(x)=(x+x.T)/2`;
`q,r=np.linalg.qr(raw); q*=np.where(diag(r)<0,-1.,1.)`; normalize sym probes by np.linalg.norm(...,'fro');
bounded-shape `u=exp(logits-logits.max()); r=8*u/u.sum()`. Loop order seed, C condition, scale, lower
spectrum, top gap, denominator. Old960 cases precede new192 cases in inputs; id old=0..959,
new=960..1151. W and directions for old cases use seed20260912 and new case probes their seed's draws.
All S/B/C/probe/direction arrays are symmetrized; reference solves these stored floats, not an idealized S.

`inputs.npz`, exact members:
`s,b,c,g,ds,db,dc` float64[1152,8,8]; `gradient_mask` bool[1152];
`negative_s,negative_b,negative_c` float64[9,8,8].
The nine negative controls intentionally include NaN; they are not scientific observations.

`measurements.npz`, for each N1/N2 prefix (short names `n1`,`n2`):
`{prefix}_f`, `_gs`, `_gb`, `_gc`, `_batch_f` float64[1152,8,8];
`{prefix}_ok`, `_gradient_ok` bool[1152]; `{prefix}_negative_rejected` bool[9].
Unused/failed numeric entries are NaN sentinels, never valid projectors/gradients. Gradient entries
are populated only at209 mask indices. Scalar forward checks all1152; scalar VJP uses loss=sum(g*F).
Batch forward runs all valid fixtures together once per method and records whole-batch failure as
failure; no dropping failed rows. Negative controls are separate nine calls per method.

`result.json`: schema `cfeg.numerical_stability.result.v1`, status `MEASUREMENTS_COMPLETE` or
`MEASUREMENT_FAILURE` (NOT scientific/numerical promotion), `start_sha256`, `inputs` and `measurements`
descriptors {path,sha256,bytes}, `case_metadata` list1152 matching exact grid, method failures with
case/stage/message, elapsed wall/CPU/bytes, versions and exclusions. Expected negative rejection is
not a method failure. All four files0400. Unexpected runner crash writes immutable failure.json;
that takes precedence over result. No automatic rerun/repair after registered execution.

## Independent auditor

CLI `--result ABSOLUTE --sha256 EXTERNAL_PIN --output FRESH_JSON`, CPU1, <=remaining7200s
numerical budget and128MiB output reservation. Read only generated archive files after binding hashes,
inventory/regular non-symlink/0400, start/code/design and expected case counts. Result artifact paths
must be exact same-parent `inputs.npz` and `measurements.npz`; reject unexpected paths/arrays/members.
Auditor imports neither producer nor cfeg. Reconstruct RNG/grid/directions literally, including original
known matrices from the exact allowed24 generated legacy members; do not read any human files.

Use80dps mpmath spectral B inverse square root and eigsy; exact float conversion via as_integer_ratio.
Compare F in C-whitened metric (reference Cholesky C is allowed), original-pair residual, C trace,
symmetry and top-root. Every209 selected input gets S/B/C/joint highprecision central differences
at1e-20 and1e-25 with G FIXED. Reference inconsistency => INCONCLUSIVE_REFERENCE, not candidate
rejection. Recheck120dps on indices851,960,1055,1056,1151. Require all saved failures/masks/gradient
coverage and negative rejections to agree with final status. Validate scalar/batch outputs for everycase.

Output immutable receipt: per-method eligible/rejected and all maxima/failure cases, reference precision
and FD consistency, exact source/input/code pins, no-human scope, every-case metrics/209 directional
metrics, elapsed wall/CPU/bytes. Status `NUMERICAL_STUDY_VERIFIED` only for intact fully audited archive,
even if methods rejected; separately `recommended_method` may be null. A method requires everyfixed
gate. Missing reference/integrity cannot produce verified promotion. Tests must detect wrong lower
eigenpair, zero/scaled F, detached gradient, missing case, forged descriptor and unauthorized path.

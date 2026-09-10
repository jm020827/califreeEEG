# One-shot reference ridge: generated-only primitive contract

2026-09-11, before implementation or generated outcomes. Same head-motion candidate;
not a new human efficacy experiment. While fixed public cohort acquisition runs, root
implements independent pure-tensor primitives. No raw data, new search or GPU required.

## Exact mechanism

Time-center each support trial/channel and every reference row. Concatenate k trials,
form Cxx=XX'/n and Cxy=XY'/n, and solve W=(Cxx+lambda*s*R)^-1*Cxy,
where s=max(trace(Cxx)/C,1e-12). This is defined at k1; no pairwise TRCA statistic.
References are sine/cosine pairs for3harmonics, arbitrary common phase.

Query Z=W'Xq; score its energy projected onto the centered reference rowspace:

`score = sum((Z Y') * solve(Y Y', Y Z')') / sum(Z^2)`.

Zero projected-output energy<=1e-12 maps to0. Reject singular reference Gram instead
of inventing a data-dependent rank. Solve only, no eigenvectors/output whitening.
Reference pair phase rotations leave score invariant. Global Wscale must cancel;
this scale statement requires both energies to remain above the fixed zero-energy floor.
anisotropic spatial changes need not cancel. This corrects the originally considered
unnormalized-reference Frobenius score's finite-window Gram weighting/1/H ceiling.

Q diagonal prior: dQ=epsilon+(1-epsilon)*C*softmax(a), epsilon=.25; traceR=C.
Frozen-Q residual: v=tanh(h), d=dQ.detach()+alpha*epsilon/2*(v-mean(v)),alpha=.4.
Trace unchanged and d>=epsilon*(1-alpha)=.15. Uniform residual logits must do nothing.
Scalar M must map to spatially nonconstant logits. These exact constants fix primitive
tests, not a complete human hyperparameter/feature/training contract.

## Bounded verification

At most3generated targeted-test invocations, CPU only, per-fixture20000numeric elements,
output16MiB, no human tensors/queries/held60. Check k1solve/scaling, >=2channel relative
margin actuation, gradcheck plus finite differences, phase rotation, trace/SPD bound,
Q-detach, zeroenergy and singularreference rejection. Stop at budget if invalid.
Synthetic actuation does not establish measured M relevance; source learning, actual
Qfeatures, SHAM retraining and subject-split evaluation remain separate work.

Root owns module/tests; reviewer is read-only. Existing failures and completed v1/v2
metadata budgets remain unchanged. Acquisition process owns only its new directory,
journal and one final observation path; no overlapping writers or new worktree.

## Executed generated result

One targeted invocation completed9testsPASS,CPUfloat64,0.274s test body (import/startup
excluded). It confirms k1solve, unit scaling, anisotropic relative-margin change,
scalar auxiliary-input gradient with Q frozen, finite-difference gradcheck, phase
invariance, trace/SPD bound, zero-energy score and invalid inputs. Independent static
math/code review found no blocker. No float32 precision validation, measured-M learning,
SHAM retraining, human accuracy or calibration-savings claim follows from this result.
Remaining2test-call capacity was not consumed just to repeat unchanged tests.

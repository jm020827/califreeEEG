# M-conditioned TRCA prior — synthetic v1 contract

2026-09-08, baseline8b78cd0. User approved the synthetic engineering next step.
This contract and its JSON are committed **before evaluation on the declared seeds**.
No human EEG, metadata packets, saved human outcomes, held60, retired inputs,
old experiment runner, network or dataset constructor is part of this stage.
The previous [design review](metadata_learning_covariance_design_review.md) remains historical.

## Scope and ownership

Root owns this contract/config, synthetic generator/learner/runner/tests, research journal
and sole research-workspace writes. A separate operator worktree owns exactly
`src/cfeg/analysis/metadata_trca_prior.py` and `tests/test_metadata_trca_prior.py`.
Read-only reviewer shares main. Contract -> operator and synthetic consumer -> root
integration -> full tests -> one declared suite -> report is the integration order.
No concurrent writers in one tree. Existing29worktrees preserved; one new checkout
budget32MiB and total testing/artifacts1GiB versus about255GiB available. No installs,
shared mutable environments, cleanup or push. Tests use the existing interpreter and
distinct temporary/cache paths, with explicit PYTHONPATH for the operator worktree.

## Frozen arrays and operator API

All arrays are real finite float64 unless metadata is explicitly NaN-missing.
`support` has shape `(k, classes, bands, channels, samples)` with k>=2;
`query` has shape `(n, bands, channels, samples)`.
This stage uses12classes/8channels/1synthetic band/125samples at250Hz; this is
not native wearable preprocessing, all8human conditions or original-paper reproduction.

Module API (no filesystem/network/data loading):

```python
trca_matrices(trials) -> (S, C)                 # trials[k,d,t]
trace_normalize(values) -> positive_diagonal   # [...,d], trace along last axis=d
residual_prior(q_prior, delta, available, bound=0.2) -> diagonal
fit_trca(support, prior, gamma) -> TrcaModel
score_trca(model, query, filter_weights) -> (scores, correlations)
```

`TrcaModel` contains `filters[bands,channels,classes]`,
`templates[classes,bands,channels,samples]` and diagnostics.
`prior[bands,channels]` is class-blind, positive and trace=channels.
Templates are unweighted support means. Native S uses uncentered cross-repeat
products; C centers the temporally concatenated trials, matching pinned
toolbox `_trca_U_2` revision3344bd199daf78888e364d9db00ae7d8128d2b5f.
Solve `S w = lambda [C + gamma*trace(C)/d*diag(prior)] w`.
Normalize each filter by its own denominator; signed ensemble flattened correlation
and linear filter-bank weighted scores match the pinned ETRCA convention.
No squared/absolute correlation, score fusion, query-derived R or candidate selection.

Use deterministic descending generalized eigenvalues, with the pinned SciPy eig
normalization convention. Reject nonfinite/complex, nonpositive denominator,
zero-energy and a top-eigenvalue gap <=1e-10*max(1,maxabs(eigenvalues)).
Reject zero query/template projected variance rather than silently report a score.
Gamma must be finite nonnegative. Gamma0 is an operator compatibility path, not k1.
Reject invalid shape, negative prior, trace mismatch, unsupported mask dtype and
nonfinite available deltas. No inputs may be mutated.

`residual_prior`: clip available delta to[-bound,+bound], missing delta=0;
multiply Q diagonal by exp(delta), then trace-normalize. If every delta in a band
is unavailable, copy that band's Q prior exactly. Partial missing can change a
missing channel through common normalization; no claim of channelwise exact fallback.
Final QM/Q ratio is bounded by exp(+/-2*bound), not exp(+/-bound).

## Proxy and features: one definition, no result-driven selection

For k3/5 use only blocks[0:k] for learner inputs. Block5 (zero-based) is a separate
future repeat for proxy supervision; blocks6/7 are classification queries.
Both budgets use the same proxy block but no support/proxy/query overlap.
Class labels of support and source proxy are legitimate calibration supervision.
Evaluation proxy targets and query labels never enter fitting/scalers/priors.

For each channel/band, target is
`log(mean_class,time((future_repeat-support_mean)**2) / support_global_power)`.
Global power is the support mean squared amplitude averaged over all trials/classes/
channels/time within each band, floored at1e-12. This scale is input-only.
The target mixes instrument noise, neural/phase variability and template error;
it is a **repeat-disagreement proxy**, never an observed noise variance.

Six Q features, class-aggregated per channel/band:

1. log(channel support power / global support power), floor1e-12;
2. log(off-reference energy fraction), clipped[1e-6,1], from QR projection onto
   known class sine/cosine fundamental and second harmonic;
3. mean one-minus centered correlation of distinct support-repeat pairs;
4. log(k);
5. fraction of observed support impedance packets;
6. synthetic oracle log-noise-variance only in q_sufficient, zero otherwise.

Feature6 is an **artificial oracle-Q control**, not an EEG-derived measurement
available to a real deployment. It tests saturation of side information and must
never be exported as a human feature. The other scenarios use zero in this column.
Information sufficiency is for the full oracle-Q vector; the restricted row-wise
linear predictor need not exploit it. Centered M supplies cross-channel transformations,
so a failed null screen is not a refutation of that conditional information property.
All synthetic data use one interface and protocol, so no variable domain-ID benefit
is present. A real-data schema must separately bind the shared context features.

M has two features: mean log(impedance) centered across available channels in a
support band, and within-support population SD of log(impedance). Units are arbitrary
positive synthetic impedance-like values, not a physical kOhm-to-noise mapping.
Nonpositive/infinite observed M is an error; NaN is missing. No query M is accepted.
Means use observed packets only, centering uses channels with at least one observation,
SD uses ddof0 (one observation gives0), and a wholly missing channel is zero-filled
with available=False. The observed fraction remains a common Q feature.
The mean/std definition deliberately discards detailed block order; this limitation
is fixed and not repaired after seeing outcomes.

## Fitting and controls

24 artificial participants; outer evaluation is participant index modulo3.
Each fold fits16participants and evaluates8. Training rows are the balanced cross
product of participants x two budgets x channels x bands, so each participant has
equal total mass. All channels/classes/windows are not independent participants.
No hyperparameter search. Ridge coefficient0.1 uses mean squared loss and
standardized training features; intercept unpenalized. Constant SD uses1.

First fit Q to the proxy, then freeze Q. Fit residual models to fixed training
target-minus-Q predictions: Q2 uses Q features with an intercept; QM uses only
the two standardized M features without an intercept; SHAM_REFIT has the same
M feature dimension, ridge, fitting budget and no intercept as QM.
This is a restricted incremental predictor, not an optimal conditional-information test.
Q2 controls extra fitting opportunity; dimension match with QM is not claimed.
Residuals use in-sample Q errors as an explicit engineering simplification; they
are not cross-fitted independent proxy validation. Outer participants are disjoint.

Predicted Q log penalty is clipped[-3,3], exponentiated and trace-normalized.
Residual predictions are clipped[-0.2,0.2] and applied through residual_prior.
All regularized arms have gamma0.1 and trace8; FULL alone has gamma0 and ISO uses I.
Positive proxy output maps to larger penalty by hypothesis, not physical proof.
This link can fail when large disagreement comes from useful neural response.

Controls: FULL, ISO, Q, Q2, QM, SHAM_REFIT, PERMUTED, STALE, MISSING.
Fixed participant derangement within fit/evaluation separately replaces complete M
packets for SHAM_REFIT. PERMUTED applies the same eval derangement to frozen QM.
One deterministic derangement per scenario/fold, never optimized. Not a conditional
randomization test. STALE repeats the first support M packet across the prefix.
MISSING denies numeric M after Q is fixed and must reproduce Q exactly; this is
a numeric-denial ablation preserving the common mask, not a natural-missingness model.
Unit tests additionally cover natural full/partial missingness and changed Q masks.

## Fixed artificial generator and screen

NumPy SeedSequence([seed,scenario_index,participant_index]); no human inputs.
Frequencies9+0.5*class Hz. Each participant/channel has Gaussian risk and a positive
lognormal signal gain. Base waveform sin(f)+0.3cos(2f), mean-centered per trial.
Independent Gaussian sample noise has SD `0.7*exp(0.6*risk)` in informative,
independent and q_sufficient. Support/query noise is drawn separately.
M is `20*exp(risk + 0.1*block_jitter)` in informative/q_sufficient;
independent uses an independently drawn channel risk for M.
Phase_only has constant noise SD0.7 but per-trial channel phase jitter with
SD `0.5*exp(0.5*risk)`, and M follows risk. Signal gains are exp(N(0,0.2)).
One band, no realistic amplifier simulator or claim of domain fidelity.

One declared suite seed26090817. Development unit fixtures use unrelated fixed
seeds and do not score/tune this evaluation suite. Generator/schema bugs may be
fixed with transparent history, but scientific changes after suite exposure require
a new proposal, not a silent rerun until positive. Numerical/audit replay of the
unchanged suite is verification and must be labeled as such.

Report every scenario/budget/arm's proxy MSE, accuracy, participant help/tie/harm,
QM−Q/Q2/SHAM, prediction changes and trace/fallback diagnostics. No significance
claim from this small fixed suite. k3/5 synthetic label counts36/60 and accuracy80%
attainment are descriptive; no measured wallclock or actual calibration savings.
FULL/ISO have no proxy predictor and their proxy MSE is null. Q proxy uses raw
regression output; residual-arm proxy uses raw Q plus the same clipped residual
passed to the prior, before base clipping and trace normalization. MISSING proxy
equals Q. These predictions are not the normalized R or calibrated noise variance.
Only the observed synthetic k3/5 grid is summarized; k0 is not measured in this suite.

Descriptive screen at k3: informative QM proxy MSE<Q and accuracy>Q,Q2,SHAM;
independent and q_sufficient abs(QM−Q accuracy)<=1/60. Phase_only is a mandatory
counterexample diagnostic, not a required success. Passing merely motivates review;
failing closes this synthetic candidate as unestablished, without retuning.
Neither status authorizes a human/source export or a held60 run.

## Verification and remaining boundary

Test matrix: native gamma0 on artificial support/query; identity regularization;
trace/positive definiteness; missing exactness; bounded residual ratio; channel/class
permutation with synchronized feature order; uniform amplitude units; mutations;
k1/zero/near-zero/eigen-degenerate/invalid inputs. A common invertible channel
gain is not treated as a new method. Random eig fixtures and synthetic suite outcomes
are not Wearable efficacy. Existing code/configs/closed results are preserved.

The next human stage, if later selected, needs a separate source-only sufficient
statistics/raw export contract, native full preprocessing, shared domain cues,
fit/eval separation, frozen practical criteria and independent audit. This stage
does not implement it and does not claim current correlation caches are sufficient.

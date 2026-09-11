# Block-scaled source router integration v1 — pre-execution contract

2026-09-11. Parent8d95cc8. Original low-calibration SSVEP objective unchanged.
This is the next generated integration gate, not another human efficacy replay.
Old144/336-fit negatives remain closed. Root owns all writes; reviewers read only.

## Mechanism and one nominated candidate

SAFE uses the existing source-only scaler, then divides blocks0:5,5:64,64:123,
and optional123:125 by5,59,59,2 respectively. This is AFTER standardization.
Zero-initialized shared linear logits over13experts, balanced probability-mixture
NLL, float64 CPU1. No clipping of final logits, no forced positive transfer floor.

AdamW lr.05,weight_decay.01,betas(.9,.999),eps1e-8 proposes one update per step.
SAFE evaluates factors1,1/2,...,1/128 in order. Accept first whose same-training
NLL <= previous NLL+1e-12 AND maximum episode-centered logit change <=1.0.
If none, restore parameters exactly. Adam moments/timestep still advance once
per proposal, including rejection: this is an explicitly distinct safeguarded
AdamW proposal algorithm, not unchanged AdamW. No validation/evaluation labels,
losses, or logits enter acceptance. Record proposal/accepted/rejected/factor,
loss-evaluation count, moment steps, actual parameter/logit changes.

ORIGINAL (old scaler/unscaled AdamW) and SCALE (block scaling only) are two
engineering ablations. SAFE is nominated BEFORE results and cannot be replaced
by whichever ablation wins. The safeguard guarantees only source-training NLL
nonincrease within tolerance, not global optimum, target stability, or M efficacy.
Centered logit change1 bounds per-step pair gaps by2, not accumulated gaps.

## Fixture frozen in code before first execution

Use actual Dataset.features/batch, support ridge, harmonic projection, Q123 and
aux125,13experts,3classes. Bank-only b00..b03 have acquisition groups[-,-,+,+].
Training t00..t07 and evaluation v00..v07 enumerate all8 combinations(h,m,r)
in{-1,+1}³; each has3speeds. Total20people/60runs. Recipients are never in bank.
Source pool is exactly the4bankpeople, yielding12sourceexperts+self. Existing
sorted-ID one-step SHAM maps bankM from[-,-,+,+] to[-,+,+,-]. It is a partial
correspondence negative control, NOT independent M or an obligatory chance result.

Two deterministic worlds share identical support/Q/common/M and fitted experts:
POSITIVE query relevant spatial group=m; NULL relevant group=h. Each query has
all3classes twice. Within NULL, EEG/query probabilities for fixed(h,r,speed,split)
are identical for m±; the nuisance factor is fully balanced, not a random shuffle.
Q/common never contains h,m,ID,query index or query labels. Q2 depends only on r.

Synthetic EEG uses two disjoint posterior channel triplets. Bank k5 supports
encode one group; recipient supports are balanced across the two groups.
Query relevant group carries class-frequency response; the other group carries
a wrong-class-frequency distractor. M thus tells which acquisition subspace is
relevant, not which class was attended. This is deliberately favorable artificial
context, NOT a physiological model of gyro, not a corruption robustness objective.
Time references are the existing5.45/8.57/12Hz,3harmonics,128Hz/128samples.
Code fixes amplitudes/noise/phase/deterministic nonduplicate Q nuisance before
the run. Evaluation waveform phase/nuisance differs by a fixed split offset;
this is an engineering holdout, not independent EEG validation.

## Exact finite budget

- Two worlds x3optimizer variants x4arms(Q,Q2,QM,SHAM)=24fits,100proposals each.
  Fit and freeze all12heads per world before constructing its evaluation batch.
- 2400 total AdamW proposals; SAFE8fits x100x8 <=6400 trial evaluations.
  Additional forward loss records are counted separately, not hidden as updates.
- 720 unique class-ridge solves (60runs x4k x3classes), shared across worlds.
  Only k1 recipient routing is trained; source experts use k5. No k-policy or
  calibration-saving estimate is made from this engineering fixture.
- One experiment attempt, no retry/hyperparameter/seed/fixture rescue, numeric
  wall deadline120seconds, persistent output<=16MiB, each world's retained
  sufficient-statistic arrays<=4MiB, transient waveform block<=1MiB.
- Nontraining unit suite at most2calls,<=30seconds each, no human IO or fits.
  Cases cover /block-size-after-scaler, accept/shrink/reject/restore, nonfinite
  candidate rejection, missing classes, shape guards, frozen moments policy.
- Independent audit of generated output only<=30seconds, no refit/optimizer replay.
- No raw/cache/human stored model access, new package, GPU, paid resources,
  outreach, held60, source39, Choi reactivation, or new worktree.

## Gates and decision before outcomes

Required implementation invariants: source/evaluation/bank separation; recipient
excluded; Q/sharedfeature identity when only h/m change; NULL paired probability
identity; SHAM multiset preserved; scaler frozen; expert joint permutation and
source-pool-order invariance; query-label change cannot alter features/probabilities;
query EEG change cannot alter routing inputs; exact no-transfer endpoint;
new state roundtrip; final-state NLL recomputation. Failures close this version.

SAFE source accepted NLL jumps must be<=1e-12 and centered-logit jumps<=1+1e-12.
For the deliberately informative evaluation world, nominate SAFE only if QM
beats both Q andQ2 by >=.02 NLL and>=.10 balanced accuracy, and frozen M-to-SHAM
intervention changes gates and class margins by>1e-4. Record SHAM's learned result
but do not mistake invertible/correlated sham information for independence.
For NULL, QM's apparent NLL improvement overQ must be<=.01 and accuracy gain<=.02.
Finite optimizer differences are not a formal conditional independence test.

Keep all24results even if individual criteria fail; no best-arm substitution.
If required SAFE gates pass, prepare a separate human-successor contract using
unchanged exposed development participants and matchedQ/Q2/QM/SHAM. Do not run
that human experiment in this generated phase. If gates fail, report and close
the nominated candidate pending a genuinely new reason, not more retries.

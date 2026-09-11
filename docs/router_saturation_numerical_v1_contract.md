# Known-optimum router saturation diagnostic v1 — frozen before numerical execution

2026-09-11. Parent human candidate `2f97ec3` remains CLOSED_NEGATIVE. User approved
the proposed outcome-free diagnostic. This is not a human efficacy rerun.

## Hypothesis, alternatives, and decision

H1: summing many separately standardized self-indicator coordinates under the
fixed AdamW update can overshoot a known interior mixture optimum and saturate
the self gate. H0/alternative: convergence remains adequate, or self selection
is appropriate because the mixture objective really prefers self.

We manipulate exact duplicated coordinates, not the actual human feature block.
This establishes a possible numerical failure mode, not its causal attribution
in the human run. Averaging the duplicate block AFTER standardization is an
engineering control, not a selected human intervention. Coordinate duplication
changes optimizer geometry/decoupled weight decay; this is not a pure dimension
effect at identical parameter-space regularization.

## Fixed fixtures and identifiable optimum

13 experts: self first, 12 identical source experts. Two queries have labels 0,1;
each listed pair is the experts' probability of the correct class, with the other
class receiving the complement. Labels are balanced. `s` denotes total self mass.

- INTERIOR (two identical episodes): self (.9,.2), source (.4,.8).
  NLL(s) = -.5 log[(.4+.5s)(.8-.6s)]. Unique optimum s=4/15.
- BOUNDARY (two identical episodes): self (.9,.8), source (.4,.3).
  Both correct probabilities increase with s; optimum s=1 (softmax limit).
- CONTEXT (four episodes): M=+1,+1 uses INTERIOR; M=-1,-1 swaps self/source.
  Q/common are otherwise identical. Q-only optimum s=.5 for every episode;
  Q+M optimum s=4/15 for +1 and 11/15 for -1. Synthetic M is a known context
  variable, NOT an actual acquisition measurement or evidence of EEG efficacy.

All start at uniform 1/13. For equal sources, gap(self,one source) equals
log(12)+logit(s), not logit(s). We compare three representations:

1. ONE: one self-indicator coordinate.
2. SUM59: 59 exact copies of that coordinate, summed by the linear layer.
3. MEAN59: same 59 coordinates, multiplied by 1/59 after source standardization.

CONTEXT additionally compares Q and QM. QM appends self_indicator*M as one
non-duplicated coordinate, not averaged. Router means/std are fit once to each
synthetic training fixture. M-flip changes only that raw M feature at evaluation;
labels, expert probabilities, all Q features, learned coefficients, and scaler
remain fixed. Report both gate change and probability/class-margin change.

## Exact budget and implementation

- 6 fits (INTERIOR/BOUNDARY x 3) + 6 (CONTEXT Q/QM x 3) = 12 fits total.
- Each fit exactly 100 full-batch AdamW updates, lr=.05, weight_decay=.01,
  betas=(.9,.999), eps=1e-8. Float64 CPU one thread, zero-initialized layer.
- Reuse frozen Router/scaler and probability mixing; no production code edits.
  MEAN59 uses an explicit post-standardization scale in the diagnostic only.
- No random fixture, seeds, learning-rate sweep, early stopping, checkpoint
  selection, or extra fits. Record every state 0..100 (including final), NLL,
  self weights, logit gap, gradient norms before updates, and parameter norm.
- Loss optimum is independently calculated directly from correct-class
  probabilities at analytical s, not by another optimizer. Also check derivative
  signs around interior optimum and that the production/plain loss agrees.
- Engineering convergence: final NLL gap <=.001 and max |s-s_opt|<=.02.
  Boundary permits limiting s=1. Actuation: both gate and class-margin changes
  >1e-12; neither criterion is human accuracy or calibration efficiency.
- One numerical attempt, <=30 CPU/wall seconds for numerical body, <=2 MiB
  simultaneously held synthetic fixture tensors, <=4 MiB total report/journal.
  Syntax/style checks are not fits. On numerical error/budget/finite failure,
  preserve started/partial records and STOP; no automatic rerun of this version.
- Hard counter <=1200 updates; unique no-overwrite start and report files.
  No full repository test suite, package installation, GPU, worktree, human
  numeric/model/cache/raw access, held60, outreach or paid resources.

## Interpretation and next gate

All 12 outcomes are retained, including failed convergence or absent M action.
If SUM59 alone fails while ONE/MEAN59 reach the known optimum, retain feature-
block scale as an engineering hypothesis; do NOT rewrite the prior human result.
If the boundary control also prefers self, distinguish legitimate self choice
from the interior failure. CONTEXT checks whether a useful synthetic side input
can affect final probabilities after learning, not just have nonzero coefficients.
Any human successor requires a separately frozen candidate/controls/budget and
source-only validation. No automatic promotion or new human run in this phase.

Root sole writer; independent math review is read-only. Existing 41 worktrees
and eight untracked pytest directories remain untouched.

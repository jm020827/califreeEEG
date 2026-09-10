# Head-motion metadata learning: frozen exploratory efficacy contract

2026-09-11. One candidate, following metadata reader freeze e36f2d6.
The original low-calibration closed-set SSVEP objective is unchanged. This is not an
OOD detector, damage-repair task, or a claim of the first one-shot SSVEP method.

## Evidence and mechanistic prediction

All 96 acquired files passed the metadata-only check; 16 subjects have all three speeds.
Observation SHA256: d2292aa61886d51bfcd05251371f710a818f9f66cfc306a4ebc4d23a54cc485b.
No waveform or human efficacy outcome has been read at this contract's creation.

Hypothesis: measured head motion during the same paid calibration examples predicts
a useful small adjustment to an EEG-quality-conditioned spatial regularizer. At k=1/2,
QM should outperform Q, an equal-capacity stronger EEG-only Q2, and retrained mismatched
metadata SHAM. A source-selected calibration budget should then use fewer acquired
trials while maintaining accuracy. The anisotropic prior's score actuation was checked
on generated data; human usefulness is unknown. A null closes this bounded mechanism,
not every possible use of metadata.

## Fixed replay and extraction

- Subjects s01 and s03–s17, all three speeds 0.0/0.8/1.6. No outcome-based exclusion.
- Exact roles from `mobilebci_cohort_metadata_v1_observation.json`: first k/class,
  k={1,2,3,5}; fixed query indices 40..59. Full chronological prefix trial cost is charged.
- Per-device sample origin is the already frozen candidate round((event.time−1)*fs/1000).
  No fitted clock correction, label-dependent shift, alternate windows, or index search.
- EEG channels Pz, PO3, POz, PO4, PO7, PO8, O1, Oz, O2 at 500 Hz.
  For each allowed trial, read [marker−0.5s, marker+1.5s), apply a causal fourth-order
  Butterworth 4–40 Hz bandpass (SciPy SOS, zero state reset per trial), retain [0.5s,1.5s),
  then time-center each channel. The first second is filter warm-up. No resampling,
  full-run preprocessing, query-derived normalization, rejection, or artifact tuning.
  Here `butter(4, bandpass)` means a fourth-order prototype, eighth-order bandpass
  transfer function (four SOS sections), not a four-pole bandpass.
- M uses only HgyroX/Y/Z at 128 Hz during each selected support [0.5s,1.5s).
  No query IMU features or full-run motion summary. MAT storage requires whole raw_x
  decompression; distinguish physical numeric decoding from permitted feature use.
- Reference: three harmonics at 5.45,8.57,12 Hz, six centered sin/cos rows at EEG rate.
  Frequency is known common task information. All pooled features use all 3k support
  examples equally; no class-specific metadata shortcut.
- Fixed floors: 1e−12 for variance/energy, ratios and normalization scales. Log all
  floor activations. No silent removal of nonfinite samples or features: stop.
- Q54: nine log channel variances; nine log residual/total variance ratios; 36 distinct
  residual cross-channel correlations. Residual removes each support trial's known
  reference span, E=X−XYᵀ(YYᵀ)⁻¹Y, then pools the selected trials.
- M2: average over selected trials of log RMS(||gyroXYZ||), and of
  log(max(std(||gyroXYZ||)/max(RMS, floor), floor)). Native units, not calibrated degrees/s.
- Q2 auxiliary2: corresponding EEG residual spatial-norm log(std/RMS), log(q95/RMS),
  calculated per selected trial and averaged. Not random padding.
- Common5 for every arm: speed, condition ordinal 0/1/2, k, full prefix trial count,
  marker-relative support-ready seconds. Ordinal is a condition/order proxy, not a
  verified independent session clock. Subject/file/session IDs are never features.
  The metadata audit found 44/48 runs share a class sequence: query index/sequence ID
  must also never be features. Ready seconds include the roughly four-second preroll.
- Store role-limited support covariance/cross-products, Q/M/Q2/common, query covariance
  and reference cross-products, query labels and provenance. Raw waveforms are not
  copied into the cache. k-specific features/statistics use only that k's support.

## Finite implementation and data budget

- Generated tests: at most 3 suite calls each for extraction/statistics and trainer;
  each fixture ≤200,000 numeric elements / 2 MiB. No public data in unit tests.
- One exact 96-file raw_x pass, 1 checksum and 1 selective decode/file, 30 s/file,
  600 s compute batch, process address-space 1 GiB; generated feature cache ≤128 MiB.
  Stat before/after; durable per-file journal; first unexpected failure stops, no retry.
  Durable stop serialization/fsync excluded from compute timer. No other human data.
- Actual training: at most 336 prior fits / 33,600 optimizer steps, CPU float64, one
  thread, 1 hour compute wall time, 2 GiB total new outputs. No new install/GPU/paid use,
  external outreach, held60, source39, new dataset, alternate split or feature sweep.
- Code and exact cache/config hashes must be frozen before human fits. Generated-test
  corrections are allowed inside their call budgets; human results do not authorize
  new tuning. Preserve failed attempts and all completed partial results.

## Participant-separated learning

Sort the eligible IDs, assign outer fold by position modulo 3. Within each outer
source list, sorted position modulo 2 defines inner validation folds. Speeds always
travel with their subject. No test features determine standardization, tuning or donors.

For each training split, k and λ in {0.01,0.1,1}, train Q once. Q is a zero-initialized
linear map from standardized Q54+common5 to nine prior logits. Prior diagonal is
0.25+0.75×9×softmax(logits), trace 9. Each feature's source-training mean/std is used
unchanged in validation/test; constant-feature std is floored at 1e−12.

Freeze Q. Q2/QM/SHAM have identical zero-initialized linear maps from standardized
Q54+common5+auxiliary2 to nine logits. Their prior is Q plus
0.05×(tanh(logits)−mean(tanh(logits))), preserving trace and strict positivity.
Q weights, ridge λ, residual bounds and all training resources are shared.

Fit class-specific reference ridge using support covariance-scale regularization and
Gram-corrected reference projection energy, as in the verified primitive. No output
whitening or scalar normalization that would undo prior actuation. Ties pick the lowest
frequency index. A score near its numerical floor is explicitly logged.

Each fit: 100 full-batch AdamW steps, lr=.01, betas=(.9,.999), eps=1e−8,
weight_decay=.01, gradient norm clip=1. Loss is CE(10×projection scores), balanced
within run by true query class, then equally across runs (three per source participant).
Training query labels are allowed supervised source data. Validation/test query labels
are evaluation only. No early stopping, restart, extra seeds or epoch extension.

SHAM: for each recipient run use a nonself donor from the current source-training
participants with the same speed and k. Minimize |prefix difference| +
|support-ready-seconds difference|/10; ties choose lexicographically smaller donor ID.
Validation/test donors also come only from that training source pool. Train and evaluate
SHAM on its assigned donor M, not on a post-hoc swapped trained QM. Standardize the
actually consumed source-training SHAM features. Log donor, distance, and actual M
change. A zero-change SHAM invalidates the measured-metadata contrast, not an efficacy win.

Inner fits: 3 outer × 2 inner × 3 λ × 4 k × 4 arms = 288. For each outer/k select λ
by source inner-OOF Q balanced accuracy only; ties prefer larger λ. Refit all four arms
at the selected common λ on all outer-source participants: 48 fits. Total 336.

Also report no-trained-prior identity-R at that λ, and zero-calibration regularized
reference CCA (λ=.001×mean channel variance, largest squared canonical correlation),
without learned heads or tuning. These are descriptive stronger-baseline checks, not
extra prior fits. k5 is the within-design higher-calibration comparator, not an unlimited
calibration oracle. No promised scalar-condition model is silently counted as performed.

## Endpoints, decisions, and honest calibration accounting

Primary fixed-budget endpoint: participant macro balanced accuracy averaged equally
over three speeds and k1/2. Report QM−Q, QM−Q2, QM−SHAM. k3/5 is a harm check.
Participant paired bootstrap: 2,000 resamples, seed 20260911; percentile 95% intervals.
These describe variability of fixed cross-fitted predictions, not all retraining uncertainty.

Before any outer-test outcomes, use selected-λ inner OOF predictions to choose each
arm's smallest k with mean balanced accuracy ≥80%. If none, fix k5 and record
source_target_unmet. Apply that fixed choice to outer-test participants and report
balanced accuracy, 80% run reach rate, full acquired prefix trials, and ready-time proxy.
Query-selected first-80% hitting costs, if reported, are secondary retrospective ORACLE
descriptions with nonreach/right-censoring; they are not stopping policies.

Retain for independent validation only if all hold:

1. QM−Q ≥2 percentage points at k1/2; QM−Q2 and QM−SHAM >0.
2. 95% lower bound for low-k QM−Q >−2 pp (exploratory harm guard, not superiority proof).
3. Source-chosen-k QM mean balanced accuracy ≥80%, its QM−Q lower bound >−2 pp,
   mean acquired-prefix cost at least 10% below Q, and 80% reach rate no lower than Q.
4. k3/5 mean QM−Q ≥−1 pp; at most three of 16 participants lose >5 pp at low k.
5. Actual consumed SHAM M changes and learned residual actuation are nonzero. If
   zero-calibration CCA is already as accurate as the QM chosen policy, do not claim
   this workload needs metadata-assisted calibration; retain at most a mechanistic signal.

If accuracy improves without policy savings, record that distinction and close the
calibration-reduction claim for this candidate. No favorable subgroup/offset/λ rescue.
If all retention rules pass, stop discovery and propose a separate independent validation
without opening held60. Otherwise report the negative or inconclusive result and why.

The marker is not verified physical flicker onset; phase invariance does not fix a wrong
window. Interpretation is limited to this frozen public-export replay. Fixed query trials
41–60 and their unused gap prohibit claims of earlier first online decisions or shorter
whole sessions. Unknown IMU mounting time prohibits claims of total setup-time savings.
Root is the only code/DB writer; agents review read-only in the shared tree.

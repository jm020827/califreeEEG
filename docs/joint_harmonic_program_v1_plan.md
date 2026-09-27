# Joint harmonic few-shot program v1 — prospective design

2026-09-23 KST. Parent: `de150428b01e4bce39fe8b390bbb99a3c0c79280`.
Status: SCIENCE_FROZEN / IMPLEMENTATION_IN_PROGRESS. The accompanying config and
this document are hashed in the program state before any numeric experiment.
Human execution additionally requires the generated full-flow checks below.

## Question and problem signature (before literature retrieval)

- Representation: channel-resolved complex coefficients at the common candidate
  stimulus frequencies and harmonics, plus labeled target support. Acquisition
  context is the already measured block/channel impedance, not a new measurement.
- Bottleneck: prior experiments tested frozen embeddings/precision residuals,
  native-filter regularization, or source-expert routing. A general conditioned
  representation was also tried under an earlier zero-calibration protocol.
  Whether an episodically learned support-template metric exploits the same M
  differently remains unknown; conditioning alone is not a new mechanism.
- Allowed operation: source-participant supervised joint training of a small
  spectral encoder and context conditioning; target personalization only through
  the paid support set. No held60, target query fitting, person contact or paid
  resources. No changes to old closed experiments or their stopping rules.
- Objective: distinguish B improvement from incremental M benefit and evaluate
  accuracy versus acquired calibration trials. Overall setup-time reduction is
  unknown without actual metadata/setup timing.
- Resource constraint: at most three mechanisms allowed by the goal; this branch
  will select a smaller explicit set if sufficient. Freeze total fits, tuning,
  reads, query evaluations, runtime, disk and engineering repairs before execution.
- Feedback: subject-separated development losses, final paired Q/Q2/QM/SHAM
  accuracy and observed calibration-cost comparisons. Reused source39 is development,
  not independent confirmation. M actuation alone is not efficacy.
- Failure modes: redundancy of M with EEG; person/interface shortcuts; losing
  phase in preprocessing; condition-independent affine cancellation; small-source
  overfitting; accidental query statistics or labels in preprocessing; capacity or
  tuning imbalance; apparent accuracy improvement without calibration savings.

## Bounded literature check

Reuse the existing research-space landscape first. Then inspect at most four
exact primary articles/pages (no broad discovery): CSDuDoFN, SSVEP-DAN, FiLM,
Prototypical Networks. Core question: what is jointly learned, where support
enters, and which acquisition metadata was actually tested? Foundation/adjacent
question: can conditional feature metrics avoid being merely a class-independent
score offset? Treat this as a transfer hypothesis, not an established SSVEP result.
Read HTML/abstract or already retained artifacts; no new PDF download in this pass.
The workspace's 2026-09-04 cutoff is historical, not a claim of current completeness.

## Existing results that must remain intact

- `task_trca_n1_transport_recovery_r1_results.md`: 39 development participants,
  no incremental impedance accuracy/calibration benefit for that fixed candidate.
- `block_scaled_router_human_v1_results.md`: 16 participants; operative gyro
  routing, low-k QM minus Q -0.162 pp and acquisition-prefix savings 0%.
- `post_no_go_reliability_design.md`: an earlier jointly trained bounded waveform
  spatial operator was proposed; its synthetic quality-probe gate failed and its
  downstream decoder was not evaluated. This is not a human test of every joint model.
- `metadata_calibration_prior.py`: frozen embeddings, source anchors and staged
  diagonal-precision residual. A new method must differ from this actual code,
  not from an inaccurate claim that no old method ever learned with M.
- `physical_conditioning.py`: FiLM/channel conditioning already exists; neither
  FiLM itself nor 'first use of metadata in learning' is a defensible novelty claim.

The previous status-only turn is NO_PROGRESS. This new program will not impose an
M-utility pretest or an indefinitely improving B as a prerequisite for the direct
matched comparison.

## Literature findings and the decision they change

Targeted HTML methods reading, 2026-09-23; not a new comprehensive search or PDF
deep read. No publication-status inference beyond the inspected arXiv records.

| Source and inspected scope | Author-reported mechanism | Consequence here |
| --- | --- | --- |
| [CSDuDoFN v1](https://arxiv.org/html/2311.07932v1), III-A–D | Source CNN plus target-support LST and SAME/eTRCA/TDCA fusion; target one-shot does not train the CNN | Few-shot itself is established. Retain explicit support personalization, but test a source-episodic conditional metric rather than claim to reproduce this system. |
| [SSVEP-DAN v1](https://arxiv.org/html/2311.12666v1), III-A–C | Shared cross-stimulus nonlinear spatial alignment to target templates; pretraining and source-wise fine-tuning | Learned spatial alignment is prior art. Our support-prototype metric is not new merely because it mixes channels. No acquisition-impedance efficacy is established by these inspected methods. |
| [FiLM v2](https://arxiv.org/abs/1709.07871), author abstract only | Conditioning by feature-wise affine transforms | A generic implementation vocabulary, not evidence that impedance helps EEG. Abstract-level evidence remains provisional. |
| [Prototypical Networks v2](https://arxiv.org/abs/1703.05175), author abstract only | Learn a metric for classification against class prototypes from few examples | Motivate a small learned metric. Their new-class setting differs from our fixed-class, new-person task. Our exact cosine implementation is an independent choice, not a paper reproduction. |

Saturation: these four sources establish the relevant component precedents. They
do not answer the present external-M question; another broad search is less useful
than a matched test. The novelty remains conditional and empirical, not a claim
of first FiLM, first few-shot, first spectral network or a new brain mechanism.

## Frozen candidate portfolio: ONE mechanism (not three sequential rescues)

**C1: support-conditioned joint harmonic prototype metric.** Learn the EEG encoder
and conditioning network together on source-participant few-shot episodes. Form
target prototypes from that person's first k complete labeled blocks; do not fit
target network weights. M is prefix impedance, which may resolve sensor-dependent
ambiguity in which complex spectral features should define the personal metric.

Differences from actual previous candidates:

- Unlike frozen-embedding precision priors, the spectral encoder is trainable and
  receives gradients through both the support prototypes and query classification.
- Unlike TRCA regularization, no eigensolver/regularized native filter or Q-first
  frozen residual is used. The learned object is a nonlinear support-template metric.
- Unlike source routing, there is no learned mixture over fixed source experts.
- Unlike `physical_hybrid_v1`/`reliability_spatial_v1`, this is not a zero-calibration
  prompt or query-waveform quality correction. Source training explicitly matches
  target k=1/3/5 support-prototype adaptation; context is shared prefix information,
  never current-query M. General feature conditioning is acknowledged as overlap.

The hypothesis fails within this program if M's operative learned representation
does not beat the matched EEG/common and EEG-extra controls and reduce calibration
cost under the fixed rules below. Failure is conditional on this complete method,
dataset and budget, not a universal information bound on impedance.

No C2/C3 reserve will be opened after seeing results. Covariance regularization
and source routing are excluded as already tested mechanisms. Neural ODE is not
included: this test needs no latent temporal dynamics or unmeasured timing proxy.

## Fixed data and information rights

- Exactly the 39 already exposed participant IDs in the JSON config; no held60.
  Both dry/wet, all 12 classes, 10 blocks, same 2-second crop (samples160:660,
  zero-based, 250Hz) in each trial. Raw shape must be `[8,710,2,10,12]`.
- Extract all-class complex coefficients with centered rectangular-window direct
  Fourier sums at the 12 common frequencies and 3 harmonics. Phase origin is crop
  time zero; no claim of measured neural/display latency. No extra filter or notch.
  Upstream public downsampling provenance is inherited, not declared causal.
- Center per trial/channel; divide spectral input by the trial's global channel
  RMS. This is per-trial, not cross-query normalization. Keep real and imaginary
  parts, input `[trial,8,72]`. Zero/invalid trial RMS fails validity, no silent removal.
- Support blocks0:k, k=1/3/5; fixed query blocks5:10. One block includes all12 targets.
  Q: mean over support trials of per-channel log RMS and log summed harmonic
  coefficient power relative to time-domain power. Q2: log difference-RMS/RMS and
  log normalized fourth moment, likewise support-only. No query labels enter context.
- M: prefix mean and population SD of log1p(impedance in kohm), channel-resolved;
  explicit two availability flags. Zero is valid, NaN is missing, negative/inf fail.
  A one-block SD is zero, not a fabricated extra measurement. No query-block M.
- Project the shared Parquet container to source39, blocks01–05 and exactly
  subject_id/electrode_type/run_id/impedance_kohm_by_channel/headband_order/
  condition_period. Validate repeated class rows agree. The container contains other
  people: hashing it is allowed, but no held rows are returned or numerically used.
- Common to all arms: complete frequency/phase list, channel order, dry/wet,
  headband order, condition period and log(k). Person ID is only an evaluator/split
  key, not a learned feature. No inferred Q feature may be relabeled external M.
- Raw files decode once per allowed subject. Feature extraction is label-blind;
  class-axis ordering remains known benchmark structure, not a cryptographic blind.
  Role-isolated fitting must never index outer-query participants or their statistics.

## Model and matched controls

Shared architecture, exact same parameter count, initialization seeds, episodes,
optimizer, checkpoints and tuning count across Q/Q2/QM/SHAM:

1. Shared channel encoder `Linear(72,32)` + GELU.
2. For each channel, context has Q2 numbers, auxiliary2 numbers + availability2,
   and four common fields (dry/wet, headband order, period, log k): 10 numbers.
   `Linear(10,32)` + GELU + `Linear(32,64)` produces scale/shift.
   `h' = GELU((1 + 0.5*tanh(scale))*h + 0.5*tanh(shift))`.
   Affine output initially zero; joint training can update all components.
3. Flatten 8×32, `Linear(256,64)`, L2-normalize each trial embedding. No BatchNorm,
   source anchors, task ID, query-batch statistics or target fine-tuning.
4. Average support embeddings per class then normalize each prototype; logits are
   `10 * cosine(query, prototype)`. CE trains the whole graph jointly.

Q's auxiliary values duplicate Q, with availability=1: the extra path is trainable
using existing information, not a smaller graph with disconnected parameters.
Q2 supplies the two extra EEG features; QM supplies measured M. SHAM uses the same
M representation with participant packets deranged inside each train/validation/
outer-evaluation role and headband-order stratum; preserve interface, channel and
block correspondence, and use the same donor for every k. No donor crosses a role
boundary; any singleton stratum is a validity failure, not an ad hoc permutation.
Every feature normalizer is fitted on the fit participants' support prefixes only.
Missing values become zero after normalization with flags retained; no cohort
imputation or normalization over held participants. QM missing-input sensitivity
is not claimed identical to the separately trained Q arm.

Fixed non-neural comparisons (no extra tuning):

- B0: cosine against support-mean raw complex coefficients, no learned encoder.
- Direct-M B0: the same prototype scorer with channel gain proportional to
  inverse sqrt(1+prefix mean impedance), divided by its geometric mean and clipped
  to [0.5,2]; missing channel gain1. This is a simple prior, not established physics.
- k0 common regularized CCA: current trial vs all 12 sine/cosine references,
  three harmonics, trace-scaled ridge1e-6. This is a fixed CCA baseline, not FBCCA
  or a reproduction of the strongest published SSVEP system.

B versus B0/CCA is a separate baseline result. Only QM versus matched learned
Q/Q2/SHAM supports an incremental-M claim. No independent source-data advantage.

## Fixed split, training and tuning

Sorted source IDs at positions `i%3==fold` form 3 outer folds of13. The remaining26
are sorted and split alternately into inner fit13 and validation13. All trials,
interfaces and k of a participant share a role. Outer folds reuse source people:
this is repeated-development cross-fitting, never independent confirmation.

- Inner: 3 folds ×4 arms ×2 learning rates(0.001,0.0003) ×1 seed =24 fits.
- Choose each arm's LR by participant-averaged validation accuracy averaged over
  k1/3/5 and both interfaces. Ties choose0.0003; exactly one final checkpoint per
  fit, no best-epoch selection. Same tuning opportunity, not forcing QM to Q's LR.
- Outer:3 folds ×4 arms ×2 seeds(20260923,20260924)=24 fits on26 people.
- Each fit:1000 AdamW steps, weight_decay0.001, gradient clip5, no scheduler.
  Four episodes per step; k cycles1/3/5, uniform source participant/interface,
  one uniformly chosen late query block per episode. Arm/LR-independent schedules.
  Train CE only, no additional loss-weight tuning. Float32 CUDA when available;
  CPU reference tests required. No human-device fallback rerun to seek efficacy.
- Inner validation is used for choice, not final evidence. Freeze all24 final
  models and policy choices before any outer-query scoring. Do not inspect outer
  metrics fold-by-fold while changing later models. Save all probabilities/scores
  and all arms, not only a winner. Mean the two seeds for primary summaries and
  retain separate-seed performance/direction; do not select the better seed.

## Endpoints, costs and stopping rules

Primary accuracy: participant-equal, interface-equal mean BA at k1/3. Also report
every k/interface, each seed, participant changes and lower-tail harm. Bootstrap
2000 participant resamples(seed20260925) for paired descriptive intervals; these
are not confirmatory p-values after repeated development and multiple prior designs.

Actual selected calibration acquisitions are complete blocks:0,12,36,60 trials,
not just decoder-window samples or chosen best trials. There are two distinct costs:

1. Source-chosen deployment policy: for each outer fold/arm use its selected-LR
   inner validation curve to choose the first k0/1/3/5 with mean BA>=80%; if none,
   use k5 and explicitly mark fallback. k0 uses identical common CCA for all arms.
   Apply the locked k on outer people; report achieved BA, target attainment and
   acquired trial count. A cheaper policy that misses accuracy is not success.
2. Descriptive observed-threshold curve: for each person/interface, first grid k
   attaining80% on the fixed60 queries. Report jointly attained paired savings,
   new/lost/unattained counts separately; do not assign zero to unattained people
   or call this retrospectively selected point an online stopping policy.

Report used EEG seconds(2×acquired trials) only as processed-window duration,
not wall-clock collection time. Full cue/rest/block duration and incremental
impedance/setup time are UNKNOWN here; no overall ready-time saving claim.

Candidate is promising for independent validation only if ALL hold:

- Mean low-k QM−Q>=2pp, QM−Q2>0 and QM−SHAM>=1pp; QM−Q positive in each seed.
- Source-selected policy QM mean BA>=80%, QM−Q policy BA>=−1pp, and mean acquired
  trials at least10% below Q (if Q cost0, the savings condition fails).
- No more than20% of participants have low-k QM−Q worse than−5pp.
- Leakage, role, finite-output, capacity and saved-prediction recomputation checks pass.

These are prospective development screens, not proof of a population effect or
publication thresholds. If only B improves, record B-only evidence. If M improves
accuracy but not cost, record partial evidence without the calibration claim.
Otherwise close C1 with all failures/negative outcomes intact. No C2 or retuning
after a negative result. A promising candidate stops search and gets an independent
validation plan; held60 remains unopened until separately approved.

## Entire-program resource ceiling

| Resource | Absolute program limit |
| --- | ---: |
| Mechanisms / configurations |1 /2 LRs per arm, fixed architecture |
| Human extraction attempts |1; 39 allowlisted MAT loads +39 byte hashes |
| Shared manifest |1 hash +1 filtered/projected read,4680 returned rows |
| Extracted cache loads |2 (runner and saved-artifact audit), no raw re-extraction |
| Human fits / updates / episodes |48 /48,000 /192,000 |
| Development validation outputs |24 inner checkpoint evaluations on fixed role/k grids |
| Outer development reveal |1 batch after24 models frozen; no adaptive query reuse |
| Human runtime |7 wall hours total; training at most6 GPU hours |
| Peak RAM / GPU allocation / new output |16GiB /8GiB /4GiB |
| Generated test invocations / total runtime |10 /1800s |
| Generated optimizer updates |3000 total; no generated efficacy gate |
| Implementation repair rounds |3 within the above limits, before human run only |
| Saved-result numerical audit |1 batch, CPU<=600s, no retraining |
| New data / people contact / paid / held60 |0 /0 /0 /0 |

Engineering tests must cover exact harmonic phase/frequency representation,
support-label permutation equivariance, query-batch independence, support-only
context, Q no-M access, missing/invalid impedance, SHAM roles/nuisances, source-only
normalization, source/query role isolation, equal parameter counts, gradients to
encoder and conditioner, serialization, CPU/CUDA agreement, and a generated complete
inner-choice/final-freeze/evaluation/cost loop. Tests need not demonstrate real-M
efficacy. Implementation errors are repaired without changing these scientific
choices; all test attempts and repairs are logged. A failed human attempt stops
this program with validity/efficacy status distinguished, not an automatic retry.

Root is the sole writer. No worktree or agent implementation lanes are required.
Existing untracked test directories, old reports and closed runner authorities are
preserved. Program state records consumed budgets and the next concrete step.

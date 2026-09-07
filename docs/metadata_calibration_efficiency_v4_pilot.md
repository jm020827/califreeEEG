# V4 waveform/context pilot 001 — pre-outcome plan

실행 후 상태: 아래 과학적 설정·판정 기준과 JSON plan은 유지하고 상태 주석만 덧붙인다. Pilot 001은
`AQ_NOT_ESTABLISHED`로 완료됐으며 [별도 결과 보고서](metadata_calibration_efficiency_v4_pilot_results.md)에
ceiling, calibration endpoint와 fusion 진단을 기록했다. 아래 pre-outcome 기록은 역사적 계획이다.

## Version clarification and objective

`development-v5` is **V3 method, development execution #5**, not a V5 scientific
method. It completed with `DEVELOPMENT_NO_GO`, zero eligible cells out of nine.
Its immutable evidence and exact source boundary remain as recorded in
[the V3 results](metadata_calibration_efficiency_v3_results.md). Do not reopen
its governed runner from the changed documentation tree or alter old artifacts.

V4 is a new, exploratory synthetic engineering candidate. The question remains:
**does external pre-query acquisition context M reduce labeled calibration
burden beyond an EEG-only Q comparator?** Neither changing confidence alone nor
recovering damage caused by an inadequate comparator establishes that claim.

Prior template-transfer literature motivates retaining waveform information;
it does not establish that metadata is helpful. The existing targeted-fulltext
LST card and official abstracts of [LST](https://arxiv.org/abs/2102.05194),
[CSDuDoFN](https://arxiv.org/abs/2311.07932), and
[SSVEP-DAN](https://arxiv.org/abs/2311.12666) support testing calibration-aware
waveform representations. This simple pilot is not a reproduction of any of
those complete algorithms. The 2026-09-07 narrow search added no decision-changing
mechanism; Semantic Scholar returned 429 after retries. This is not an exhaustive
new literature review. The earlier workspace contains the wider prior-art map.

## Fixed experiment, in plain language

Twenty-four synthetic participants each see twelve targets. A block means one
labeled EEG trial per target: k=1 therefore costs **12 labeled trials**, not one
trial total. All methods receive the same first k=0,1,3,5 complete support blocks
and exactly the same five disjoint future query blocks. All eight channels are
present. One-second trials use 250 Hz sampling.

Three conditions are fixed: stable acquisition; changing channel gain/noise with
a noisy context measurement; and the **identical changing EEG** with unrelated
context measurements. The helpful condition deliberately gives EEG and M a
common cause. It is an engineered mechanism check, not a validated physiological
or impedance model. Metadata cannot contain target labels, participant identity,
decoder accuracy, or query-derived measurements. Query Q may use only the
current query EEG, never its label or the other query trials.

The exact numbers, RNG namespace, estimators, controls and screen are in
[the machine-readable plan](../configs/analysis/metadata_calibration_v4_pilot.json).
Participant/class harmonic phase and modest spatial patterns are stable within
a participant. Independent AR(1) noise and trial phase jitter prevent literal
support/query waveform reuse. State affects every target through channel gains
and noise, not a class-specific confuser shortcut. Random stream components are
keyed independently; stable/drift share base waveforms, trial innovations and
orders; drift/null share EEG byte-for-byte.

## Compared methods

- A0: existing public FBCCA, no labeled target support.
- AQ: compare query waveform to each labeled support block; combine it with A0.
  Its weights already use labeled support quality and **EEG-only query/support
  context similarity**. This is a stronger comparator than support quality alone.
- AQM-block: metadata decides how much to trust each block separately.
- AQM-scalar: same overall metadata trust, but no block-specific reallocation.
- Missing M: exactly AQ; shuffled M: all block derangements at k=3 and k=5.
- Pooled-template diagnostic: average support waveforms per class before scoring.
  This tests whether waveform averaging learns even if per-block posterior
  mixing fails. It cannot be promoted after seeing this pilot's results.

All methods use the same seven-band filtering. A0 uses squared CCA correlation;
template methods use phase-preserving signed squared correlation. Their scores
are divided by the same sum of
filter weights before the fixed softmax temperature .1. The fusion strength is
.5. Q is log residual channel variance after a fixed all-class harmonic projection;
the projection uses SVD with relative rank cutoff 1e-10. Support reliability is
projected labeled-harmonic power / total centered power, i.e. SNR/(1+SNR).

For block posterior `pb`, EEG-only block weight `pi_b`, and metadata affinity
`a_b`, define `g=sum(pi_b*a_b)`:

```text
AQ         = p0 + .5 * sum_b pi_b       * (pb-p0)
AQM-block  = p0 + .5 * sum_b pi_b * a_b * (pb-p0)
AQM-scalar = p0 + .5 * g * sum_b pi_b   * (pb-p0)
```

At k=0 every fusion returns exact A0; at k=1 block and scalar coincide. Missing
M gives affinity one and exact AQ. Uniform affinities make block and scalar
equal. No outcome-dependent safety gate is fitted in this bounded pilot; report
harm explicitly instead of calling the candidate deployment-qualified.

## Interpretation fixed before outcomes

Primary measurement is participant-level balanced accuracy (BA), including the
normalized trapezoidal low-budget area `BA0/6 + BA1/2 + BA3/3`. Use paired
participant differences with one-sided 95% t lower bounds and two-sided 95% t
intervals, explicitly exploratory and not multiplicity-adjusted confirmation.
Correct-class log probability, true-class margin, actual prediction flips,
metadata weight turnover and harm are diagnostics. Equal BA by itself does not
prove that every prediction stayed unchanged.

First AQ must improve stable-condition low-budget area by at least .01, with
positive lower bound and nonnegative k1 mean gain. Otherwise status is
`AQ_NOT_ESTABLISHED`, regardless of any attractive metadata result.

If AQ passes, metadata must improve drift low-budget area by at least .01 with
positive lower bound; beat deranged metadata at k3 by at least .01 with positive
lower bound; beat the equal-total-trust scalar at k3 with positive lower bound;
have the null k3 90% difference interval inside +/-1/60; and have nonnegative
drift k3 mean gain over A0. Failure is `METADATA_NOT_ESTABLISHED`. Passing only
earns `READY_FOR_INDEPENDENT_SYNTHETIC_REPLICATION`, never human-study approval.

For descriptive calibration attainment, report the smallest observed k reaching
80% BA; if unattained report `>5`, never numeric imputation. Report k3-minus-k1
curves and k1 AQM versus k3 AQ as diagnostics, not proof of noninferiority or
actual human calibration savings. All families and controls are reported.

## Execution and coordination

This plan is committed before study RNG draws. Only unit-test fixture RNG may
run during implementation. Freeze source commit, plan SHA, filterbank SHA and
Python/NumPy/SciPy/runtime identity in an exclusively created start record before
study draws. Publish a result exclusively; an existing start consumes the attempt
and is not automatically resumed. This is lightweight engineering provenance,
not V3's scientific-lockbox authority or an external preregistration service.

Base: main `df62c5b2533b7ce2eff62b9f6ac1b5c3948fc4e2`. One new checkout costs
about 7 MB, no new dependencies or datasets; reserve under 1 GB including pilot
artifacts against over 300 GB free. Shared interpreter is invoked read-only by
absolute path; worktree pytest temporary/cache paths remain local.

| Lane | Owner and paths | Runtime | Order |
| --- | --- | --- | --- |
| Contract/integration/report | main; this document, new plan, append-only research log, `scripts/audit_metadata_calibration_v4_pilot.py`, `tests/test_metadata_calibration_v4_pilot_audit.py` | main and sole research-workspace writer | Freeze first; integrate; run once; report |
| Implementation | v4_baseline_mapper; new V4 module, runner, tests only in `/home/whwovy/califreeEEG-wt-v4-pilot`, branch `codex/v4-pilot001` | unit fixtures only; no full study seeds or old artifacts | Starts after contract commit; returns one reviewed commit |
| Independent review | v4_design_skeptic; read-only | no writes or experiments | Review frozen contract and implementation before study |

No agent writes concurrently in the same working tree. Existing V3 source,
governance, artifacts, dependencies and human datasets are forbidden targets.
No worktrees or branches are deleted. Main checks API leakage, exact controls,
paired data identity, metric aggregation, tests and source cleanliness before the
single four-worker CPU run. Any numerical caching must be checked against the
existing public FBCCA on independent unit fixtures. No CUDA/library migration is
part of this candidate.

### Pre-outcome implementation clarifications

Channelwise gains are `exp(.35*z*u)` for the signal and `exp(.5*z*u)` for noise;
`log M=log(50)+.5*z*u+measurement_noise`. Stable z=0 retains the same underlying
waveform/noise/order draws. True-class margin means `p_true-max(p_other)`.
Weight turnover means half the L1 distance between `pi` and `pi*a/g`, set to zero
if g=0 and always reported alongside g. Trial-level diagnostics are averaged
within each participant before uncertainty calculation. The claim is M beyond
**this EEG-derived Q comparator**, not beyond all information recoverable from EEG.

### Pre-outcome verification record

Plan commit `b56339de50007d743650cecc38636f3757561ea2`; plan SHA-256
`50607e9b8abe7058565b45cdb8a98dcd1a78f516c393af911f0e5f3ab0484124`.
Implementation worktree commit `052d867a359063de70e142796228d9a838ebe782`
integrated as `b54b10c`; independent audit added in `8eff1a1`.
There were no overlapping writing paths or textual/semantic integration conflicts.
The main-owned incidental test-only change in `3338137` makes an existing
ciphertext tamper fixture always change a character; production crypto and all
old scientific/governance source remain unchanged. That commit also aligns the
new independent audit's corruption fixture with the final result schema.

Integrated verification: **44 focused tests passed**, and **915 full-suite tests
passed** in 109.18 s (68 existing PyTorch nested-tensor warnings). The first full
run had 914 passes and the pre-existing tamper no-op failure, now corrected.
Changed Python lint/format and `git diff --check` passed. The independent reviewer
also compared cached and public FBCCA with all twelve frequencies/seven bands,
including rank deficiency and zero EEG: maximum score difference 0.0.

New tests use explicitly separate fixture seeds only. Study start/result did not
exist at this verification point. The executable runner will record the final
clean source commit before the first study draw; no exact V3 artifact is reopened
through its governed runner. The old V5 result file SHA remains
`1af9f65910da1d753a30a957dfb5bbef7366b696f9f32d444bf95dd411afcc8c`.

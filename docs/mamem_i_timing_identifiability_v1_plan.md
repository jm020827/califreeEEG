# MAMEM I metadata/measurement identifiability gate v1

2026-09-13 KST, base9cafdce/main. User says continue. This activates the preceding
[draft](mamem_i_timing_identifiability_v1_draft.md) as the bounded scope below, not
an extension of yesterday's expired raw-download/probe manifest.

Signature: representation=optical DIN timestamp/sample events paired with EEG;
bottleneck=ID/channel mapping and separating physical stimulus timing from frame
schedule/measurement error; allowed=public source documents and generated arrays;
objective=decide what M can legitimately represent before low-calibration learning;
feedback=source locators and fixed measurement-model invariants, not accuracy;
resource=4unique official source targets,3generated scenarios,fit0;
failure=aliasing people, guessing channel257, interpreting observed spread as jitter,
or rewarding metadata for deterministic correction that controls lack.

## Ownership, input and time bounds

- Root sole repo/research-space writer and final integrator. Author-code scout and
  vendor-clock scout read-only; separate skeptic read-only. No new worktrees:
  shared contracts are small and disk reserve is tight. Existing41worktrees and
 8untracked pytest directories untouched, no install/delete/push.
- Initial free8,479,136KiB; task new artifacts<=16MiB and keep8GiB free. No raw
  archive extraction/rehash or MAT reopening, including yesterday's first prefix.
  Existing saved schema/probe/inventory JSON may be read, not new numeric values.
- Public sources: root exact release DataAcquisitionDetails.pdf(file3687771,
 276431bytes,MD5a0ed5f0d7130f4fa662ee3657ff31036), author scout<=2exact official
  code/doc targets, vendor scout<=1exact official HTML target plus1locator search.
  Root may retain<=2successful scout targets (duplicates, not new targets).
  Total unique4, attempts1/target/lane, ordinary body<=2MiB/30seconds. No bypass,
  retries of denied endpoints, auth/DUA/outreach/paid resources. Source budget
  ends2026-09-13T03:08:00Z; whole round ends03:20:00Z. Stop if sufficient sooner.
- Root queues the release document as a decision-relevant deep-read before GET.
  PDF skill is unavailable; use installed pdftotext+pdftoppm as explicit fallback,
  retain original PDF/hash and render relevant pages before paper-card checkpoint.
  No new scholarly discovery or unrelated PDF. Questions: does this exact release
  define row257/reference/channel order, participant IDs, and DIN units/clock/
  polarity/photodiode provenance? Missing details remain unknown.

## Fixed generated measurement test (not EEG efficacy)

Exactly3constructed scenarios, seed20260913 (deterministic shared sequence),
120events, nominal60Hz display frames,250Hz EEG sample grid and illustrative1ms
timestamp quantization. These rates/quantizer details are simulation assumptions,
not measured facts about the archive export. No frequency inference from raw DIN.

Model: optical time t_i; common detector/delivery delay d_i before two event fields;
observed z_i=round_ms(t_i+d_i), s_i=round_sample(t_i+d_i). Joint observed times can
therefore be identical under optical changes versus changes in this shared path.
An error only in z after sample assignment is a different, potentially detectable
model and must not be conflated with a shared pre-assignment error.

1. Normal unequal integer-frame schedule+quantization: t follows a fixed cycle of
   frame counts[4,4,5,4,3], d=0. Spread may exist with no physical departure from
   intended schedule. Expected: positive interval spread, |delta-z−4delta-s|<=5ms
   bound (1ms+4ms difference quantization), no independent jitter claim.
2. Physical optical timing departure: add fixed bounded sequence j_i to t, d=0.
   Generate a toy constant-phase harmonic response on optical-event coordinates.
   Exact common timing-coordinate correction restores the canonical representation
   for every arm, up to numerical tolerance1e-12. This is an idealized algebraic
   correction test, not an EEG reconstruction/learner/accuracy result.
3. Shared marker-path timing error: keep optical time/response as scenario1, use
   d=j_i. Expected z/s exactly identical to scenario2 despite different true optical
   times; applying scenario2's optical correction here creates representation error.

Freeze j_i as repeating[0,2,-1,3,-2]ms; no parameter tuning to the observed first
prefix's71intervals. Report exact observation equality, true-optical nonidentity,
quantization bound and common-correction invariants. Scenarios2/3 ambiguity is
conditional on the shared-path error model being allowed; it does not prove that
this error occurred in MAMEM. If authoritative provenance rules it out, revise the
measurement premise rather than treating the constructed counterexample as fact.

## Decision rules and follow-on

If metadata provenance cannot distinguish physical and shared-path effects, do not
promote interval spread to verified optical jitter or a learned-M feature. Preserve
the dataset as acquired; park this specific interpretation, not all possible uses
of timing M. If common correction fully removes the simulated issue, attribute its
benefit to common correction, not extra learned metadata. No outcome-driven search.
ID/channel unresolved means no automatic row deletion/subject alias/human fitting.

After sources and generated tests, write results/state/research log, claim-evidence,
and next executable question. Q/Q2/QM/SHAM, source-only selection, query-DIN ban,
whole acquired-prefix/setup costs, S001 all-dev, old negatives and held60 protection
remain unchanged. New EEG fits/query access/held60/outreach/paid resources all0.

## Prerun algebra clarification after independent review

Use fixed common baseline delay D0=4ms in all3worlds. j_i denotes deviation:
world2=(t=u+j,d=D0), world3=(t=u,d=D0+j). All absolute marker-path delays are
nonnegative; this changes no constructed observation-equality test or fit budget.
Exact1e-12 correction uses oracle unquantized optical coordinates, not estimated
quantized DIN. Report identifiability only for DIN(z,s), not EEG+DIN jointly.
The generated5ms quantization bound does not relax yesterday's4.001ms real-probe
criterion. These are preregistered algebra clarifications before generated run.

Execution deviation: persistent PDF deep-read insertion failed twice (missing
paper entry, then uncommitted backend upsert). Download started while the second
queue failure was in the same tool batch. Questions existed in this committed plan
before GET; queue was fixed before any PDF text/pages were read. Do not claim the
strict persisted-queue-before-download requirement passed. Preserve this limitation.

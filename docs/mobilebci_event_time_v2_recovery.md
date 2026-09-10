# MobileBCI event/t v2: bounded parser repair and conditional recovery

2026-09-11. Base579172e. Previous turn was PROGRESS: public pair acquisition recorded,
release mismatch identified, and one real event reader compatibility failure preserved.
No v1 rerun or reinterpretation of its failure is permitted.

## Reason and expected difference, before execution

V1 accepts dictionaries and primitive NumPy arrays but rejects object-array cells;
it does not report the rejected path/dtype. The exact real failed dtype is unknown.
V2 will safely traverse common scipy MATLAB cell/struct containers, retaining shape,
and allow only numeric/string leaves. Unknown arbitrary Python objects remain rejected.
This changes representation handling, not participant selection, timing thresholds,
SSVEP scoring or the metadata hypothesis. It may recover timing metadata; it cannot
produce performance evidence by itself.

## Finite stages and resources

1. Generated-only implementation: at most3 test-suite invocations, each fixture at
   most8192 numeric elements and64KiB; at most16MiB new artifacts excluding existing
   database/pair. Test nested cells/struct lists, empty/scalar/ragged containers,
   bytes/unicode, arbitrary/complex/nonfinite leaves, path/depth/count/output limits,
   checksum failure, one-pass selective decode, and first-error stopping. No human
   files in tests. Stop repair at that budget if still invalid.
2. Conditional real recovery: only after generated PASS and static review, freeze
   exact input pins and reader revision in a separate run manifest, then one run
   on the same two existing public files; event/t only,1decode+1checksum pass/file,
   30s/file,768MiB process address space,8192 visited scalar/container units and
  64KiB report. These element guards are postdecode. Stop first error, no rerun.
3. If complete, analyze ONLY the saved small metadata report for field meanings,
   event label/count/order, cross-device time differences and candidate index units.
   No waveform/other participants, fits/outcomes, network downloads, outreach,
   paid resources, held60 or source39 access. Do not invent certainty on index origin
   or clock drift. A candidate conversion is not source-verified acquisition timing.

This is one finite compatibility recovery, not a new scientific efficacy candidate
or an extension of v1's exhausted test budget. Report success/failure and remaining
decision at the end; do not continue automatic human retries.

## Ownership and subsequent research

Root owns code, configs, runtime outputs and research DB in main. Read-only reviewer
shares the repository, no concurrent writer; existing41worktrees/8untracked dirs
remain intact. Available66,567,400KiB, no new tree/installation needed.

If timing can be defensibly anchored, next freeze raw support-only extraction and
exact Q/Q2/QM/SHAM one-shot reference-ridge evaluation. Keep shared speed/order,
participant-disjoint learning and actual chronological calibration costs. If timing
is still genuinely unknown, identify a concrete public definition needed instead
of labeling parser success scientific eligibility. Preserve Choi PARKED and all
previous failures/negative results.

# Public paired MobileBCI cohort acquisition v1

2026-09-11. Same scientific candidate, no new efficacy selection. V2 metadata recovery
completed and independent review supports engineering event-relative support extraction
with explicit time assumptions. One exact API inventoryGET then verified the file cohort.

## Fixed selection before downloads

Figshare13604078/v1,16complete participants s01,s03–s17, speeds0.0/0.8/1.6,
scalp+IMU:48runs/96files2,204,445,401bytes. Each subject is one independent unit.
S02 lacks IMU0.8; exclude it by availability, not performance. No optional2.0m/s,
ear/ERP, other releases or author requests. Description18 versus17SSVEPfilenameIDs
is preserved. Existing s01/0.0 pair43,427,682bytes is reused after checksum verification,
leaving94newfiles2,161,017,719bytes. Inventory is availability, not scientific eligibility.

## Budget and stopping

- One acquisition attempt per new file,94initial GETs maximum, sequential reuse of
  the existing tested streaming downloader;3normalredirects/file maximum, no auth.
- Whole cohort retained-byte cap2.5GiB, including reused files. New files stay in one
  newly allocated task directory; no deletion of failed partial files or old worktrees.
- Socket15s,filewall90s,wholebatch30min. Do not begin or continue past batch deadline.
- Size/MD5/SHA verification and top-level MAT schema only; no waveform/event numeric
  decode in this acquisition. Report each durable file result and stop at first failure.
- Generated wrapper tests at most2calls, synthetic content64KiB maximum. No fullrepo
  suite or repeated old tests required for the unchanged transport routine.
- Stop incomplete with exact records; no silent retry/substitution/cohort expansion.
  No source39/held60/outcomes/learner fitting/outreach/paid/GPU access.

## Next scientific use, not yet an efficacy contract

Validate paired metadata across the48runs, implement marker-relative raw support-only
features (no future-run fitting/no queryIMU), and fix Q/Q2/QM/SHAM reference-ridge
scoring, participant splits, numerical thresholds and calibration costs before outcomes.
Keep all prior failures/negative results. More downloaded bytes cannot establish that M
adds information beyond Q/common speed/order or reduces calibration requirements.

Root owns download runtime/files/research DB and main integration; reviewers remain
read-only. No new worktree needed. Disk availability~66GB exceeds2.5GiB cap; no cleanup.

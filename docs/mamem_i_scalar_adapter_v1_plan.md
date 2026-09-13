# MAMEM I channel/event schema and scalar adapter gate v1

2026-09-13 KST; base5e4f4f5/main. User says continue. This activates the next
bounded adapter question, not an expired prior raw/EEG experiment manifest.

Signature: representation=EEG rows and external event descriptors/samplingRate;
bottleneck=row257/reference and event-field semantics; allowed=official docs/code
and at most one already-dev S001a scalar-metadata pass; objective=remove concrete
input ambiguity before testing recorded-event M beyond Q for low-calibration SSVEP;
feedback=source definitions and schema/scalar values, not accuracy;
failure=guessing a row's meaning, silently aliasing IDs, treating a nominal/example
configuration as actual, or reading extra arrays to justify a desired M story.

## Scope and budget

- Root only repo/data/SQLite writer; read-only channel scout(2targets+1locator),
  original-ST100/EGI-setup scout(1target,no locator), and protocol reviewer.
  No concurrent writing lanes; new worktrees would add cost without independence.
  Preserve41worktrees/8untracked directories. No install/delete/push.
- Initialfree7,575,676KiB is below the old8GiB bulk-acquisition reserve due to
  shared filesystem changes. Bulk download/extraction remains disallowed. For this
  new small-doc/code round only, <=8MiB task artifacts and >=7GiB free; recheck
  before retention. No cleanup or attribution of other disk use to this task.
- <=3unique official source targets, each<=2MiB/30seconds/oneattempt. Root may
  retain<=2successful scout targets, not additional unique sources. Official
  implementation repos (MNE/EEGLAB/MAMEM) may clarify format conventions but do
  not automatically establish this exact file's reference/channel identity.
  Source deadline2026-09-13T10:49:00Z; fullround11:00:00Z. No failedendpoint retry,
  proxy/bypass, accounts, outreach or paid services. If a PDF is necessary root
  queues a decision-relevant read and checks successful persistence before GET;
  PDF skill unavailable means explicit local text/render fallback. No bulkmanual
  whose required artifact exceeds the bound. Unknown is an acceptable result.
- No new participant/EEG waveform, query, fit, accuracy, held60 or previous
  numerical-outcome reopening. Existing report/header JSON allowed.

## Smallest actual-file check (execute only after generated tests/review)

Pinned existing `/home/whwovy/data/mamem_i_v1_20260913/development_first.mat`,
SHA25657a72c3fde0ff3bc9aaae10721cd7cb450bda63eb5299a96f41704ea696ad10a,
137357437bytes, original memberS001a; entireS001remains development-only.

1. Exactly one successful-or-failed scalar pass loads `samplingRate` only via
   scipy.loadmat(variable_names). Header must identify a1x1 numeric scalar; scalar
   finite/positive, expected250Hz from release declaration. If different, report
   mismatch without resampling or selecting another record.
2. Only if an official source explicitly defines the first event's descriptor
   fields, root may freeze an additional descriptor manifest BEFORE reading their
   values. Maximum two documented cells in the first DIN event, char<=64characters
   or single bounded numeric descriptor, no timestamp/sample reread/other events.
   If schema definition not established, skip this optional branch; do not infer
   field semantics by opening cells. Whole DIN decode must be declared if used.
3. No EEG array requested. Hash/header may process file/compressed bytes; this is
   not a claim of zero decompression. Use2GiBAS/30CPU-seconds/60wall-seconds,
   singleBLASthread, parent output<=64KiB, durable STARTED+terminal receipt,
   fresh output files and no retry. Unsupported structure/memory/timeout stops.

Freeze reader+manifest before real scalar access; tests use generated MAT/loader
spies for whitelist, hash/role/header/size and scalar constraints and durable
failure receipts. Parent compares its terminal/child report, not just exit0.

## Adapter decisions

Use known v1 filename namespaceS001-S011, not code'sS013alias. Do not drop or treat
row257 as EEG merely because a generic NetStation file often includes reference.
No automatic montage mapping, waveform opening, elapsed-order predictor or
offset correction. Confirmed schema may define a later adapter; it is not proof
of incremental M utility or calibration reduction.

After the bounded sources/scalar check, update evidence/results/log and the next
specific unresolved dependency. Preserve physical-jitter interpretation parked,
recorded-event predictive value open, Q/Q2/QM/SHAM common information/correction,
actual acquired-prefix/setup cost, old negatives and held60 approval boundary.

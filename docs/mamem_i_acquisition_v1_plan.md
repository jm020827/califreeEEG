# MAMEM I public acquisition and DIN-only schema probe v1

2026-09-13 KST / UTC start2026-09-12T17:26. User explicitly permits downloading
all files if needed. Base16df18e/main. Goal remains low-calibration SSVEP with
independent acquisition metadata, not raw-data accumulation or a new outcome search.

Signature: EEG plus optically recorded DIN events; bottleneck=exact public release
and actual DIN schema/independent timing variation; allowed=public catalogue,
necessary dataset download, hashes/header metadata and one prespecified DIN-only
development prefix; objective=obtain reusable files and determine whether a timing
candidate is implementable; feedback=release provenance/checksums/shapes/clock
quantization, not accuracy; failure=duplicate archives, target leakage, invented
photometry, unbounded disk use or treating download as learning success.

## Budget and ownership

- Root is sole repo/data/SQLite writer. Two read-only agents inspect release
  provenance and leakage/timing mechanisms. No agent writes in shared worktree.
  Preserve41worktrees/8untracked pytest directories; no new worktree/install/
  cleanup/push. Initial free16,433,608KiB (filesystem100% displayed).
- Up to6public catalogue/document/header targets across lanes (root3, source
  scout3), scout1locator if necessary. Root may retain1successful scout source.
  Normal docs2MiB/30s; browser bytes unmeasured. No PDFs in this round.
- Root seed is the official Figshare public article API for2068677, not the
  previously403 DOI resolver. This reads an openly offered catalogue, not a
  proxy, account, private API or retry of a denied file. Stop on actual denial,
  challenge, login/DUA requirement; do not bypass or request from authors.
- Only release Dataset I. After catalogue verification, freeze exact names/IDs/
  sizes/checksums/URLs before any download. Choose standalone MATs (not duplicate
  RAR/ZIP or PDF) if entire set <=4GiB; otherwise retain at most the lexicographically
  first standalone MAT for schema and report the full-set storage requirement.
  Each file <=512MiB. Maintain at least8GiB free; if concurrent disk use violates
  reserve, stop safely. At most2concurrent downloads,1attempt/file,no automatic
  retries. Preserve incomplete artifacts with clear .part status, no destructive
  cleanup. Maxdownload45minutes from frozen manifest; maxoverall60minutes.
- Download means bytes+hashes only, not EEG interpretation. All downloaded files
  initially quarantine outside Git under an explicit new dataset directory. Do
  not overwrite user files or recursively traverse unrelated data.

## Ordered gates

1. Public article title/version/license and file inventory must match MAMEM I.
   Confirm paper↔release linkage from a public reference or catalogue description;
   do not mix I/II/III or the separate multimodal PhaseI corpus.
2. Download and size/MD5/SHA256 verification. Hash equality is not biological validity.
3. Only lexicographically first standalone MAT: root variable names/shapes/types
   (whosmat or equivalent); no EEG values printed/interpreted. If MAT structure is
   unsupported or exceeds memory bound, record and stop the probe.
4. If documented DIN_1 exists, designate that first record and its subject as
   development-only (not future independent query evidence) before parsing. Load
   DIN_1 only, using existing local SciPy; max512MiB decoded DIN allocation. EEG
   waveform, labels and other records remain unopened for interpretation.
   Inspect at most first200DIN events and only the earliest continuous group,
   ending at a timestamp gap>2000ms as in the author loader. Require numeric,
   finite, ordered timestamp/sample fields; no ad-hoc alternative field inference.
   Report counts, interval spread and timestamp↔sample-grid consistency without
   dumping full event arrays. If a group boundary is not present within200events,
   treat it as truncated, not a complete trial. No class prediction or fitting.
5. Distinguish actual event variability from the4ms EEG grid and nominal video
   frame schedule. Observed spread is not automatically physical jitter or an
   independent M effect; learned residual beyond Q/common correction remains open.

No classifiers/fits/query labels/held60/external request/paid resources. No third
dataset, failed optimizer rescue or automatic held-out evaluation. Preserve prior
failures. If download/schema succeeds, the next step is a bounded source-only
timing mechanism test with common correction/Q/Q2/QM/SHAM and actual prefix costs,
not a claim of calibration savings. Missing provenance or schema remains unknown.

## Catalogue-based amendment A — before any raw download

At17:31UTC the official version1 catalogue succeeded: it has no standalone MATs,
only EEG-SSVEP-Part1.rar(3279547652bytes), Part2.rar(3305471846bytes), and a PDF.
The paper reference44 explicitly names article2068677.v1. The user's explicit
permission to download all necessary files covers these two non-duplicate parts.
This is a storage/input-format adjustment, not an outcome-based candidate change.

Override the standalone-only/4GiB/512MiB acquisition clauses above: download
exactly those two archives, total6585019498bytes, max7GiB cumulative and4GiB per
file, sequentially. Do not download the PDF. Reserve8GiB free plus all remaining
download bytes and512MiB for at most one extracted record; recheck during writes.
If reserve cannot be maintained stop without deleting user or partial files.
Raw acquisition remains45min from the frozen manifest. No full archive expansion.
List member metadata using already installed tools or a bounded header parser;
extract at most lexicographically first MAT only if existing support, non-solid/
safe structure and uncompressed size<=512MiB permit it. Otherwise retain verified
archives and report extraction as unexecuted. No installation or silent memory
bound relaxation. The DIN probe's subject/first-prefix/no-EEG-interpretation
rules remain unchanged. Use a memory-limited subprocess for any MAT probe.

## Probe implementation clarification — frozen before any MAT values

Author Session.m uses `cell2mat(dins(2,:))` and `cell2mat(dins(4,:))` in
`split`, with `SAMPLING_RATE=250`. Therefore expect a cell array, not a numeric
matrix inferred from a plot. Timestamp/sample cells must contain numeric scalars.
Use a2GiB process address-space cap,90CPU-seconds and120wall-seconds; the parent
records missing-report/timeout failures without replay. DIN allocation accounting
is conservatively capped at512MiB. SciPy decodes the entire DIN variable before
the prefix-only calculation; it does not materialize the EEG variable. Header
scanning can process compressed bytes, so this is not a no-byte-decompression claim.

Select first MAT from the combined verified archive listings; save the selection
and entire-subject development role before extraction/header/DIN interpretation.
Extract exactly that literal member to stdout using installed unar, not archive
paths on disk. Cap its single output to the declared member size and verify CRC32/
SHA256 before publication. No fallback to another record on failure. Generated ZIP
test validates stdout/single-member semantics; it does not prove RAR extraction.

The first boundary event contributes only its timestamp to detect gap>2000ms;
its sample and all later values are uninspected. No boundary within200events (or
end of record without a boundary) means censored, not complete. Each selected
sample must be one-based integer, increasing and <=the2-D EEG header's time length.
Report delta-t minus4*delta-sample; abs residual>4.001ms marks clock mapping
unresolved, never automatic unit conversion. A residual within that conservative
one-sample tolerance, even exactly0, does not validate physical jitter or an
independent metadata learning mechanism. No frequency/label inference is emitted.

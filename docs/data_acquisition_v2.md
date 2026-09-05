# V2 external-data acquisition and Choi2019 frozen anchor

Status: **frozen before decoding outcomes, 2026-09-06**
Scope: public/source-data acquisition only. This work did not open the wearable
held-60 outcome bundle.

## Decision

Choi et al. 2019 is now locally usable as a public, CC0, cross-day and
frequency-band **signal-only external anchor**. It adds 30 participants, two
days, three separately run four-target frequency bands, and 14,400 stimulus
trials. It does **not** contain wet/dry labels, randomized electrode order, or
impedance and therefore cannot independently validate the V2 acquisition-
metadata claim.

The direct metadata replication remains blocked on one of these two routes:

1. obtain the Liu et al. 2022 raw wet/dry, three-day EEG and block-level
   impedance with written reuse terms; or
2. obtain IRB/consent/data-governance approval and collect a prospective
   randomized, repeated-measures wet/dry SSVEP cohort with trial-linkable
   impedance.

Xing et al. 2018 is a useful second request target for randomized wet/dry data,
but its article only reports an impedance threshold/stability check, not the
block-level impedance trajectory needed for the strongest V2 test.

## 1. Authoritative Choi sources and immutable inventory

Authoritative records:

- [GigaDB dataset 100660](https://gigadb.org/dataset/100660), DOI
  [10.5524/100660](https://doi.org/10.5524/100660)
- [GigaDB dataset API](https://gigadb.org/gigadb/api/dataset/get_dataset/?dataset_id=100660)
- [GigaDB file-list API](https://gigadb.org/gigadb/api/dataset/list_dataset_files/?dataset_id=100660&per_page=1000&start=1)
- [DataCite DOI metadata](https://api.datacite.org/dois/10.5524/100660)
- [Choi et al. data paper](https://doi.org/10.1093/gigascience/giz133)

The file-list API currently exposes exactly three files. Its checksum fields
are empty, so the fetcher additionally checks object metadata and reproduces
the digest locally. The archive's ETag has a multipart suffix and is **not** an
MD5; its MD5 comes from the object's base64 `x-amz-meta-md5chksum`. The two
small single-part object ETags are MD5s. SHA-256 values identify the exact bytes
downloaded for this study.

| GigaDB file id | file | catalog / object bytes | MD5 | local SHA-256 |
|---:|---|---:|---|---|
| 419467 | `mrk-and-cnt_datasets.tar.gz` | 11,067,348,725 / 11,067,348,725 | `0c7156a80ede01ee1a62e1f489f2677f` | `e58921007e5fcd93775ed2b9fc276e13d65e74f6ddc1d2188cbbf89a04df3434` |
| 379994 | `readme_100660.txt` | 3,187 / 3,267 | `9ccf901cb40a6cc8ce007ba8c32cd9c7` | `29c25b539ab191a77341356249ef2702852244dcefb64cd182951791b17092ab` |
| 379942 | `questionnaires_answers.csv` | 2,147 / 2,147 | `c860cabdb782311a0506cc9a0e986b85` | `5decd7baa975dfb5de98ff9f0c5fd38f372ca451ba02f516ca827a69dc3c6813` |

The README size discrepancy is upstream catalog/object drift, not a tolerated
local mismatch: the frozen config records both values and requires the live
API to remain at catalog size 3,187 while HEAD and the local file remain 3,267.
Any other inventory, size, ETag, metadata MD5, local MD5, or local SHA-256 fails
closed. Total official downloaded bytes are **11,067,354,139**.

The dataset and DataCite records declare `CC0-1.0`; no login, token, data-use
agreement, or manual approval was required. Raw artifacts and the original
archive remain preserved at
`/home/whwovy/eeg-data/raw/choi2019_gigadb`. The extractor will not overwrite
an existing raw member.

## 2. Reproducible fetch, audit, and preparation

The immutable contract is `configs/data/choi2019.yaml`.

```bash
python scripts/fetch_choi2019.py --probe-remote
python scripts/fetch_choi2019.py
python scripts/prepare_choi2019.py --audit-only
python scripts/prepare_choi2019.py
```

`fetch_choi2019.py` snapshots the official GigaDB dataset/file responses and
DataCite response, validates their exact identity/license/inventory, validates
each object's HEAD metadata, supports range-resume through a `.part` file, and
only promotes bytes after MD5 and SHA-256 match. An existing verified file is
never redownloaded. A mismatching existing file is left unchanged and the
fetch fails; it is never silently overwritten. The immutable receipt preserves
the first successful run's transport status while idempotent reruns require the
same file identities and hashes.

An `--only` fetch is explicitly partial: it writes a selection-addressed
`download_receipt.partial-<digest>.json` and can never create or occupy the
canonical full-inventory `download_receipt.json`. Before extraction, schema
audit, preparation, or questionnaire parsing, the preparation entry points
rehash **all three** frozen raw files and require the configured byte count,
MD5, and SHA-256 for each one.

The archive audit accepts only the frozen path grammar, regular files, and the
exact 30-subject × 2-day × 3-band × 2-session × `{cnt,mrk}` grid. The observed
archive has 720 regular MAT files, 90 directories, and 11,109,841,350
uncompressed regular-file bytes. In addition to the archive-level digest, each
member is streamed and SHA-256 hashed. The canonical aggregate is SHA-256 over
path-sorted JSONL records with sorted keys and compact separators, one record
`{"path":...,"sha256":...,"size_bytes":...}` per MAT file. Both the verified
archive and the extracted tree must equal
`86db8a34b66b14b92c02b41334ea05f8a730be042750ab4950ca2d24933f5202`.
Reusing an extracted tree therefore rehashes every MAT byte and rejects any
missing, duplicate, extra, symlink, or non-regular entry; a matching filename
alone is never sufficient. The full source-schema audit found:

- 360 exact `cnt`/`mrk` pairs and 14,400 stimulus trials;
- MATLAB v7.3/HDF5 continuous arrays with 39 rows, distributed at 200 Hz;
- 33 EEG rows plus respiration, gyro, temperature, EMG1, EMG2, and ECG1;
- continuous lengths from 111,944 to 115,660 samples;
- 80 one-hot alternating stimulus/fixation markers per run, containing exactly
  ten stimulus trials for each of four targets; and
- marker times as one-based indices into the distributed 200-Hz signal, not
  milliseconds.

Preparation filters each continuous recording before cropping, keeps the 33
EEG channels, excludes but records the availability of the six auxiliary
channels, crops 2.0 s starting 0.14 s after each stimulus marker, and maps EEG
channels into the canonical 64-slot representation. The manifest retains day,
band, session, marker order, and pre-normalization signal-QC provenance. It
never fabricates impedance or electrode type.

Preparation is directory-atomic. It obtains an exclusive sibling lock, writes
all seven final assets to a new sibling staging directory, verifies the complete
asset inventory and absence of temporary filenames, and publishes with one
rename. It refuses to overwrite an existing final directory. Failed staging
directories are deliberately preserved for diagnosis; they are not promoted
or silently deleted.

The published analysis flagged ten participant-days for unusually large
non-stimulation-frequency amplitudes: S2 days 1/2, S10 day 2, S11 days 1/2,
S13 day 2, S18 day 1, S20 day 1, and S29 days 1/2. Preparation retains every
trial and records the flag; silently reproducing the paper's aggregate-only
exclusion would change the target population.

The questionnaire has one Before/After pair per participant but no published
day linkage. It is normalized into a companion table and is never joined to
trials or supplied to a model. S7's source rows disagree (`M` Before, `W`
After); only Before yields the paper's descriptive 21/9 aggregate, and neither
value is eligible as trial-level context. The GigaDB sample API's per-subject
sex assignment also conflicts with the questionnaire and must not be used.

## 3. Frozen Choi evaluation view and FBCCA compatibility

The exact outcome-free anchor is
`configs/baselines/fbcca_choi2019_bandwise_v1.yaml`.

Each recorded run contains only four simultaneously available targets:

| band | candidates (Hz) | frozen harmonics | highest requested component |
|---|---|---:|---:|
| LOW | 5, 5.5, 6, 6.5 | 5 | 32.5 Hz |
| MID | 21, 21.5, 22, 22.5 | 4 | 90 Hz |
| HIGH | 40, 40.5, 41, 41.5 | 2 | 83 Hz |

Consequently, evaluation is three band-specific four-class tasks, never an
invented twelve-class task. All three bands are required and are averaged with
equal weight within each participant before participant-level inference; no
band may be selected or dropped after outcomes are seen.

The source does not report the chronological acquisition order of LOW, MID,
and HIGH runs. `[LOW, MID, HIGH]` is frozen solely as deterministic processing
order and must not be interpreted or modeled as acquisition metadata. The
evaluation montage is exactly `[PO7, PO3, POz, PO4, PO8, O1, Oz, O2]`, canonical
channel IDs `[53, 55, 56, 57, 59, 61, 62, 63]`, in that order. Every one of the
14,400 manifest rows and every corresponding HDF5 channel mask must contain all
eight positions; missing-channel fallback is forbidden for this anchor.

The existing Chen-M3 configuration cannot be copied literally at the
distributed 200-Hz sampling rate. With `n_harmonics=5`, MID requests
`22.5 × 5 = 112.5 Hz` and HIGH requests `41.5 × 5 = 207.5 Hz`, both outside
the strict 100-Hz Nyquist limit; `predict_fbcca` correctly raises an error.
Upsampling would not recover those unrecorded components. LOW is Nyquist-safe,
but the parent first filter-bank cutoff of 6 Hz removes the 5- and 5.5-Hz
fundamentals.

Before outcomes, the smallest bandwise compatibility correction was frozen:

- LOW: keep five harmonics and change only the first subband from 6–90 to
  4–90 Hz;
- MID: keep the parent filter bank and use the largest common strictly
  Nyquist-safe count, four harmonics; and
- HIGH: keep the parent filter bank and use the largest common strictly
  Nyquist-safe count, two harmonics.

This is a separately named external-anchor view, not a silent alteration of
the original baseline and not a hyperparameter search. Direct parent-config
results for MID/HIGH are formally incompatible and must not be reported.

The seven V2 c4 FBCCA subband weights are explicit decimal float64 values in
the Choi anchor. Runtime evaluation of `m**(-1.25)+0.25` is forbidden because
NumPy/libm implementations can differ by one ULP. The anchor validator resolves
each band with the explicit vector and requires byte-for-byte float64 equality;
the little-endian vector SHA-256 is
`50f6a45a03855dbbded636553388571db189bdc1291b7b180b436e54f199b282`.
The exponent and offset remain provenance only.

For the cross-day few-shot view, Day 1 is the only support day. Budgets are
`k={0,1,3,5}` trials per class, selected as nested chronological prefixes by
numeric session then source marker order. Both complete Day-2 sessions—80
queries per band—remain the same immutable query set at every budget. There is
no random split seed. This tests signal-only cross-day calibration behavior;
it does not turn Choi's day or band labels into evidence for a wet/dry or
impedance mechanism.

The executable outcome-free constructor makes 20 complete Day-1 pseudo-blocks
per participant × band. Pseudo-block `j` contains the `j`th chronological trial
of each of the four within-band labels, where chronology is numeric source
session then source-marker index. Budgets 1, 3, and 5 use pseudo-blocks 1,
1–3, and 1–5 respectively; budget 0 is empty. Day 2 contributes both sessions,
exactly 80 immutable query trials per band at every budget. The constructor
checks exact counts, class balance, nesting, support/query disjointness, marker
grids, and query-set identity without reading or producing decoding scores:

```bash
python scripts/build_choi2019_partition.py          # inspect identity receipt
python scripts/build_choi2019_partition.py --write  # verify governance, publish once
```

`configs/data/choi2019_processed_partition_v1.yaml` binds all seven processed
assets, the data/anchor/canonical-channel configs, the exact montage and
weights, and the partition identities. The frozen partition has 32,040 JSONL
assignments (8,671,724 bytes; SHA-256
`4e354b6ae73e09b027c68a9191e3b929b01fd7bd17a16dd3b82ee408b2deddea`),
14,400 sample identities (digest
`1e9b4f4e93cb1afd5a1003138c7572a547aaa2d26881783eaa5dfd44cf6b73fb`),
and group/query identity digest
`de55ef91d3735d4c2c9d4db69c847aaba042f85483ae02de26ea3841552fce32`.
The external receipt and assignments live under
`/home/whwovy/eeg-data/processed/choi2019_v1/partitions/` and contain no scores,
predictions, or decoding outcomes. The publisher rejects symlinks anywhere in
the processed-root, governance-config, or output-parent path. It flushes and
`fsync`s the assignment and receipt, changes all three bundle files to mode
`0400`, records the receipt's SHA-256 in the detached
`governance_receipt.sha256`, `fsync`s the staging directory, renames it, and
then `fsync`s the parent directory. A pre-rename failure leaves the named
staging directory intact and never creates the canonical final directory.

## 4. License and human-subject boundaries

| source | present access | allowed current role | unresolved boundary |
|---|---|---|---|
| Choi2019 / GigaDB 100660 | public, dataset CC0-1.0; consent included anonymous public release; Kumoh IRB 6250 | cross-day/band signal-only anchor; public derivatives subject to repository policy | no wet/dry, impedance, or randomized interface context |
| Liu et al. 2022 | article CC BY; raw data “available upon reasonable request”; Tsinghua IRB and written consent reported | none until data and written terms arrive | raw-data license, storage/derivative/model rights, exact block impedance fields, consent scope, participant overlap |
| Xing et al. 2018 | article CC BY; no public raw-data deposit or raw-data license identified | none until authors reply | raw-data access/license, exact impedance values, consent/IRB reuse scope, participant overlap |

An article's CC BY license licenses the article, figures, and article text; it
does not by itself grant a license to an unpublished raw EEG dataset. Requested
files remain non-redistributable until the data owner explicitly states the
terms. Do not commit raw human EEG, participant keys, signed URLs, request
attachments, or credentials to this repository.

For any requested dataset, obtain written answers on:

- research-only versus commercial use;
- local/institutional storage and collaborator access;
- publication of aggregate statistics, trained weights, code, and derived
  non-identifying features;
- raw or transformed-data redistribution;
- deletion/retention date and security requirements;
- whether original consent and ethics approval cover this secondary analysis;
- stable pseudonymous subject/day/electrode/block keys and any known overlap
  with Zhu et al. 2021 or other released cohorts; and
- whether per-channel impedance is recorded before every block with unit,
  missingness, and time linkage.

## 5. Ready-to-send Liu request

Primary source: [Liu et al., 2022](https://doi.org/10.3389/fnins.2022.863359).
The official article lists Guoya Dong (`dong_guoya@126.com`) and Yijun Wang
(`wangyj@semi.ac.cn`) as corresponding authors. It reports 16 participants,
three days, both wet and dry sessions, randomized headset order, six blocks per
session, and impedance recorded before each block.

Subject: Request for de-identified wet/dry SSVEP data and block impedance
(Liu et al., 2022)

> Dear Prof. Dong and Prof. Wang,
>
> We are studying whether acquisition context can safely reduce subject-specific
> calibration for SSVEP decoding. Your 2022 Frontiers in Neuroscience study
> (DOI 10.3389/fnins.2022.863359) is uniquely relevant because it used repeated
> days, randomized wet/dry headset order, and impedance recorded before each
> block.
>
> Could you share the de-identified raw or minimally preprocessed EEG, event
> markers/codebook, participant/day/headset/block identifiers, randomized
> headset order, and the per-channel impedance records linked to each block?
> We would also appreciate the acquisition/preprocessing metadata needed to
> reproduce the published trial epochs and a description of missing or excluded
> blocks.
>
> Before transfer, could you please confirm in writing: (1) the raw-data
> license or data-use agreement; (2) whether institutional storage and access by
> named project collaborators are allowed; (3) whether aggregate results,
> non-identifying derived features, analysis code, and trained model weights may
> be published; (4) whether any raw or derived data may be redistributed; and
> (5) whether the original consent/IRB permits this secondary analysis?
>
> To prevent statistical double-counting, could you also state whether any
> participants overlap with Zhu et al. (2021), the 102-participant wearable
> SSVEP release, Xing et al. (2018), or another public cohort? Stable anonymous
> overlap keys would be sufficient; no identifying information is requested.
>
> We will preserve all missingness, pre-register the role of the dataset before
> viewing decoding outcomes, acknowledge the dataset and team, and follow any
> security, retention, authorship, or citation conditions you specify. We can
> provide a short protocol and institutional contact details on request.
>
> Sincerely,
>
> [name, role, institution]
>
> [project title and contact]

## 6. Ready-to-send Xing request

Primary source: [Xing et al., 2018](https://doi.org/10.1038/s41598-018-32283-8).
The Nature record directs correspondence to Yijun Wang or Weihua Pei. Use the
publisher's correspondence link for Prof. Pei rather than guessing an address;
Prof. Wang's official address is also listed in the Liu article above. The
paper reports 11 participants, a randomized dry/wet order separated by four
hours, 12 targets, 10 training blocks and five testing blocks. It reports
adjusting maximum dry-electrode impedance below 50 kΩ and checking stability,
but does not establish that exact values were saved per block.

Subject: Request for paired randomized wet/dry SSVEP data (Xing et al., 2018)

> Dear Prof. Wang and Prof. Pei,
>
> We are studying whether acquisition context can safely reduce
> subject-specific calibration for SSVEP decoding. Your 2018 Scientific Reports
> study (DOI 10.1038/s41598-018-32283-8) provides a valuable within-participant,
> randomized wet/dry comparison.
>
> Could you share the de-identified raw or minimally preprocessed EEG, event
> markers and frequency/phase codebook, participant/headset/block identifiers,
> randomized headset order, and training/testing split? If numeric impedance
> values were saved—not only checked against the reported 50-kΩ threshold—could
> you also share each per-channel measurement with its time or block linkage,
> units, and missingness?
>
> Please also confirm the raw-data license/data-use terms, storage and
> collaborator-access rules, permission to publish aggregate results,
> non-identifying features, code and trained weights, redistribution limits,
> and whether the original consent/IRB covers this secondary analysis. To avoid
> double-counting, please state whether participants overlap with Liu et al.
> (2022), Zhu et al. (2021), or another released cohort. Stable anonymous
> overlap keys are sufficient; no identifying information is requested.
>
> We will pre-register the dataset role before viewing outcomes, preserve all
> missingness, acknowledge the data and team, and comply with your security,
> retention, authorship, and citation requirements.
>
> Sincerely,
>
> [name, role, institution]
>
> [project title and contact]

## 7. Go/no-go after author responses

- **Go for direct metadata replication:** stable within-participant wet/dry and
  repeated-session keys, randomized order, block-linked numeric impedance,
  usable EEG/events, explicit reuse permission, and no unresolved cohort
  overlap.
- **Restricted go:** paired wet/dry EEG but no numeric block impedance. It may
  test interface matching, but not an impedance-distance mechanism; relabel the
  claim before outcomes.
- **Signal-only comparator:** EEG available without reliable interface/order or
  impedance. It can test safe few-shot adaptation, not metadata reliance.
- **No-go:** participant linkage, event codebook, permission, consent scope, or
  critical trial/block correspondence cannot be established.

Until one direct-metadata route clears its gate, Choi supports external
cross-day signal robustness only. It must not be used to imply that acquisition
metadata reduces calibration burden.

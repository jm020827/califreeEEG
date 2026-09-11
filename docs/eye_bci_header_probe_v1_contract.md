# Eye-BCI exact two-file header-only probe

2026-09-11. New narrow stage after four successful frozen S01 inventory calls.
Purpose: distinguish public metadata listing from actual anonymous file access
and check the first CSV header, not EEG/eye samples or efficacy.

Exact version1 targets, paired by observed S01/Sess01 hierarchy:
- Neuroscan SSVEP011.csv, syn64072665, parent syn64072621.
- Tobii SSVEP011.csv, syn64086409, parent syn64086404.

One attempt. At most2official FileEntity GETs and2versioned file GETs. A file GET
may follow at most2HTTPS redirects; count each redirect separately, max8network
requests overall. Only official repo host or amazonaws.com storage hosts allowed.
Timeout15s/request,120s overall. Stop first failedauth/status/identity/header;
no other session, credentials, account creation, terms acceptance or endpoint
workaround. Failed anonymous read does not prove the dataset is private.

Read metadata <=1MiB each. For each CSV read **only the first line**, up to64KiB,
then close the stream; no dataframe, samples, row labels, pupil values or video.
HTTP client buffering can prefetch body bytes beyond that line: no assertion
that transport delivered zero numeric bytes. Only header decoded/retained, no
numeric values inspected. Verify nonnumeric expected channel/Validity header
before recording field names. Reject unexpected/MAT/archive content.

Record exactentity/version/etag/dataFileHandleId, requests/redirectcounts,status,
firstlinebytecount/hash and headerfields only. Never log presigned URLs/queries.
No filesize/checksum-of-wholefile claim from a header. Keep output<=256KiB.
No raw/fullfile download, numeric human decode, training, outcomes or calibration
claim. A success only permits designing a later bounded timestamp/value adapter.

Root-only lightweight reader may have one generated unit-suite invocation,
<=8cases,0network/data/optimizer. Record failures without expanding that budget.
Freeze reader and tests before actual access. Originalclosures/held60/private/
paid restrictions persist. Header failure ends thisprobe; do not ask authors.

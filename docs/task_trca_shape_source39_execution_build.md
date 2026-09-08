# Task-shape source39 execution build (2026-09-08)

User continuation: “응 다음할일 계속 끝까지.” The fixed design is unchanged;
this continuation covers remaining infrastructure and one source39 development
attempt after artificial validation and a separately pinned executable manifest.
Held60, retired participants 1–3, full102 raw/M, and old closed studies are excluded.
No outcome-driven hyperparameter, seed, window, or arm changes are authorized.

Base: cdb988332cf45fa990aa6e8f0468d687d9eaaf61, integration owner: main/root.
Historical DESIGN_ONLY JSON remains immutable. Main owns execution authority,
shared schemas, integration, runtime, research SQLite, and end-to-end verification.

## Coordination

| Lane | Owned paths | Runtime | Integration |
|---|---|---|---|
| Reader | new `src/cfeg/analysis/task_trca_shape_archive.py`, matching test | separate reader worktree; CPU1, artificial fixtures only | first |
| Auditor | new `src/cfeg/analysis/task_trca_shape_audit.py`, matching test | separate audit worktree; CPU1, artificial fixtures only | second |
| Root | batching, runner, inference, manifest, docs and their tests | main; sole GPU / real-data / SQLite owner | contracts → leaves → E2E |

Two new checkouts budget64MiB plus test artifacts1GiB; real runtime output budget
12GiB, with ~235GiB free at start. Existing34 worktrees and all artifacts remain.
Read-only shared environments, no installs, dependency edits, pushes, or cleanup.

## Leaf contracts

Reader: explicit pinned regular archive and exact source39 participant allowlist;
role partition (fit/validation/evaluation), support k3/5, block5 only fit/validation,
query6–9 and native full/a0 only after verified all-outer-model freeze. Header/hash
integrity scans are distinct from decoding data. Numeric M only support0..k−1;
no query M. Denied calls fail before array bytes are decoded. No numpy.load of the
full native archive; stored ZIP64/NPY direct spans. Events record exact roles/blocks.

Auditor: independent NumPy/SciPy implementation, no imports from producer learning,
operator, features, or endpoint aggregation. Public pure APIs to audit:
`independent_prior(q, m, available, pipeline_record, arm, donor_m=None,
stale_m=None, stale_available=None)` -> R[5,8];
`independent_filters(s,c,anchors,r)` -> w[5,8,12];
`independent_scores(w, statistics, weights)` -> scores[n,12], where statistics is
a dict of query_gram[n,5,8,8], template_gram[12,5,8,8],
cross_gram[n,12,5,8,8], query_mean[n,5,8], template_mean[12,5,8], samples=int;
`summarize(scores,a0,coverage,actuation)` -> JSON-safe independent endpoints.
Scores axes [39,2,4,2,9,48,12], arms FULL,ISO,Q,Q2,QM,SHAM_REFIT,PERMUTED,
STALE,MISSING; a0[39,2,4,48,12]. Integers/argmax first; participant8condition
means and paired descriptive t intervals df38. Coverage is independently derived
changed-M fraction over78 k3 participant×interface units. Actuation records R,
filters, scores, and argmax differences, not just accuracy.

Both leaves may commit only their owned files and return revision, tests,
assumptions, evidence gaps. Main inspects semantic compatibility before cherry-pick.
No leaf may inspect real numeric EEG/M/outcomes or write the research database.

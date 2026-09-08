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

## Integrated engineering checkpoint

Reader7b0bcbe → main32ca62e; auditor2b57bbef → maind76f946, disjoint files and no
textual conflicts. Main evaluated semantic contracts and corrected event naming,
failed-decode attempt logging, case/Gram sample equality, exact grids/labels,
saved donor/M reconstruction, cached-native FULL argmax, and final receipt order.
Read-only review stayed in the shared repository. Two new worktrees stay intact.

Task-shape suite336 PASS38.57s; whole repository regression is running at this
checkpoint, not claimed complete. Artificial saved evaluation NPZ is reopened
and independently reconstructed; deliberate feature/key/donor/score/R corruption
and source-selection/label corruption are rejected. New cold auditor reconstructs
the three full source/evaluation folds, access rights, aggregate scores/A0,
coverage/actuation and exact terminal summary. It imports no producer code.

[Artificial batch receipt](reports/task_trca_shape_batch_engineering_v1.json):
8 generated cases, all4heads×200steps, CPU scalar14.826s / CPU batch5.781s /
CUDA batch4.298s. Largest CUDA-vs-scalar coefficient difference6.39e−16;
all tested argmaxes identical. Representative416-case forward/backward after
warmup: CPU .347–.376s, RTX4090 .0245–.0250s, peak allocated767,801,856bytes.
This is a computational workload probe, not an efficacy or full source39 fit.
Runtime frozen to float64 CUDA batch, one CPU thread, GPU process budget16% of
24GiB, output12GiB and whole-attempt7200s; no silent runtime/scientific fallback.

Authority is the user's current continuation of the fixed candidate. An exact
code/input/runtime manifest is written only after integrated verification; it
does not modify the historical DESIGN_ONLY document. No human EEG/M values have
been accessed at this checkpoint. A separate cold process must pass the saved
artifact audit before the final scientific report is accepted.

## Execution freeze

Whole-suite2213 PASS251.86s,68 existing Torch warnings. JUnit receipt:
`/home/whwovy/task-trca-shape-execution-engineering-pdnQo9/full-tests.xml`,
SHA81991c81ed7e38b142c93b2b51239488afa0010c41a402170a1f3a4ba9efd703.
The cold auditor's explicit method/interface weight-binding assertions were added
while the full suite ran; that CLI is not imported by the suite and is additionally
checked by Ruff/CLI import now and by its actual cold execution after the study.
The previously tested scientific operator/features/design are unchanged.

Separate [execution manifest](../configs/analysis/task_trca_shape_source39_execution_v1.json)
pins15 implementation dependencies, native input provenance, the exact old
design hash and all runtime/evidence limits. No final query can be read before
all3 models and their selection/source records are hashed in globalfreeze.
This checkpoint precedes the first actual input read of the new candidate.

## Terminal overlay

[실제 결과](task_trca_shape_source39_v1_results.md): source39 attempt는212.780초 뒤
outer1/inner0의SHAM 학습 중 native-anchor 정렬 하한 위반으로 VALIDITY_FAILURE 종료했다.
첫10pipelines만 완료했으며 final query/globalfreeze/score/result는 없다. 추가로 한 번의
실패-only source-statistic 재현(12.571초,새효능평가 아님)에서 finite Ccosine6.1057e−7을
독립CPU도 확인했다. 별도 실패경계/첫sourcefold cold 감사18.287초PASS,1456access exact,
held60/query0. 전체 완료-path cold score audit를 실제 자료에서 실행했다고 주장하지 않는다.
진단/실패감사 추가3인공tests PASS. 원 과학/실행pins는 그대로이며 이후 코드는 별도 진단만이다.

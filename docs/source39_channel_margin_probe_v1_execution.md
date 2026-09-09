# Channel-margin probe v1 execution contract

2026-09-10 KST. User approved the preceding exact scope with “응 시작.”
The scientific [design](source39_channel_margin_probe_v1_design.md) and config at
1f1f363 remain immutable prospective records (including their historical
`human_execution_authorized:false`). This new record grants the approved scope;
it does not rewrite old N1/generated-only authorities. Overall goal is incomplete.

## Ownership and resources

Base main1f1f36381b40ed3041ad0b214240cf8c8b7a6c44. Existing40worktrees preserved.
Free134510376KiB; reserve4GiB incremental (study1GiB/tests1GiB plus checkouts,
integration and logs2GiB). No new dependency installs, GPU or paid resources.

| Lane | Workspace | Owned files | Resources | Integration |
|---|---|---|---|---|
| root | main | this contract, new reader/run/launcher/tests, reports/DB | sole real-input access and execution | contracts→learner→audit→E2E |
| task_shape_reader | temporal-runtime, new branch codex/channel-margin-learner-v1 | source_channel_margin_probe.py, test_source_channel_margin_probe.py only | generated tests≤120s/128MiB | learner first |
| goal_scope_review | temporal-cold, new branch codex/channel-margin-audit-v1 | source_channel_margin_audit.py, test_source_channel_margin_audit.py only | generated tests≤120s/128MiB | independent audit second |

No shared-tree concurrent writes; reviewers/writers may read others' code but
must not read human arrays or prior model/query artifacts. Root final tests≤660s,
all tests aggregate≤900s/1GiB. No cleanup of old trees/branches/untracked files.

## Shared API, shapes and publication boundary

Pure study input mapping `data` (all float64 except integer IDs/order):

- `ids[p]`, `orders[p]` first-interface bit0/1;
- `support[p,2,3,12,5,8,N]`, `source_block[p,2,12,5,8,N]`;
- `packet[p,2,3,8]` first3 only, NaN-only missing, no future numeric M;
- `q[p,2,5,8,15]`, `target[p,2,5,8]` from frozen helpers;
- real run p39/N250; generated tests may use p>=9 divisible3/N>=24 and arbitrary
  nonnegative unique IDs. Generated variants are not human reader authority.

Learner API `fit_probe(data, event_sink=callable) -> models` JSON-safe dict;
`evaluate_probe(data, models) -> (predictions, result)` where predictions ordered
Q/Q2/QM/SHAM with shape[4,p,2,5,8]. Fit stores all inner selection/model/scaler/
donor/rank receipts required for independent reconstruction. It must not compute
or publish outer losses during fitting. Root persists models.json and a durable
globalfreeze.json event before calling evaluate_probe. Source block5 is allowed
as training target in other folds, never in its own predictive model fitting.

Auditor API `audit_probe(data, models, predictions, result) -> audit dict`.
It independently reconstructs target using scalar/vector Pearson math, Q via
shared frozen support_q_mask permitted/disclosed, M via independent equations,
fit-only scaling/paired donors/nested ridge/selected alpha and OOF predictions,
losses/rank/coverage/terminal. It must not call producer fitter/predict/decision
helpers to establish agreement. Model-schema details may be clarified between
agents before integration; science must not change. Audit up to120 deterministic
ridge reconstructions are accounted separately from primary120fits; these are
one audit replay, not new model search or a second efficacy attempt.

## New human input authority

Exact39 IDs and numeric-free native constants from frozen config
`configs/analysis/metadata_prior_source39_v1.json` SHA
ed67cc1f361ed89c37b6f7c5df1e9170b78484e428a5105020485c8c03e3934b.
Native root `/home/whwovy/metadata-prior-source39-v1/native-cold-r1`, manifest SHA
15c088048f3435a500d3758138c3cd6a1616e869d568bdf68d10bcc5e978d83a.
Source projection `/home/whwovy/context-template-artifacts/source39-v1/source-projection.json`,
SHA1082a81d23ebffe26e8451a9e99f1a7d4c84d0ba2d9d00c6695b3a5c6a5114d4.
Existing manifests may be read for input pins/lineage, never old model/outcome
arrays or raw MAT. Full archive byte hashes and metadata lexical indexing are
authorized for integrity; numeric decode only support0–2/block5/N250 and M0–2.

Composition facade only exposes support3/source_block5/metadata3; low-level
NativeArchive/SupportMetadata may be reused with explicit source-extraction fit
role, not as a forged old evaluation role. New reader owns this source-only
role and documents fold-specific learning exclusion. No query/FULL/A0 API.
Durable before-decode journal failure must prevent decode. No global mutation.

One primary extraction saves the permitted support/source arrays and packet
along with features so the single auditor need not extract the archives again.
Primary one attempt≤1800s/RSS4GiB; audit one≤900s/RSS4GiB; retry0/output1GiB.
Code/input/config/test pins registered before any human numerical reads. Fresh
exclusive output parent and file-backed logs, bounded child lifecycle, failure
record and exact PID termination/reaping. A completed result plus audit failure
is terminal AUDIT_FAILURE, never silently reported as valid efficacy.

## Engineering before registration

Generated tests cover pure fit/selection, target/scalers/donors/decision, denied
reads/journal failures, complete producer→audit, and perturbed-artifact rejection.
Tests are not efficacy simulations or extra human attempts. No mandatory new
registered synthetic research stage. Any failure after real registration ends
the attempt; pre-registration engineering fixes preserve their test records.
Result bounds and stopping rules remain exactly the prospective design.

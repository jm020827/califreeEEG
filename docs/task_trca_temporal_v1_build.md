# Temporal-score Q/QM integration — frozen implementation contract

2026-09-09. Base f32fa22a086aa407b2354d6f175558c702fe4705, integration branch main.
User authorized design + actual learner integration on generated arrays, NOT human execution.
Old15 execution code pins, old design/execution manifests and signfree primitive stay unchanged.
No source archive, diagnostic JSON, raw EEG, real numerical M or held60 reads this turn.

## Shared scientific/API contract

- Study/schema `task-trca-temporal-v1`; scorer `component-time-centered-ensemble-pearson-v1`.
- Use unchanged Q15 feature/global16parameter head, frozen Q then3parameter Q2/QM/SHAM_REFIT;
  exact-prefix mask-onlyQ, partition-local donors, fit-only scalers, same80/20logit budget,
  fixedR/tau/.1bound and200stepAdam/.001 default probe lambda, nested3lambda QCE-only selection.
- CE is `scores / sum(positive native weights) / .1`; all source cases/classes equally weighted.
- Use existing `task_trca_shape_signfree.bounded_projectors` and `TemporalGramStatistics`.
  No native anchor orientation in positive-mass score/loss. Support S/C/Q/templates unchanged.
- New modules `task_trca_temporal_learning.py`, `task_trca_temporal_batch.py`,
  `task_trca_temporal_evaluation.py`, `task_trca_temporal_audit.py`. No monkeypatch of old globals.
- Learner API mirrors old `make_task_case`, `fit_pipeline`, `predict`, `validation_ce`,
  `choose_lambda`, `nested_fit`. Expose `TaskCase`, `Pipeline`, `HeadFit`, `_head`, `_as_tensor`,
  `_readonly` for evaluation. New TaskCase rejects legacy global statistics; a separate new
  Pipeline type records `schema` and `score_schema`; loaders/predictors reject legacy models.
  Reuse old pure feature/optimizer/helpers only where semantics and global bindings are unchanged.
  Native support filter/anchors may remain stored for immutable native controls, never positive loss.
- `make_task_case` retains12classes/5bands/8channels and k3/5 support; allowed role fit/validation
  only, reject eval before coercing arrays. Pack keys(pid,interface,N,k),orders,q,m,available,s,c,
  anchors,weights,labels,packet5 and temporal query_gram/template_gram/cross_gram.
- `TaskBatch(cases).scores(logits[P,5,8])->scores[P,12,12]`; `.loss` matches scalar CE exactly,
  each case retains its weights and sample count. No alternate optimizer/batch stochasticity.
- Ordered evaluation ARMS:
  FULL_NATIVE,FULL_CENTERED,ISO,Q,Q2,QM,SHAM_REFIT,PERMUTED,STALE,MISSING.
  FULL_NATIVE uses exact unchanged native gamma0 fit/score. FULL_CENTERED uses SAME native
  filters/templates, only temporal score changes; no re-fitting or renormalizing the native W.
  A0 remains separately supplied frozen native k0 reference, not a learned/scorer-converted arm.
- Positive priors `[8,5,8]`, projectors `[8,5,12,8,8]`, scores `[10,48,12]`,
  native_filters `[5,8,12]`, native_full_correlations `[48,5,12]`,
  temporal statistics mapping keys(query_gram,template_gram,cross_gram,samples).
  Root evaluation uses an explicit complete evaluation partition, validates no fit IDs,
  deterministic exact-mask/order/condition donor identity; does not consume query labels/block5.
  Evaluation API uses `EvaluationPartition(states, evaluation_ids, conditions)` with expected
  IDs/grid supplied from a frozen contract, not inferred from provided states. `evaluate` and
  `frozen_prior` require keyword `partition=`. Exact expected Cartesian keys must match, so
  dropping a person/condition cannot turn an intended donor into an allowed singleton.
- Audit module imports NO producer (learner/features/signfree/operator/evaluation) modules.
  Old independent NumPy audit/helper reuse allowed. API:
  `independent_prior(q,m,available,pipeline_record,arm,donor_m=None,stale_m=None,stale_available=None)`;
  `independent_projectors(s[5,12,8,8],c,r[5,8])->F[5,12,8,8]` using SciPy generalized eig;
  `independent_scores(F,statistics,weights)->(scores[n,12],corr[n,5,12])` temporal Gram;
  `audit_training(data,model,outer_ids,evaluation_ids)` where model={pipeline,selection}
  and selection is learner nested_fit returned record. Independently recompute raw M/scalers/
  donors/prior/inner validationCE and Q-onlyselection, validate schema. No claim of independent
  Q15/rawpreprocessing/Adam reproduction. Reject bad schema, altered M/R/score/CE/donor/scaler.
  `summarize(scores[39,2,4,2,10,48,12],a0,coverage,actuation)` preserves previous gates,
  explicitly maps FULL_NATIVE to the original FULL guard and adds QM3−FULL_CENTERED3
  noninferiority guard in the correct direction: lowerCI(QM3−FULL_CENTERED3)>−1pp.
  Keep original metadata increment gates; the added centered guard can only demote calibration
  candidate to classification-only, never promote an old failure. Do not rename nulls to zero.

## Ownership / budget

| Lane | Owned new paths | Dependencies | Runtime / tests | Integration |
|---|---|---|---|---|
| Root/main | design/config, this contract, evaluation, engineering CLI, status/log/SQLite | shared contract | sole GPU/SQLite; end-to-end/fullsuite | contract then learner then audit then root |
| Learner writer | temporal_learning.py, temporal_batch.py, test_task_trca_temporal_learning.py, test_task_trca_temporal_batch.py | frozen contract/signfree primitive | CPU1 only, own worktree/basetemp | first |
| Independent audit writer | temporal_audit.py, test_task_trca_temporal_audit.py | frozen array/record/arm contract | CPU1 only, no producer imports/human I/O | second |
| Reviewer/shared repo | no writes | design + integrated code | read-only, no data/GPU | feedback only |

New2worktrees: each checkout≤64MiB + tests≤512MiB, existing root env read-only with
explicit PYTHONPATH to each worktree/src; no environment symlink/install. Root new output≤2GiB,
GPU cap≤1GiB, CPU1thread. Disk available239,664,448KiB (~228GiB); existing36trees preserved.
Reject concurrent writes in same tree. No push, cleanup, or deletion. Return commits/tests/
changedfiles/semantic issues. Main alone resolves contract changes and finalverification.

## Verification scope / stops

Fixed generated seed20260909,12class/5band/N17 andN23/k3,k5 as appropriate; no seed/window
efficacy screen. Scalar/batch CPU gradient/loss, full4heads×200 CPU/CUDA parity; tiny6-person
one-condition nested3fold×3lambda+final8000updates; fullschema roundtrip; native/centered/
missing controls; donor/role leaks; independent persisted generated training/score audit.
The one-condition nestedgraph does not certify actual39/8condition runtime or efficacy.
Any numerical failure is reported; no topgap/eta rescue. Old human failures stay closed.
Actual archive/manifest and completed-path cold provenance remain a later separately bounded
step; do not present pure in-memory APIs as an OS sandbox or a completed human runner.

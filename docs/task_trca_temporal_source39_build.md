# Temporal source39 runtime — shared contracts and isolated ownership

2026-09-09. Program metadata-learning-program-v1; C1 scientificdesign remains SHAe456c3bcfc066e6ba3cb95a9c76798a006416bee11d064d083d9bd98ce7b97da.
Mainownsprogram/design/executionmanifest/integration/SQLite/GPU/generatedend-to-end and actualhumanrun.
AgentswriteONLYnewownedfiles inisolatedworktrees. NOhumandatareads byagents or beforemanifest/preflight.
Oldpinnedcode/config/artifacts remainunchanged. Runtimefixedsource39 IDs/windows/weights fromoldprovenance.

## Frozen API and artifact contract

- New archive `task_trca_temporal_archive.py` exposes same ArchiveSpec/NativeArchive/SupportMetadata/
  RolePartition/SOURCE_IDS/SAMPLE_COUNTS/INTERFACES APIs; pureoldbytehelpers maybereused.
  New `verify_freeze(path, expected_sha256, expected_manifest_sha256)` returns newtoken, rejectsoldtoken.
  Newfreeze schema `cfeg.task_trca_temporal.all_models_frozen.v1`, otherenvelopekeys sameold.
  Validate model{fold}.json payloadpipeline schema/score_schema/fit_ids andmanifest_sha256 againstenvelope;
  exactparent/fixedbasename/outputbinding, notmerelyopaquemodelbytes. Oldclasses/globalsnotmonkeypatched.
- NewruntimeCLI `scripts/run_task_trca_temporal_source39.py`:
  `--manifest PATH --manifest-sha256 SHA`; schema `cfeg.task_trca_temporal.source39_execution.v1`,
  status EXECUTION_FROZEN; program_path/program_sha256 and C1design/nativeplan/input/codepinsrequired.
  Oldclosedrunnerneverimported/executed. Reusepurehelperlogicbynewimplementation whereappropriate.
  Existingnewtemporallearner/eval/audit modules unchanged unless root separatelyapprovescontractfix.
- Training artifact is unchanged `temporal_evaluation.pack_cases`: scalarUnicode schema/score_schema,
  keys[P,4],orders,q,m,available,s,c,anchors,weights,labels[P,12],packet5,3temporalGrams.
  Wrapindependent `audit_training` to enforce exacthuman16conditions andexpectedfitIDcartesianproduct.
  ModelJSON keys pipeline,selection,source_artifact,manifest_sha256,independent_source_audit asold.
- Evaluation perparticipant file `evaluation_S{pid:03d}.npz`, artifactkey `evaluation{pid}`;
  keys[P,4],orders[P],q[P,5,8,15],m[P,8,2],available[P,8],packet5[P,5,8],s/c/anchors/weights,
  donor_id[P],a0_correlations[P,48,5,12],cached_full_correlations[P,48,5,12].
  schema/score_schema are SINGLE scalarUnicode perfile/combinedfold, arms SINGLEUnicode[10].
  scores[P,10,48,12],r[P,8,5,8],projectors[P,8,5,12,8,8],native_filters[P,5,8,12],
  native_full_correlations[P,48,5,12]. Flatten temporal statistics as query_gram/template_gram/cross_gram;
  flatten original native_statistics as native_query_gram/native_template_gram/native_cross_gram/
  native_query_mean/native_template_mean. samples derivedfromkeys, noobjectarrays/dicts insideNPZ.
  Do NOTincludeactuationdict; independentlyderiveitfromarrays. `pack_evaluation(records)` and
  `combine_evaluation(parts)` are auditmodule purehelpers for scalarheaders and stackedrowarrays.
- New independentmodule `task_trca_temporal_artifact_audit.py`:
  exports sha,require,ARMS,SAMPLES,donor_positions,pack_evaluation,combine_evaluation,
  `audit_training(data,model,outer_ids,evaluation_ids)` fixedhumanwrapper,
  `audit_evaluation(data,pipeline,evaluation_ids)` exactexpected13person×16grid;
  imports onlyNumPy/SciPy/stdlib +oldandnewindependentaudit, NOproducer/nativelearner/features/operator.
  NativeFULL independentlyfromnativeGrams and savednativeW, sameprojectexact fullvsstoredcorr;
  cachednativecompatibility atol1e-9 plusargmax/countexact. FULL_CENTERED uses SAMEsavedWouterproducts.
  Positive8arms independentR/F/scoresatol1e-8, argmaxand integercountsEXACT acrossall10;
  Q/MISSING exact. VerifyMmaskprefix/donor/grid/scalers/weights/finitegeometry.
- Aggregate `scores.npz`: scores[39,2,4,2,10,48,12],a0[39,2,4,48,12],coverage_m,
  coverage_donor_m,coverage_available,r_changes,projector_changes,j_changes,qm_coefficients,weights,a0_weights.
  Callnewtemporal summarize with r_max_abs_qm_minus_q,projector_max_abs_qm_minus_q,j_max_abs_qm_minus_q.
- NewcoldCLI `scripts/audit_task_trca_temporal_source39.py`:
  `--output PATH --manifest PATH --manifest-sha256 SHA`; independentlybindexactmanifestbytes/startembeddedbody,
  originalmanifesthash/result/program/design/codepins; source/model/freeze/evaluation/artifactkeys MUSTpointto
  expectedfixedbasenamesandhashes/sizes insideoutput. Freeze.models mustEXACTLYmatchsavedmodel{fold}entries,
  fit/evalIDs andmodel.source_artifact/sourcefiles, notjustlen3. Derivefullaccess-eventmultiset
  (fold,role,pid,interface,samples,kind,blocks,freezehashwherequery) notaggregatekindcountsalone.
  Both failure.json andcold_audit_failure.json takeprecedence overanysuccess. coldappend-only0400.
  Failurepathmayreceive separateboundedreceipt butmustnotpublishCOMPLETE afterfailure.
  Reconstructaggregatefromperevalandverifyall10armsintegercounts+summaryexact.
  Readonlysavedartifacts, nohumanraw/Marchive/producerimports/optimizer.
- Mainmanifestlaterpinsallnewcode andunchangedscientificdependencies; generatedsource IDs cannotbeused
  toclaimactualhumanend-to-end. Tinygeneratedarchives mayusesource39-IDlabels butM/EEGmustbegeneratedandlabeled.
  SavedQ/S/Cstartpointaudit isnotindependentQ15/nativefit/Adam/rawEEGprocessing certification.

## Ownership and budget

| Lane | Owned files | Runtime | Integration |
|---|---|---|---|
| runtime writer | newtemporalarchive, runCLI, test_task_trca_temporal_archive.py, test_task_trca_temporal_runner.py | isolatedruntimeworktree; CPU1,generatedonly | first |
| cold writer | newtemporalartifactaudit, coldCLI, test_task_trca_temporal_artifact_audit.py, test_task_trca_temporal_cold.py | isolatedcoldworktree; CPU1,generatedonly | second |
| root | thiscontract,program/C2/executiondesigns,rootgeneratedintegrationCLI/tests,status/log/SQLite | soleGPU/mainverification; nohumanbeforegate | contract,lanes,rootintegration |
| history reviewer | nofiles | sharedrepo readonly | feedback |

Eachnewcheckout64MiB+test512MiB, outputoutsidecheckout≤program4GiB, noenvinstall/symlink.
Useexistingrootvenvreadonly withPYTHONPATH=<ownworktree>/src, PYTHONDONTWRITEBYTECODE=1,
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1. Mainrootfullsuite andCUDAonly.
Current38worktrees preserved; available237,484,296KiB. Twoadditionalcheckouts~24MiBarewellwithinbudget.
Freezecontracts beforeconcurrentwrite; commitownedfiles only; returncommit/diff/tests/semanticissues.

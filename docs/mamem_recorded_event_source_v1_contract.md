# MAMEM recorded-event source-learning v1 — shared implementation contract

2026-09-13, basefb47c76/main. One new acquisition-M candidate, original low-calibration
SSVEP goal unchanged. Previous turn=progress; active goal is not complete. This replaces
the prior source-learning draft with an implementable candidate before any new trial
labels/features/outcomes. No old negative-result rescue, held60, outreach or paid resource.

## Question, candidate and budgets

H: support recorded-event frame-grid residual MAD/lag1 explains repeat-template error
beyond EEG-Q/common conditions, improving one-/two-shot source/template shrinkage.
M is not verified physical jitter. One candidate, harmonics1/2, window2sec, ridgealpha.1,
no feature/penalty/model sweep. Max80 real multi-output gate fits (10LOSO×2k×4arms),
plus at most32 GENERATED integration fits (4fakepeople×2k×4arms). Prior/template estimates
are counted separately. Development S001 IO has0fits and is excluded from real efficacy.

Overall round deadline2026-09-13T13:20:00Z. Source/newnetwork0. At most22selected existing
MATmembers (S001–S011,filesa/b), each<=512MiB, <=4GiBnewextraction; no other files.
Use existing S001a whenhashmatches; no overwrite. Need8GiBfree+remainingextraction+1GiBcache
and1GiB for two writingworktrees/tests. Rawextraction/IO each<=120wall/90CPUsec,2GiBAS;
fitwholeprocess<=600wall/600CPU,4GiBAS. CPU/BLAS1 perworker. Outputsbounded; no raw arrays
inGit. Code/tests/source artifacts<=2MiB, outsideGit featurecache<=1GiB. One dev reader
attempt and one real preparation/fit attempt, with per-file/per-fit STARTED/COMPLETE;
failure stops without reopening another file/subject/config. Expiry stops, never restarts.

## Roles and chronology

Each filenamea is nominal acquisition run1/support; b is nominal run2/query/repeat target.
Do not infer wall-clock chronology from ZIP modification dates. This is a declared
offline cross-run support→query protocol using author run ordering, not online chronology
measurement. Main15trials only after8adaptation groups. Require23complete,validgroups,
main labels exactly3perclass, eachclass contiguousblock3 (orderneednotbeassumedascending).
No adaptation EEG is used, but all pre-support acquisition time is counted inprefixcost.

Within a, supportk1/2 means firstk maintrials PERCLASS by sample index. Query=all15main
trials inb, sameforbothk/allarms. Never use later a trials to improve k1/2 templates.
For targetLOSO, exclude entire target fromgate fitting,scalers,sourceprior,T* anddonors.
For each source pseudo-target p,class, supportT fromp/a firstk; T*=meanPSD ofp/b3class
trials; P excludes BOTH realtarget andp and uses all eligible maintrials a/b of remaining
sourcepeople. For realtarget use P fromall9sourcepeople. RawqueryDIN supplies evaluation
labels and the common predeclared window segmentation only, notgate/predictor features.
Querytimestamps/count/order/labels neverenterpredictor API.

## Frozen label/selection policy (not legacy floor-key fallback)

Authorcontinuous f_est=1000/(2mean within-group Δtimestamp_ms). Nominal frequencies
[6.66,7.50,8.57,10.00,12.00],zero-basedlabels0..4. Primarydecoder uses unique CLOSED
frequency guardbands nominal±onequarter nearest-neighbor nominal spacing. This deliberate
continuous-author decoder avoids importing MOABB's integer-division key convention as Hz.
There is no nearest-label fallback: outside/allambiguousbands STOP. This policy is fixed
before completeS001/otherDIN value reads, and label inference is reported as inferred,
not independentgroundtruth verification. Legacy keys may be diagnostic only, neverrepair.

Group split only on Δtimestamp>2000ms, all within-group deltas positive. Sample row4
one-basedpositiveinteger strictlyincreasing, timestamp row2 finite/increasing, rows1/3
descriptors untouched. Lastgroup completion requires fixed500sample window contained by
lastmarker and file (no invention of followingboundary). For eachgroup select
start0=firstsample−1+250,end0=start0+500,requireend0<=lastsample andend0<=T. No dropped
shortgroups/zipmisalignment. Requireexact23groupsbeforemain8:23. Use0-basedgroupindex.

M on support only: select group's timestamp events whose sample1−1 lies in[start0,end0),
require>=4events; computeexistingtwo-feature definition (τ=1000/60ms,numericalzero1e−9).
Count is common information; noabsolutetimestamp/meanperiod/framecountsequence/order M.
Forquery include_metadata=False must avoidcallingMextractor and returnmetadata=None.

## Shared pure API ownership (freeze before agents write)

Root owns this contract, engine, runner/manifests, docs/reports/SQLite and integration.
Parser lane owns ONLY `src/cfeg/mamem_events_v1.py`, `tests/test_mamem_events_v1.py`.
Features lane owns ONLY `src/cfeg/mamem_signal_v1.py`, `tests/test_mamem_signal_v1.py`.
No __init__/lockfile/otherfiles. Exception type ValueError with concise staticreason.
No I/O/network/realdata in either module or tests; NumPy/SciPy existingdependencies only.

Parser API: `parse_main_trials(din,total_samples,include_metadata=True)->list[dict]`.
Each of15dicts: group_index:int,label:int,start0:int,end0:int,event_count:int,
metadata:float64array(2,) orNone, trial_end0:int (firstsample−1+1250,forcost;must<=T).
Export `FREQUENCIES`float64array5 and `marker_features(timestamps)->array(2,)`.
InputDIN4xNobject,N<=10000,readONLYtimestamp/sample rows. Supportwindow500exact.

Features API: `analyze_window(eeg,start0,end0)->dict` takes raw257xTdouble, firstcopies
rows0:256 and500samples beforeanynumericalQC. finite,thenper-channelwindowmean subtraction,
then divide byglobal centeredRMS (stored unit invariant),stopzeroenergy; nofilter/reref/
channel-selection/montage. Return:
- covariance:(256,256)float64, Xscaled@Xscaled.T/500.
- factors:(5,2,256,2)float64. For each nominalfrequency/harmonic, QR-orthonormalize
  column-centered sin/cos references on the500sampletimeaxis, B=Xscaled@U/sqrt(500);
  H=B@B.T. Center references before all full/neighbor/half/joint QR operations.
  This is a REAL PSD from sine/cosine quadrature, not a full complex temporal template.
- q:(5,2,4): [relative projectionenergy, log harmonic-to-neighbor(.5Hz either side)
  energy ratio, effective spatialrank/C, split-half trace-normalized H discrepancy].
  Reference residual1−relativeenergy is derivable, no fabricated metadata.
- q2:(5,2): twoextraEEGfeatures, log1p(maxchannelpower/meanpower) and normalizedlag1
  autocorrelation (identicalacrosscandidatefrequencyallowed; preciselydocumentformulas).
- zero_shot:(5,): regularized multiharmonicCCA squaredlargestcorrelation using joint
  QR sin/cos h1/h2 reference and C+1e−6*trace(C)/C*I. No target labels orDIN argument.
Alloutputsfinite, units/source geometry notclaimed. Population normalization; noeps
amplitude cutoff inunknownphysicalunits. Freeze numericalzero floors inmodule/tests.

## Root engine and fair comparisons

Hnormalize=H/trace(H), trace(H)<=1e-12*trace(C) ->I/C (flag).
Effective rank is exp(-sum(p log p))/C from covariance eigenvalue proportions.
Neighbor log ratio floor is 1e-12*trace(C); half-PSD uses the same relative floor.
Q2 lag1 is the flattened shifted-window cosine; channel power uses diagonal(C).
Support T averageofk pertrialnormalizedPSD,
P/T* mean same normalizedPSD. A_h=(1−λ_h)T_h+λ_hP_h. Queryscore_j=sum_h h^{-1}*
tr(A_jh H_query_jh)/max(tr(A_jh C_query),1e−12); eigengramvalidity tested. Argmax stable
lowestindex tie. Baselines identical inputs: zero-shotCCA, λ0target-only, λ1source-only.
No transfer from a querylabel into this allcandidatefrequency scoring.

Q input=mean pertrial8Q values pluscommon onehotclass5,k,meanM-eventcount,window2s.
Q2 addsmean2Q2; QM adds2M; SHAM addsjointdonor2M. Ridgeoutput2harmonic λ* withunpenalized
intercept,train-onlypopulationstandardization/std<1e−9→scale1,alpha.1,λclip[0,1].
Oracle lambda minimizes Frobenius squared error to T*: clip(<T*-T,P-T>/||P-T||²,0,1);
denominator<=1e-24 returns0, recorded as degenerate. This uses source repeats only.
Subtracttrainclassmean M fromM beforeQM/SHAM; unobservedclassnotallowed. SourceSHAM
cyclesparticipantIDs within EXACT(class,k,mean eventcount)strata; singletonselfmarked,
donor vectors neverdimensionwisemixed. Eval uses realM inbothQM/SHAM. Recordchangedfraction;
if no conditionallychangingM/sourceSHAM anywhere,stopbeforeefficacyfits,notnewfeatures.

Count realfitmax80; sourceoracle targets10fold×2k×9pseudo×5class×2harmonic=1800;
targetsupport templatecases10×2×5×2=200; priorcachekeys realtarget+excludedpseudo explicit.
Persistmodels,scalarfeatures/lambda/predictions/sourceonlyauditcounts; no held60 orS001inreal.
SourceT* loss is secondarydiagnostic, actualtargetqueryaccuracy primaryutility. No data-driven
threshold/calibrationpolicyselection. Actualcalibrationpolicy is predeclared k1vsk2 comparison.

## Decision and honest cost

Newdatasetcriterion fixednow, not borrowed fromoldsource39 outcomes: promisingonlyif
QMk1−Qk1,QMk1−Q2k1,QMk1−SHAMk1 allmean>=.01, paired95%bootstrapCI lower>0,
and QM k1−Q k2 lower95%CI>=−.02 with feweractualacquiredprefixsamples. Harmcount at
participantdecline>.05reported. Fixedseed20260913/10000participantbootstrap draws,
nottrainingfits. This is exploratoryscreen, multiplicity/10subjectuncertainty explicit.
Perclass5labelsk1/10labelsk2 is not acquiredtimehalving: prefixcost=end oflatestselected
5sectrialfromrecordstart,includesrest/adaptation/interveningtrials. Compare samequeryb.
Unknownsensor/setup increments =>reportbreak-evenaddedsetup only, notnetsavingsproof.
Support-prefix seconds measure required support acquisition from a's record start;
actual query-ready elapsed remains UNKNOWN because the remaining a tail and a→b gap
have not been removed or measured. No chronological/online savings claim is licensed.
Baselinezero-shot accuracy and80%attainmentreported; ifzero-shotalreadyattains80%,
donotclaimcalibrationreductionforit. No onlineITR/simultaneousBCI generalization.

On schema/info/numeric failure stopandpreservephase/partialcounts; noselectivepeople,
files,labels,features,windows,thresholdsoroptimizerrescue. Ifnoeffect,retirethiscandidate
underthisprotocol; ifpromising,reportindependentconfirmationplan withoutheld60opening.

## Coordination budget and integration

Two new writingworktrees atfixedunique paths below, eachbudget<=256MiB checkout+tests,
existing `.venv/bin/python` invoked read-only withperworktree PYTHONPATH/tests/tmp; noinstall
orsymlinks. Root engine reads neitheragent workingtreeuntilcommit returned. Sharedcontract
changes rootonlyafterbroadcast. Integration parser→features→rootengine/runner; inspect
diffs/semanticinterfaces then targeted+combinedtests andseparateaudit. Preserveworktrees
afterintegration; nocleanupauthorization. Readonlyreviewer stayssharedmain.
- `/home/whwovy/califreeEEG-wt-mamem-events-v1`, branchcodex/mamem-events-v1.
- `/home/whwovy/califreeEEG-wt-mamem-features-v1`, branchcodex/mamem-features-v1.

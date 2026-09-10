# Whole-cohort metadata validation and fixed replay roles

2026-09-11. Base1778af5. Previous turn was PROGRESS:16participant publiccohort acquired,
sample event recovery and k1 primitives verified. Do not redownload or repeat access triage.

## Before any new file access

One pass on the exact96acquired files, no new files. Select only raw_fs,raw_clab,event,t.
Verify each pinnedSHA/size and before-after stat;1checksum pass+1selective decode/file.
At most600s batch/30s file/768MiB process address space;8192projector visitedunits/file;
report4MiB/newgeneratedartifacts16MiB; generated tests at most3calls, each fixture<=20000
numeric elements/64KiB. No raw_x/preprocess_x/interp_clab, fits/outcomes, network/paid/
outreach/held60/source39/GPU. Unexpected IO/parser/guard error stops with durable progress,
no automatic retry. Structural scientific mismatches are reported, not silently repaired.
The 600-second deadline covers checks, selective decoding, pairing and normal report
serialization; durable failure serialization and final fsync are outside that compute cap.

## Prespecified metadata checks

- Rawrate500Hz scalp/128Hz IMU; labels unique, rawcounts36/27; selected posteriorEEG
  Pz,PO3,POz,PO4,PO7,PO8,O1,Oz,O2 and HgyroX/Y/Z present.
- className5.45/8.57/12,60strictlyincreasingevents,20perclass; onehot exactly agrees decs;
  paireddevices classorder identical; t10..4000step10. Keep observed fields, not onlyPASS.
- Candidate marker sample=`round((event.time−1)*fs/1000)`; integergriderror<=1e-6sample,
  extraction bounds marker−0.5s through+1.5s fit raw shape. This operational interpretation
  does not establish exactphysicalonset/indexorigin. Record cross-device deltas, no fitted
  fullrun clock correction or threshold selected from EEG outcomes.
- Fixed query indices40..59(zero-based; trials41–60) must containall3classes. For k1/2/3/5,
  classifier/feature support is exactly firstk/class(3k total), allselectedindices<40.
  Count all chronological prefix trials through the latestselectedtrial as acquired cost.
  Unusedprefixtrials and gap before query are not model inputs. Record gap separately.
  A negative support-ready-to-query-marker gap is a structural overlap failure.
- If a run fails these prespecified metadata rules, mark it and its participant incomplete
  for primary3condition pairing; no per-k/outcome-favorable deletion. Require>=12complete
  participants for the proposed3outer/2inner development design, otherwise stop for redesign.

## How this connects to the actual hypothesis

After metadata passes, separately freeze raw extraction and fit execution. Proposedprimary
signal window is marker+0.5..1.5s(1second) with causal4–40Hz filtering initialized from−0.5s;
thus no supplied fullrun preprocessing or queryIMU informs a learned prior. Allarms use same
paidfirstk/classsupport and sharedspeed/order/k/prefixcost. Query41–60 is fixed acrossbudgets.

Accuracy-versus-budget curves and query-selected first80%hittingcost are retrospective
oracle descriptions, not deployable stopping. For a genuine label-budget reduction test,
choose eacharm's fixedk using only source inner-heldout predictions before outer-testoutcomes,
then evaluate its actualprefixcost and accuracy on outerheldout participants. The fixedquery
gap prevents claiming earlier firstclassification/fullsessionwallclock savings. IMUsetup
cost remains unknown; labelbudget benefits must not be called totalsetup savings.

Root sole main/DBwriter; two read-only feature/protocol reviewers. No new worktree. Previous
negative results, ChoiPARKED and held60 protections remain unchanged.

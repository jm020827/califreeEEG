# Public paired acquisition-M round2 — access and mechanism before training

2026-09-12 KST. User: 시작해보자. Base main527e3f6. This is a new finite
feasibility round, not a retry of Eye-BCI403, XR timing, Choi or closed gyro/source39.
The objective remains fewer target-user labeled calibration trials for closed-set
SSVEP, using external acquisition context beyond Q and common stimulus/geometry.

## Problem signature and portfolio

- Representation: trial/session-linked EEG, labels, common stimulus bank, Q,
  and independently measured external acquisition M.
- Bottleneck: executable public paired measurements with physical provenance,
  meaningful variation and lawful support/prequery availability; not GPU power.
- Allowed operation: public literature/repository/metadata inspection, followed
  by a schema proposal only if suitable. No human waveform/fit/outcome in this round.
- Feedback: access success, exact field/clock description, source of measurement,
  label shortcuts, actual support availability, additional setup cost.
- Failure modes: Q renamed M; nominal protocol/subjectID counted as new M;
  query gaze or true target cue shortcut; request-only material; task mismatch;
  constant/nonidentifiable M; no actual calibration cost accounting.
- Unknowns: which public releases include measured M; whether fields vary within
  comparable tasks; whether M remains useful beyond Q and deterministic correction.

Three lenses, at most three shortlisted dataset candidates:
1. Electrode/contact: actual per-channel impedance/contact telemetry, not a
   paper-wide impedance threshold or an EEG-derived contact score.
2. Visual acquisition: actual photodiode/frame/luminance measurements or recorded
   viewing geometry, not nominal frequency/contrast or target-specific gaze.
3. Independent sensor geometry/quality: externally measured recording/viewing
   context with plausible support learning; not revival of gyro source-routing.
Adjacent physiology-only, MI/ERP-only or corrupted-EEG-cleaning corpora may explain
absence but cannot substitute for the SSVEP objective.

## Frozen finite budget and access rules

- Root-only structured discovery: at most3queries, OpenAlex+Crossref up to10/source,
  n20/query, cutoff2026-09-04 unchanged. Up to6additional web locator queries
  (root2/contact scout2/visual scout2); results are leads, not read-paper evidence.
- At most16 distinct exact public text/API targets total: root8, contact scout4,
  visual scout4. Root's allocation includes up to4 metadata/schema targets after
  a suitable lead. Failed targets count; no pagination loops or reallocation
  that increases the aggregate budget. Repeat GET of a successful official target
  for provenance retention allowed at most4total, not a failure retry.
- No retry/bypass of403/429/CAPTCHA. No old parked endpoints, credential use,
  account creation, DUA acceptance, outreach or paid resources. Search-visible
  publisher abstracts are provisional if exact access fails.
- Retained public text <=16MiB total; each retained request <=2MiB,timeout30s.
  No PDF in this round: PDF skill unavailable; use official HTML/XML/README and
  explicitly scoped evidence, no PDF card promotion. No rawEEG/eye/video files,
  archive member bytes, numeric human rows, fits or new synthetic efficacy.
- Aim to complete within30min of first query; stop after budget or three assessed
  candidates, whichever first. Source failures and exclusions are preserved.
- If a candidate meets access+task+M provenance+time-boundary gates, freeze one
  exact-file schema preflight as the next action, without automatic efficacy run.
  If none meets them, report the negative feasibility result and unresolved gaps;
  do not add searches merely to obtain a positive candidate.

## Coordination and verification

Main alone writes docs, reports, academic SQLite and integrations. Both scouts
are read-only and return exact queries/URLs/access outcomes, author claims vs
inferences, physical-field provenance and unmet gates. A third read-only reviewer
checks prior-exclusion overlap and final claims. No shared-tree concurrent writes.
No new worktree: the task is read-only research plus one dependent synthesis lane;
41 existing worktrees/8 untracked pytest outputs preserved, disk99% with
22,042,536KiB initially free. No installs/cleanup/push. Main owns final source-hash,
JSON/links/status review; no repository-wide tests for literature-only changes.

Required learning design if a future candidate proceeds: common deterministic
correction baseline, matched Q/Q2/QM/SHAM, support-only M, participant-disjoint
source selection, fixed label-prefix costs plus setup burden, predefined harm,
retention and stopping rules. Access feasibility is not metadata efficacy.

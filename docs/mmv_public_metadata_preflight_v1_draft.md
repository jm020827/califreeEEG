# MMV exact-release metadata preflight — DRAFT, not executed

2026-09-12. This successor is not an efficacy candidate or authorization to use
query gaze/physiology. It answers the remaining round2 access/provenance question.

Exact seed: https://doi.org/10.57760/sciencedb.ai.00010 (MMV paper reference33).
The previous generic ai.scidb.cn homepage did not yield a dataset file inventory.

If taken forward, freeze at most6distinct public metadata/document/code targets,
2MiB/target,12MiBtotal,30s/network request,oneattempt/target,no auth/DUA/account/
request/paid. Read DOI record,license/version,file inventory and documented loader
schema,not CNT/EDF/MAT bodies,annotations or person-level numeric values. No
retry/bypass of403/429/challenges. Stop immediately if access needs authorization.

Decision questions, ordered:

1. Is the named release public and are raw offline EEG+eye files and loading code
   listed? Resolve exact subject/session/block only from documentation, not outcomes.
2. Are original tracking-validity/loss fields independently defined and available
   in the offline calibration block, distinct from blinks and amended PERCLOS?
3. Is the original four-target cue label preserved, with comparable clocks and
   boundaries? Do not use decoded online commands or gaze as target ground truth.
4. Can paid support and later query be separated without global preprocessing,
   future smoothing/imputation, or unsupported assumptions about task order?
5. What extra tracker calibration/setup burden must be counted?
6. Before any later learning nomination, what acquisition mechanism connects
   independent tracker validity to EEG support reliability or transfer beyond Q?
   A field's existence alone is insufficient. Reject substitution with physiology,
   gaze-derived target identity, or predicting tracker failure as the new objective.

Result must be DOCUMENTED_PAIR_CANDIDATE (metadata documentation only; requires
later bounded actualschema and a separately justified learning mechanism),
PARK_ACCESS_UNRESOLVED, or REJECT_M_OR_TASK_PROVENANCE. No learning is triggered
by finding a link. Keep lowcalibration SSVEP objective, Q/common correction/Q2/SHAM
controls and held60 boundaries. Use Markdown-only local text searches to avoid
the round2 accidental scan of old numerical report records. Any future raw reader
requires its own actualfile/roles/byte-boundary contract and generated validation.

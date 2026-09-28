# DAN teacher v1 — final result and handoff

Completed 2026-09-28 07:31 KST (2026-09-27T22:31:35Z). This is a
protocol-adapted, independently implemented DAN experiment, not an exact author-code replication.
The bounded candidate is closed: no promotion and no validated calibration reduction.

## Results

| Method | 24 calibration trials | 36 trials | 60 trials |
|---|---:|---:|---:|
| DAN Q | 28.18% | 33.87% | 46.18% |
| DAN QM | 28.31% | 34.56% | 46.09% |
| Target-only eTRCA | 41.08% | 48.08% | 57.18% |

Zero-calibration FBCCA: 60.84%. Primary k2/k3 QM minus Q: +0.414 pp,
descriptive participant-bootstrap 95% interval [-0.127, +1.256] pp.
QM minus Q2: +0.260 pp; QM minus shuffled metadata: +0.334 pp.
Seed-specific QM minus Q: -0.053 and +0.881 pp. Preregistered promotion gates failed.
These reused source39 development results do not establish universal metadata uselessness.
They do not justify claiming a successful low-calibration method or reproducing the paper's efficacy.

## Evidence

- [Result summary](reports/dan_teacher_v1_result_summary.json): original result fields except bulky trial rows and attainment records.
- [Saved-artifact audit](reports/dan_teacher_v1_audit.json): PASS; 7,020 cells, 42,120 selected states, 127,296 decisions; zero argmax disagreements.
- [Terminal receipt](reports/dan_teacher_v1_terminal.json): COMPLETE_AUDITED, 17,905.86 seconds total.
- [Frozen qualification](reports/dan_teacher_v1_qualification.json), [plan](dan_teacher_v1_plan.md), and [configuration](../configs/analysis/dan_teacher_v1.json).

The original producer result retains its historical AUDIT_PENDING status; the later audit
and terminal receipts establish completion. Audit verifies saved models, decoders and endpoints,
not independent raw preprocessing or every training update. No new fitting or evaluation
was performed to prepare this handoff. held60, external contacts and paid resources remain unused.

## Artifact storage and reproduction boundary

Full output (~5.7 GiB) remains outside Git at
`/home/whwovy/eeg-data/dan-teacher-human-v1-1sfi54y0`.
It contains cache, selected checkpoints, decoders, freeze.json, scores.npz, full result.json,
journal.jsonl, meter.json, audit.json and terminal.json. Preserve this directory separately;
GitHub alone is not a complete data/checkpoint backup. Raw datasets and the separate
`/home/whwovy/research-spaces/califree-eeg-experiment-design` workspace are also not bundled.

Original invocation (provenance only; do not rerun this closed one-shot experiment):

```bash
env PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -u -m cfeg.analysis.dan_runner \
  --qualification docs/reports/dan_teacher_v1_qualification.json
```

The qualification records Python/library versions and frozen implementation hashes.
On a different machine, first restore authorized data/artifacts and resolve configured local
paths; do not remove the exclusive start receipt to bypass no-retry protection.
A new replication or follow-up requires a separate plan and authorization.

Publication target: `research/local-state-20260928`; remote main has divergent history
and is deliberately not overwritten. Temporary pytest artifacts and large EEG/model files
are excluded. Earlier failures and preregistered thresholds are preserved.

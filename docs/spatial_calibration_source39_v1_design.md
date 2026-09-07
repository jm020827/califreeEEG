# Spatial calibration source39 v1 — pre-outcome contract

2026-09-07. Goal unchanged: test whether external acquisition context can reduce labeled calibration beyond a useful EEG-only learner. This new stage tests that learner prerequisite only. Prior context-template-source39-v1 remains closed with AQ_NOT_ESTABLISHED; no historical score or public helper is overwritten.

## 쉽게 설명

기존 방법은 여러 채널 파형을 길게 펼쳐 평균 파형과 바로 비교했다. 이번 IT-CCA는 현재 EEG와 보정 파형에서 각각 유용한 채널 조합을 찾고, 두 조합의 상관으로 자극을 구분한다. 보정 1/3/5회는 **각 자극당** 횟수이며 전체 labeled trial 수는 12/36/60이다. 같은 후반 5개 block을 모든 보정량에서 평가한다.

주 비교는 보정 없는 FBCCA(A0) 대 IT-CCA를 섞은 AQ다. 보정 정답을 11가지로 잘못 붙인 대조군과, 보정 내용을 버리고 균등확률만 섞은 대조군을 포함한다. 좋은 확률 보정만 생긴 것을 EEG 학습 성공으로 세지 않는다. Metadata, impedance, 순서 정보와 manifest는 읽지 않는다. Wet/dry는 결과를 나눠 보고하는 실험 조건일 뿐 모델별 가중치 입력이 아니다.

## Fixed scientific scope

[Machine contract](../configs/analysis/spatial_calibration_source39_v1.json) is authoritative. Only the same previously exposed39 participants are accessible, at all six dry/wet ×125/188/250 sample windows. Sorted-ID modulo3 creates fit26/evaluate13 folds. This is supervised nuisance fitting with outer cross-fitting, not nested validation or an independent confirmation cohort. Source39 historical exposure and shared training-fold dependence remain limitations.

One primary decoder family: ridge-regularized filter-bank IT-CCA. Joint fusion weight/temperature fits use only26 other participants per fold, pooled interfaces, separately by window and budget. Candidate order fixes exact ties toward lambda0. Standalone methods receive separately fitted probability temperatures; ranking accuracy does not depend on those temperatures. No best-family or best-condition choice after outcomes.

eAUC uses k0/1/3 only: BA0/6+BA1/2+BA3/3. The all-condition AQ increment and real-versus-wrong-label increment must each have mean>=1 percentage point and positive one-sided95 LCB. k1 mean cannot worsen; each interface has its own prespecified safety bound. k5,80% attainment and harm remain diagnostic. Passing establishes only useful EEG calibration in development, not metadata benefit or calibration saving.

## Source-grounded implementation differences

The [2015 original comparison](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140703), PDF p8 Eq13, defines IT-CCA using an averaged individual template and separate canonical spatial projections. Extended CCA on pp8–9/Eqs14–15 uses different projections/features and is **not** implemented here. Main read targeted PDF pp7–9 and visually inspected p8; no full-text-complete or transferred effect-size claim. The seven-band bank, ridge.01, chronological split and fusion are explicit variants of the operator, not exact reproduction of the paper's14-block comparison.

The [author TRCA implementation](https://github.com/mnakanishi/TRCA-SSVEP/tree/35c1b8a7ebfa3774dd7266931784d04ae09f3461/src) uses a linear filter-band correlation sum in test_model.m. Our old helper uses signed squares, generalized-eigenvector scaling and global flattened centering. These differences predate this study; they do not establish the cause of its poor dry performance. Current author train_model.m defaults to fastTRCA, so the current repository is not assumed to reproduce the2018 experiment bitwise.

Two diagnostic TRCA arms distinguish the untouched legacy helper from a named unit-Euclidean-filter, component-centered, linear-band-score variant. Its Q/S/ridge formula is fixed in JSON. Arbitrary eigenvector sign must not change that new score. Neither diagnostic is selected to replace primary AQ after seeing results.

## Artifacts, verification and ownership

Four immutable artifacts: start.json before any human EEG open; score-only cache; all three fold-freezes; result. Existing start consumes the attempt. Only explicit allowlisted S{ID}.mat paths through pure loaders; no glob, global preparation, metadata or held data. CPU4/BLAS1; study RNG none. New runtime/test budget1GiB with >300GiB free; checkout about11MiB. No dependency or GPU/environment changes and no old worktree cleanup.

Main owns this contract, docs, independent score auditor, integration and final verification, and is sole literature SQLite writer. One isolated worktree owns only new decoder module/runner/tests; shared reviewer is read-only. Root integrates by three-way reviewed cherry-pick. Fixture checks include scalar CCA parity, query locality, prefix discipline, label permutation equivalence, exact lambda0 and uniform argmax, TRCA sign/DC invariance, and fit/evaluation isolation. The independent auditor recomputes4770 fit objectives,5382 metric rows and derived summaries from saved scores; it does not claim to independently replay raw human preprocessing.

No result exists at contract time. Further metadata experiments require a separate frozen protocol; neither positive nor negative learner results unlock held60 or retired subjects1–3. Independent acquisition-metadata replication data remains unacquired, and the author-request draft remains unsent.

"""Twelve preregistered toy ridge fits; not real EEG efficacy or strong-Q validation."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
from datetime import datetime, timezone
import probe_mamem_i_scalar_metadata_v1 as common

require, Stop, sha, write_json, now = common.require, common.Stop, common.sha, common.write_json, common.now
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'docs/reports/recorded_event_shrinkage_generated_v1_run'
DEADLINE = '2026-09-13T12:10:00+00:00'
TAU, EPS, ALPHA = 1000 / 60, 1e-9, 0.1
SCOPE = dict(generated_only=True, real_EEG_reads=0, actual_strong_Q_tested=False,
             actual_metadata_utility_tested=False, accuracy_evaluations=0,
             calibration_savings_evaluations=0, OOD_generalization_tested=False,
             held60=0, random_draws=0, seed_label=20260913)


def features(timestamps):
    import numpy as np
    z = np.asarray(timestamps, dtype=float)
    require(z.ndim == 1 and len(z) >= 4 and bool(np.isfinite(z).all()), 'invalid_timestamps')
    delta = np.diff(z)
    require(bool((delta > 0).all()), 'nonincreasing_timestamp')
    residual = delta - TAU * np.rint(delta / TAU)
    residual[np.abs(residual) < EPS] = 0.
    centered = residual - np.median(residual)
    mad = float(np.median(np.abs(centered)))
    a, b = centered[:-1], centered[1:]
    if min(float(np.std(a)), float(np.std(b))) < EPS:
        correlation = 0.
    else:
        correlation = float(np.corrcoef(a, b)[0, 1])
    return np.array([mad, correlation])


def timestamps(state, schedule=0, offset=0.):
    import numpy as np
    frames = np.resize([4, 4, 5, 4, 3] if schedule == 0 else [2, 3], 16)
    r = np.resize([-2., 2.], 16) * state
    return offset + np.r_[0., np.cumsum(TAU * frames + r)]


def donor_map(n):
    import numpy as np
    require(n > 0 and n % 8 == 0, 'balanced_multiple_of_eight_required')
    i = np.arange(n)
    return (i & ~7) | ((i >> 1) & 1) | (((i >> 2) & 1) << 1) | ((i & 1) << 2)


def world(name, n):
    import numpy as np
    require(name in ('positive', 'null', 'schedule_only'), 'unknown_world')
    i = np.arange(n)
    target = (i & 1).astype(float)
    state = (i & 1) if name == 'positive' else ((i >> 1) & 1) if name == 'null' else np.zeros(n, dtype=int)
    metadata = np.array([features(timestamps(int(s), int(j & 1) if name == 'schedule_only' else 0))
                         for j, s in zip(i, state)])
    return target, metadata


def mix(template, prior, amount):
    return (1 - amount) * template + amount * prior


def oracle(template, prior, target):
    import numpy as np
    d = prior - template
    norm = float(np.vdot(d, d).real)
    return 0. if norm <= 1e-12 else float(np.clip(np.vdot(d, target - template).real / norm, 0, 1))


def fit_gate(x, target):
    import numpy as np
    mean, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    scale = np.where(scale < EPS, 1., scale)
    centered = (x - mean) / scale
    intercept = float(target.mean())
    beta = np.linalg.solve(centered.T @ centered + ALPHA * np.eye(x.shape[1]),
                           centered.T @ (target - intercept))
    return mean, scale, beta, intercept


def predict(model, x):
    import numpy as np
    mean, scale, beta, intercept = model
    return np.clip((x - mean) / scale @ beta + intercept, 0, 1)


def score(metric, projected, covariance):
    import numpy as np
    denominator = float(np.trace(metric @ covariance).real)
    require(denominator > 1e-12, 'zero_query_metric_energy')
    return float(np.trace(metric @ projected).real / denominator)


def evaluate(ledger=None):
    import numpy as np
    template, prior = np.diag([1., 0.]), np.eye(2) / 2
    invariants = {}
    for state in (0, 1):
        invariants[f'schedule_offset_invariance_state{state}'] = bool(np.allclose(
            features(timestamps(state)), features(timestamps(state, 1, 1e6)), atol=EPS, rtol=0))
    results, fit_count, positive_qm = {}, 0, None
    for name in ('positive', 'null', 'schedule_only'):
        lam_train, metadata = world(name, 32)
        lam_eval, eval_m = world(name, 16)
        train_targets = np.array([mix(template, prior, v) for v in lam_train])
        oracle_target = np.array([oracle(template, prior, t) for t in train_targets])
        invariants[name + '_oracle_matches_constructed_target'] = bool(np.array_equal(oracle_target, lam_train))
        donor = donor_map(32)
        shuffled = metadata[donor]
        table = np.zeros((2, 2), dtype=int)
        for y, row in zip(lam_train.astype(int), shuffled):
            table[y, int(row[0] > 1)] += 1
        permutation_ok = sorted(map(tuple, metadata)) == sorted(map(tuple, shuffled))
        invariants[name + '_sham_multiset_preserved'] = permutation_ok
        if name != 'schedule_only':
            invariants[name + '_sham_independent'] = bool(np.all(table == 8))
        else:
            invariants['schedule_only_M_is_exact_zero'] = bool(np.array_equal(metadata, np.zeros((32, 2))))
        arms = dict(Q=np.zeros((32, 2)), Q2=np.zeros((32, 4)),
                    QM=np.c_[np.zeros((32, 2)), metadata], SHAM=np.c_[np.zeros((32, 2)), shuffled])
        losses, models = {}, {}
        for arm, x in arms.items():
            number = fit_count + 1
            if ledger is not None:
                write_json(ledger / f'fit_{number:02d}_started.json',
                           dict(status='FIT_STARTED', world=name, arm=arm, started_utc=now()))
            model = fit_gate(x, oracle_target)
            fit_count += 1
            if ledger is not None:
                write_json(ledger / f'fit_{number:02d}_complete.json',
                           dict(status='FIT_COMPLETE', world=name, arm=arm, ended_utc=now()))
            ex = np.zeros((16, x.shape[1])) if arm in ('Q', 'Q2') else np.c_[np.zeros((16, 2)), eval_m]
            predicted = predict(model, ex)
            error = np.array([np.linalg.norm(mix(template, prior, p) - mix(template, prior, t), 'fro')**2
                              for p, t in zip(predicted, lam_eval)])
            losses[arm] = float(error.mean())
            mean, scale, beta, intercept = model
            models[arm] = dict(mean=mean.tolist(), scale=scale.tolist(), beta=beta.tolist(),
                               intercept=intercept, eval_predicted_lambda=predicted.tolist())
            if name == 'positive' and arm == 'QM': positive_qm = model
        results[name] = dict(template_frobenius_mse=losses, models=models, sham_cross_table=table.tolist(),
                             sham_changed_fraction=float(np.mean(np.any(metadata != shuffled, axis=1))),
                             train_episodes=32, eval_episodes=16)
    invariants['twelve_fits'] = fit_count == 12
    invariants['positive_QM_ability'] = results['positive']['template_frobenius_mse']['QM'] < .01
    for name, row in results.items():
        for arm, loss in row['template_frobenius_mse'].items():
            if not (name == 'positive' and arm == 'QM'):
                invariants[name + '_' + arm + '_baseline'] = abs(loss - .125) <= 1e-10
    # Six metric/score forwards, using one trained model and two metadata-only settings.
    intervention_x = np.array([np.r_[0., 0., features(timestamps(s))] for s in (0, 1)])
    amounts = predict(positive_qm, intervention_x)
    matrices = [mix(template, prior, v) for v in amounts]
    scores = [score(a, template, np.eye(2)) for a in matrices]
    same_matrices = [mix(prior, prior, v) for v in amounts]
    same_scores = [score(a, template, np.eye(2)) for a in same_matrices]
    pure_scores = [score(a, np.eye(2), np.eye(2)) for a in matrices]
    invariants['k1_metadata_changes_metric_and_score'] = bool(np.linalg.norm(matrices[1] - matrices[0]) > .5
                                                            and abs(scores[1] - scores[0]) > .4)
    invariants['same_template_prior_cancels'] = bool(np.array_equal(same_matrices[0], same_matrices[1])
                                                    and abs(same_scores[1] - same_scores[0]) <= 1e-12)
    invariants['pure_reference_query_cancels'] = bool(np.allclose(pure_scores, 1, atol=1e-12, rtol=0))
    return dict(status='GENERATED_ABILITY_PASS' if all(invariants.values()) else 'GENERATED_ABILITY_FAIL',
                scope=SCOPE, fits=fit_count, additional_score_forwards=6, worlds=results, invariants=invariants,
                intervention=dict(lambda_values=amounts.tolist(), scores=scores,
                                  same_template_prior_scores=same_scores, pure_reference_scores=pure_scores),
                limitation='Designed missing-Q information; deterministic repeated train/eval states; not human efficacy')


def check_time():
    require(datetime.now(timezone.utc) < datetime.fromisoformat(DEADLINE), 'deadline')


def check_manifest(cfg):
    require(cfg == dict(producer_sha256=sha(__file__), helper_sha256=sha(common.__file__),
                        deadline_utc=DEADLINE, fits=12, worlds=3, train_episodes=32, eval_episodes=16,
                        alpha=.1, residual_zero_epsilon=EPS, scaler_std_floor=EPS), 'manifest_not_frozen')


def invariant_names():
    names = {f'schedule_offset_invariance_state{s}' for s in (0, 1)}
    for name in ('positive', 'null', 'schedule_only'):
        names.add(name + '_oracle_matches_constructed_target')
        names.add(name + '_sham_multiset_preserved')
        if name != 'schedule_only': names.add(name + '_sham_independent')
        for arm in ('Q', 'Q2', 'QM', 'SHAM'):
            if not (name == 'positive' and arm == 'QM'): names.add(name + '_' + arm + '_baseline')
    return names | {'schedule_only_M_is_exact_zero', 'twelve_fits', 'positive_QM_ability',
                    'k1_metadata_changes_metric_and_score', 'same_template_prior_cancels',
                    'pure_reference_query_cancels'}


def validate_report(result):
    require(result.get('scope') == SCOPE and result.get('fits') == 12
            and result.get('additional_score_forwards') == 6, 'child_scope_or_fit_count')
    invariants = result.get('invariants', {})
    require(set(invariants) == invariant_names() and all(type(v) is bool for v in invariants.values()),
            'child_invariant_schema')
    worlds = result.get('worlds', {})
    require(set(worlds) == {'positive', 'null', 'schedule_only'}, 'child_worlds')
    for name, row in worlds.items():
        losses = row.get('template_frobenius_mse', {})
        require(set(losses) == {'Q', 'Q2', 'QM', 'SHAM'}, 'child_arms')
        for arm, loss in losses.items():
            require(type(loss) in (int, float) and math.isfinite(loss) and loss >= 0, 'child_loss')
            key = 'positive_QM_ability' if name == 'positive' and arm == 'QM' else name + '_' + arm + '_baseline'
            expected = loss < .01 if key == 'positive_QM_ability' else abs(loss - .125) <= 1e-10
            require(invariants[key] is expected, 'child_loss_invariant_mismatch')
    passed = all(invariants.values())
    require(result['status'] == ('GENERATED_ABILITY_PASS' if passed else 'GENERATED_ABILITY_FAIL'), 'child_status')


def child(config, output):
    require(output == RUN / 'child.json' and not output.exists() and not (RUN / 'terminal.json').exists(),
            'worker_output_or_already_terminal')
    start = json.loads((RUN / 'started.json').read_bytes())
    require(start['manifest_sha256'] == sha(config) and start['producer_sha256'] == sha(__file__)
            and start['helper_sha256'] == sha(common.__file__), 'worker_binding')
    write_json(RUN / 'worker_claim.json', dict(claimed_utc=now(), attempts=1))
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    for limit, value in ((resource.RLIMIT_AS, 2 * 1024**3), (resource.RLIMIT_CPU, 30), (resource.RLIMIT_FSIZE, 32768)):
        resource.setrlimit(limit, (value, value))
    result = dict(started_utc=now(), planned_scope=SCOPE)
    try:
        check_manifest(json.loads(Path(config).read_bytes()))
        check_time()
        result.update(evaluate(RUN))
        check_time()
    except Exception as error:
        result.update(status='STOPPED', error_type=type(error).__name__,
                      reason=str(error) if isinstance(error, Stop) else 'details_not_exported')
    result['ended_utc'] = now()
    write_json(output, result)


def run(config, output):
    check_manifest(json.loads(Path(config).read_bytes()))
    check_time()
    require(output.resolve() == RUN, 'unpinned_output')
    output.mkdir()
    write_json(output / 'started.json', dict(started_utc=now(), attempts=1, manifest_sha256=sha(config),
                                           producer_sha256=sha(__file__), helper_sha256=sha(common.__file__),
                                           planned_fits=12, planned_scope=SCOPE))
    terminal = dict(status='STOPPED', attempts=1, planned_fits=12)
    try:
        with (output / 'stdout.log').open('xb') as out, (output / 'stderr.log').open('xb') as err:
            remaining = (datetime.fromisoformat(DEADLINE) - datetime.now(timezone.utc)).total_seconds()
            require(remaining > 0, 'deadline')
            done = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                                   str(config.resolve()), str(output / 'child.json')],
                                  stdout=out, stderr=err, timeout=min(30, remaining), check=False)
        terminal['child_returncode'] = done.returncode
        require(done.returncode == 0, 'child_nonzero')
        check_time()
        for name in ('stdout.log', 'stderr.log', 'child.json'):
            require((output / name).stat().st_size <= 32768, 'child_output_cap')
        result = json.loads((output / 'child.json').read_bytes())
        validate_report(result)
        terminal.update(status=result['status'], fits=12, child_sha256=sha(output / 'child.json'), scope=SCOPE)
    except Exception as error:
        terminal.update(error_type=type(error).__name__,
                        reason=str(error) if isinstance(error, Stop) else 'details_not_exported')
    finally:
        terminal['fit_started_receipts'] = len(list(output.glob('fit_*_started.json')))
        terminal['fit_completed_receipts'] = len(list(output.glob('fit_*_complete.json')))
        terminal['ended_utc'] = now()
        write_json(output / 'terminal.json', terminal)
    return terminal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.worker: child(args.config, args.output)
    else:
        result = run(args.config, args.output.resolve())
        print(json.dumps(result, allow_nan=False))
        return 0 if result['status'] == 'GENERATED_ABILITY_PASS' else 1


if __name__ == '__main__': sys.exit(main())

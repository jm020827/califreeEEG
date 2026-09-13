"""Three fixed generated observation models; no human inputs, EEG decoder or fit."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SEED_LABEL = 20260913  # Deterministic fixture: no random draws.
EVENTS = 120
FRAME_COUNTS = (4, 4, 5, 4, 3)
JITTER_MS = (0, 2, -1, 3, -2)
BASELINE_DELAY_MS = 4.0
TOY_FREQUENCIES_HZ = np.array([10.0, 20.0, 30.0])
TOLERANCE = 1e-12


def observe(optical, delay):
    marker = np.asarray(optical) + np.asarray(delay)
    if not np.all(np.asarray(delay) >= 0):
        raise ValueError('negative_absolute_delay')
    # Illustrative nearest rounding, not a claim about NetStation's exporter.
    z = np.rint(marker).astype(np.int64)
    s = np.rint(marker / 4).astype(np.int64)
    return z, s


def construct():
    frames = np.resize(np.array(FRAME_COUNTS), EVENTS - 1)
    intended = 1000 + np.r_[0, np.cumsum(frames)] * (1000 / 60)
    j = np.resize(np.array(JITTER_MS, dtype=float), EVENTS)
    fixed_delay = np.full(EVENTS, BASELINE_DELAY_MS)
    worlds = [
        ('intended_frame_schedule', intended.copy(), fixed_delay),
        ('optical_change', intended + j, fixed_delay.copy()),
        ('shared_marker_path_change', intended.copy(), fixed_delay + j),
    ]
    canonical = np.broadcast_to(np.exp(1j * np.array([0.2, 0.4, 0.6])), (EVENTS, 3)).copy()
    output = []
    for name, optical, delay in worlds:
        z, s = observe(optical, delay)
        optical_deviation = optical - intended
        # Toy event-coordinate phase representation, not time-sampled neural EEG.
        phase = 2 * np.pi * optical_deviation[:, None] * TOY_FREQUENCIES_HZ / 1000
        response = canonical * np.exp(-1j * phase)
        output.append(dict(name=name, optical=optical, delay=delay, z=z, s=s, response=response))
    return intended, j, canonical, output


def oracle_correct(response, optical_deviation):
    phase = 2 * np.pi * np.asarray(optical_deviation)[:, None] * TOY_FREQUENCIES_HZ / 1000
    return response * np.exp(1j * phase)


def report():
    intended, j, canonical, worlds = construct()
    normal, optical, shared = worlds
    rows = []
    for w in worlds:
        delta_z, delta_s = np.diff(w['z']), np.diff(w['s'])
        rows.append({
            'name': w['name'], 'events': EVENTS,
            'true_optical_deviation_max_abs_ms': float(np.max(np.abs(w['optical'] - intended))),
            'absolute_marker_delay_min_max_ms': [float(w['delay'].min()), float(w['delay'].max())],
            'timestamp_interval_min_max_ms': [int(delta_z.min()), int(delta_z.max())],
            'timestamp_interval_std_ms': float(delta_z.std()),
            'delta_z_minus_4delta_s_max_abs_ms': int(np.max(np.abs(delta_z - 4 * delta_s))),
            'observation_sha256': hashlib.sha256(np.stack([w['z'], w['s']]).astype('<i8').tobytes()).hexdigest(),
        })
    correction = optical['optical'] - intended  # Oracle truth, not inferred from DIN.
    corrected = oracle_correct(optical['response'], correction)
    wrong = oracle_correct(shared['response'], correction)
    # Copies only show equality of a common transform; they are not trained arms.
    copies = {name: oracle_correct(optical['response'].copy(), correction.copy())
              for name in ('Q', 'Q2', 'QM', 'SHAM')}
    before = float(np.max(np.abs(optical['response'] - canonical)))
    after = float(np.max(np.abs(corrected - canonical)))
    wrong_error = float(np.max(np.abs(wrong - canonical)))
    invariants = {
        'normal_spread_without_optical_departure': rows[0]['timestamp_interval_std_ms'] > 0
            and rows[0]['true_optical_deviation_max_abs_ms'] == 0,
        'nearest_quantization_delta_bound_5ms': all(r['delta_z_minus_4delta_s_max_abs_ms'] <= 5 for r in rows),
        'all_marker_delays_nonnegative': all(np.all(w['delay'] >= 0) for w in worlds),
        'physical_vs_shared_z_identical': np.array_equal(optical['z'], shared['z']),
        'physical_vs_shared_s_identical': np.array_equal(optical['s'], shared['s']),
        'physical_vs_shared_true_optical_different': not np.array_equal(optical['optical'], shared['optical']),
        'physical_vs_shared_toy_response_different': not np.array_equal(optical['response'], shared['response']),
        'oracle_correction_recovers_canonical': before > TOLERANCE and after <= TOLERANCE,
        'same_oracle_correction_is_wrong_in_shared_world': wrong_error > TOLERANCE,
        'all_named_copies_same_correction': all(np.array_equal(v, corrected) for v in copies.values()),
    }
    return {
        'schema': 'cfeg.mamem-timing-identifiability-generated.v1',
        'status': 'GENERATED_INVARIANTS_PASS' if all(invariants.values()) else 'GENERATED_INVARIANT_FAILURE',
        'seed_label': SEED_LABEL, 'random_draws': 0, 'scenario_count': 3, 'scenarios': rows,
        'invariants': invariants,
        'oracle_algebra': {'uncorrected_max_complex_error': before,
                           'corrected_max_complex_error': after,
                           'wrong_world_max_complex_error': wrong_error,
                           'tolerance': TOLERANCE},
        'scope': {'real_data_inputs': 0, 'new_raw_opens': 0, 'fits': 0, 'accuracy_evaluations': 0,
                  'held60': 0, 'learned_arms_instantiated': False,
                  'quantized_DIN_to_true_j_recovery': False,
                  'EEG_plus_DIN_nonidentifiability_proved': False,
                  'MAMEM_physical_or_marker_error_verified': False,
                  'calibration_savings_evaluated': False},
        'conclusion': 'DIN_pair_alone_is_nonidentifying_under_the_constructed_shared_path_error_model',
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output')
    args = p.parse_args()
    now = datetime.now(timezone.utc)
    if now >= datetime.fromisoformat('2026-09-13T03:20:00+00:00'):
        raise RuntimeError('round_deadline')
    result = report()
    result['run_at_utc'] = now.isoformat()
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with Path(args.output).open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()

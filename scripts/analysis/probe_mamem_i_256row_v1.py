"""Pinned one-window EEG input gate; slice before numerical QC; no model fitting."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import probe_mamem_i_scalar_metadata_v1 as common

Stop, require, sha, write_json, now = common.Stop, common.require, common.sha, common.write_json, common.now
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'docs/reports/mamem_i_256row_v1_run'
DEADLINE = '2026-09-13T12:10:00+00:00'
LIMITS = dict(address_space_bytes=2 * 1024**3, cpu_seconds=30, wall_seconds=60,
              each_output_bytes=32768, free_reserve_bytes=8 * 1024**3)
PINS = dict(mat_path=common.MAT, mat_sha256=common.MAT_SHA, mat_bytes=137357437,
            role_path=common.ROLE, role_sha256=common.ROLE_SHA,
            din_report_path='/home/whwovy/data/mamem_i_v1_20260913/din_probe.json',
            din_report_sha256='c16c288ddcfb1d17e5f9c2e2ae1a0fed5f9090378f5cab7f37f3ea40d025f4e0',
            sampling_receipt_path=str(ROOT / 'docs/reports/mamem_i_scalar_adapter_v1_run/child.json'),
            sampling_receipt_sha256='09e994853875fb406c73201e0d1f2983bc5d4790de5abef22aa9ef52176106e9',
            eeg_shape=[257, 117917], din_shape=[4, 1966], deadline_utc=DEADLINE,
            variables=['DIN_1', 'eeg'], limits=LIMITS,
            rows=[0, 256], offset_samples=250, window_samples=500,
            din_sample_cells=[[3, 0], [3, 71]], descriptor_read=False)
SCOPE = dict(EEG_variable_fully_decoded=True, DIN_variable_fully_decoded=True,
             EEG_numeric_scope='copied rows[0:256], fixed marker-relative500samples only',
             DIN_numeric_cells=[[3, 0], [3, 71]], timestamp_or_descriptor_values_inspected=False,
             EEG_excluded_values_used_in_QC=False, physical_latency_corrected=False,
             class_labels_inferred=False, raw_arrays_exported=False, fits=0, held60=0)


def deadline_ok():
    require(datetime.now(timezone.utc) < datetime.fromisoformat(DEADLINE), 'deadline')


def validate_manifest(cfg):
    require(set(cfg) == set(PINS) | {'producer_sha256', 'helper_sha256'}, 'manifest_keys')
    require(all(cfg[k] == v for k, v in PINS.items()), 'changed_frozen_contract')
    require(cfg['producer_sha256'] == sha(__file__), 'producer_not_frozen')
    require(cfg['helper_sha256'] == sha(common.__file__), 'helper_not_frozen')


def bound_json(path, expected_sha):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size < 65536,
            'invalid_receipt_file')
    require(sha(path) == expected_sha, 'receipt_hash_mismatch')
    return json.loads(path.read_bytes())


def validate_input(cfg):
    path = Path(cfg['mat_path'])
    require(path.is_file() and not path.is_symlink() and str(path.resolve()) == cfg['mat_path'],
            'regular_canonical_MAT_required')
    require(path.stat().st_size == cfg['mat_bytes'], 'MAT_size_mismatch')
    role = bound_json(cfg['role_path'], cfg['role_sha256'])
    require(role['subject'] == 'S001' and role['subject_role'] == 'development_only_all_records',
            'S001_development_role_required')
    require(role['member'] == 'EEG-SSVEP-Part1/S001a.mat'
            and role['member_selection'] == 'lexicographic_first_MAT_across_both_archives',
            'member_selection_mismatch')
    require(role['mat_path'] == str(path) and role['mat_sha256'] == cfg['mat_sha256']
            and role['bytes'] == cfg['mat_bytes'], 'role_binding_mismatch')
    din = bound_json(cfg['din_report_path'], cfg['din_report_sha256'])
    require(din['mat_sha256'] == cfg['mat_sha256'] and din['subject'] == 'S001'
            and din['events_in_first_group_prefix'] == 72 and din['group_complete'] is True
            and din['boundary_observed'] is True, 'DIN_report_binding_mismatch')
    sr = bound_json(cfg['sampling_receipt_path'], cfg['sampling_receipt_sha256'])
    require(sr['mat_sha256'] == cfg['mat_sha256'] and sr['subject'] == 'S001'
            and sr['status'] == 'MATCH' and sr['stored_samplingRate_hz'] == 250,
            'sampling_receipt_binding_mismatch')
    require(sha(path) == cfg['mat_sha256'], 'MAT_hash_mismatch')
    return path


def sample_scalar(value, total):
    import numpy as np
    require(isinstance(value, np.ndarray) and value.size == 1 and value.dtype.kind in 'iuf',
            'sample_scalar_numeric_required')
    number = float(value.item())
    require(math.isfinite(number) and number.is_integer() and 1 <= number <= total,
            'sample_scalar_range_or_integer')
    return int(number)


def window_from_din(din, eeg_samples, expected_shape):
    import numpy as np
    require(isinstance(din, np.ndarray) and din.dtype == object
            and list(din.shape) == expected_shape, 'unsupported_DIN_array')
    first = sample_scalar(din[3, 0], eeg_samples)
    last = sample_scalar(din[3, 71], eeg_samples)
    require(first < last, 'nonincreasing_endpoint')
    start, end = first - 1 + 250, first - 1 + 750
    require(end <= last and end <= eeg_samples, 'window_not_contained_no_fallback')
    return dict(start0=start, end0=end, first_marker_sample1=first, last_marker_sample1=last,
                samples=500, nominal_sampling_hz=250, end_policy='last_marker_sample_inclusive')


def select_copy(eeg, window, expected_shape):
    import numpy as np
    require(isinstance(eeg, np.ndarray) and list(eeg.shape) == expected_shape
            and eeg.dtype.kind == 'f' and eeg.dtype.itemsize == 8, 'unsupported_EEG_array')
    start, end = window['start0'], window['end0']
    require(type(start) is int and type(end) is int and 0 <= start < end <= eeg.shape[1]
            and end - start == 500 and eeg.shape[0] == 257, 'invalid_window_or_rows')
    # First value operation: isolate. No full-array isfinite, scaling, QC, reference or filter.
    return np.array(eeg[:256, start:end], dtype='<f8', order='C', copy=True)


def summarize(selected):
    import numpy as np
    require(selected.shape == (256, 500), 'selected_shape')
    require(bool(np.isfinite(selected).all()), 'selected_nonfinite')
    # Equality, not variance: avoid overflow and arbitrary physical-unit thresholds.
    constant = np.all(selected == selected[:, :1], axis=1)
    n_constant = int(np.count_nonzero(constant))
    return dict(status='INPUT_INTEGRITY_PASS' if n_constant < 256 else 'INPUT_ALL_CONSTANT',
                selected_shape=[256, 500], all_finite=True, constant_channels=n_constant,
                max_abs_stored_units=float(np.max(np.abs(selected))), voltage_unit='unverified_stored_units',
                selected_sha256=hashlib.sha256(selected.tobytes(order='C')).hexdigest(),
                hash_encoding='little-endian float64 C-order', SSVEP_presence_verified=False)


def inspect(cfg, loader=None, header_loader=None, access=None):
    access = [] if access is None else access
    path = validate_input(cfg)
    if loader is None or header_loader is None:
        from scipy.io import loadmat, whosmat
        loader, header_loader = loader or loadmat, header_loader or whosmat
    headers = header_loader(path)
    for name, shape, cls in [('eeg', cfg['eeg_shape'], 'double'), ('DIN_1', cfg['din_shape'], 'cell')]:
        matches = [(list(s), c) for n, s, c in headers if n == name]
        require(matches == [(shape, cls)], 'header_missing_duplicate_or_shape')
    def load_one(name):
        access.append(name + '_load_requested')
        loaded = loader(path, variable_names=[name], squeeze_me=False,
                        struct_as_record=True, verify_compressed_data_integrity=True)
        require(name in loaded and set(loaded) <= {'__header__', '__version__', '__globals__', name},
                'unexpected_loaded_variable')
        access.append(name + '_decoded_whitelist_validated')
        return loaded[name]
    din = load_one('DIN_1')
    window = window_from_din(din, cfg['eeg_shape'][1], cfg['din_shape'])
    access.append('two_sample_cells_and_window_validated')
    del din
    deadline_ok()
    eeg = load_one('eeg')
    selected = select_copy(eeg, window, cfg['eeg_shape'])
    access.append('selected_256x500_copied_before_QC')
    del eeg
    result = summarize(selected)
    return dict(result, window=window, mat_sha256=cfg['mat_sha256'], subject='S001', scope=SCOPE)


def worker_claim(config, output):
    require(output == RUN / 'child.json', 'unpinned_worker_output')
    require(not output.exists() and not (RUN / 'terminal.json').exists(), 'worker_already_terminal')
    started = json.loads((RUN / 'started.json').read_bytes())
    require(started['status'] == 'STARTED' and started['manifest_sha256'] == sha(config)
            and started['producer_sha256'] == sha(__file__)
            and started['helper_sha256'] == sha(common.__file__), 'worker_start_binding_mismatch')
    write_json(RUN / 'worker_claim.json', dict(claimed_utc=now(), attempts=1))


def child(config, output):
    worker_claim(config, output)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = '1'
    for limit, value in ((resource.RLIMIT_AS, LIMITS['address_space_bytes']),
                         (resource.RLIMIT_CPU, LIMITS['cpu_seconds']),
                         (resource.RLIMIT_FSIZE, LIMITS['each_output_bytes'])):
        resource.setrlimit(limit, (value, value))
    report = dict(started_utc=now(), planned_scope=SCOPE, access_stages=[])
    try:
        cfg = json.loads(Path(config).read_bytes())
        validate_manifest(cfg)
        deadline_ok()
        report.update(inspect(cfg, access=report['access_stages']))
        deadline_ok()
    except Exception as error:
        report.update(status='STOPPED', error_type=type(error).__name__,
                      reason=str(error) if isinstance(error, Stop) else 'details_not_exported')
    report['ended_utc'] = now()
    write_json(output, report)


def validate_child(report, cfg):
    require(report.get('scope') == SCOPE and report.get('subject') == 'S001'
            and report.get('mat_sha256') == cfg['mat_sha256'], 'child_binding_or_scope')
    require(report.get('selected_shape') == [256, 500] and report.get('all_finite') is True,
            'child_shape_or_finite')
    require(report.get('SSVEP_presence_verified') is False
            and report.get('voltage_unit') == 'unverified_stored_units'
            and report.get('hash_encoding') == 'little-endian float64 C-order', 'child_meaning')
    count = report.get('constant_channels')
    require(type(count) is int and 0 <= count <= 256, 'child_constant_count')
    require(report.get('status') == ('INPUT_INTEGRITY_PASS' if count < 256 else 'INPUT_ALL_CONSTANT'),
            'child_status_mismatch')
    maximum = report.get('max_abs_stored_units')
    require(type(maximum) in (int, float) and math.isfinite(maximum) and maximum >= 0, 'child_max_abs')
    digest = report.get('selected_sha256')
    require(type(digest) is str and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
            'child_hash')
    win = report.get('window', {})
    require(win.get('samples') == 500 and win.get('nominal_sampling_hz') == 250
            and win.get('end_policy') == 'last_marker_sample_inclusive', 'child_window_convention')
    require(all(type(win.get(k)) is int for k in ('start0', 'end0', 'first_marker_sample1', 'last_marker_sample1')),
            'child_window_type')
    require(win['start0'] == win['first_marker_sample1'] - 1 + 250
            and win['end0'] == win['start0'] + 500 and win['first_marker_sample1'] >= 1
            and win['end0'] <= win['last_marker_sample1'] <= cfg['eeg_shape'][1], 'child_window_bounds')


def run(config, output, runner=None):
    cfg = json.loads(Path(config).read_bytes())
    validate_manifest(cfg)
    deadline_ok()
    output = Path(output)
    require(output.resolve() == RUN, 'unpinned_run_output')
    require(shutil.disk_usage(output.parent).free >= LIMITS['free_reserve_bytes'], 'disk_reserve')
    output.mkdir()
    write_json(output / 'started.json', dict(status='STARTED', started_utc=now(), attempts=1,
               manifest_sha256=sha(config), producer_sha256=sha(__file__), helper_sha256=sha(common.__file__),
               limits=LIMITS, planned_scope=SCOPE))
    terminal = dict(status='STOPPED', attempts=1, planned_scope=SCOPE)
    try:
        with (output / 'stdout.log').open('xb') as stdout, (output / 'stderr.log').open('xb') as stderr:
            remaining = (datetime.fromisoformat(DEADLINE) - datetime.now(timezone.utc)).total_seconds()
            require(remaining > 0, 'deadline_before_worker')
            completed = (runner or subprocess.run)([sys.executable, str(Path(__file__).resolve()),
                '--worker', str(Path(config).resolve()), str(output.resolve() / 'child.json')],
                stdout=stdout, stderr=stderr, timeout=min(60, remaining), check=False)
        terminal['child_returncode'] = completed.returncode
        require(completed.returncode == 0, 'child_nonzero')
        deadline_ok()
        for name in ('stdout.log', 'stderr.log', 'child.json'):
            require((output / name).stat().st_size <= LIMITS['each_output_bytes'], 'child_output_cap')
        report = json.loads((output / 'child.json').read_bytes())
        validate_child(report, cfg)
        terminal.update(status=report['status'], child_sha256=sha(output / 'child.json'),
                        selected_sha256=report['selected_sha256'], scope=SCOPE)
    except Exception as error:
        terminal.update(error_type=type(error).__name__,
                        reason=str(error) if isinstance(error, Stop) else 'details_not_exported')
    finally:
        terminal['ended_utc'] = now()
        write_json(output / 'terminal.json', terminal)
    return terminal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('config')
    parser.add_argument('output')
    args = parser.parse_args()
    if args.worker:
        child(Path(args.config), Path(args.output))
    else:
        result = run(args.config, args.output)
        print(json.dumps(result, allow_nan=False))
        return 0 if result['status'] == 'INPUT_INTEGRITY_PASS' else 1


if __name__ == '__main__':
    sys.exit(main())

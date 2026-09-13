"""One pinned development-file samplingRate read; no DIN/EEG arrays or fitting."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
from datetime import datetime, timezone

MAT = '/home/whwovy/data/mamem_i_v1_20260913/development_first.mat'
MAT_SHA = '57a72c3fde0ff3bc9aaae10721cd7cb450bda63eb5299a96f41704ea696ad10a'
ROLE = '/home/whwovy/data/mamem_i_v1_20260913/development_role.json'
ROLE_SHA = 'dc26f85099ed667208ec7b992430d257d01f2a4aa727b7b98f8a5195f6d9af18'
DEADLINE = '2026-09-13T11:00:00+00:00'
RUN = Path(__file__).resolve().parents[2] / 'docs/reports/mamem_i_scalar_adapter_v1_run'
NUMERIC = {'double', 'single', 'int8', 'int16', 'int32', 'int64',
           'uint8', 'uint16', 'uint32', 'uint64'}
LIMITS = {'address_space_bytes': 2 * 1024**3, 'cpu_seconds': 30,
          'wall_seconds': 60, 'each_output_bytes': 32 * 1024}
SCOPE = {'requested_variables': ['samplingRate'], 'EEG_arrays_requested': False,
         'DIN_arrays_requested': False, 'fits': 0, 'held60': 0,
         'hash_header_may_process_compressed_file_bytes': True,
         'physical_sampling_clock_measured': False}


class Stop(RuntimeError):
    pass


def require(condition, reason):
    if not condition:
        raise Stop(reason)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(str(Path(path).parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def validate_manifest(cfg):
    require(cfg['mat_path'] == MAT and cfg['mat_sha256'] == MAT_SHA
            and cfg['mat_bytes'] == 137357437, 'unpinned_MAT')
    require(cfg['role_path'] == ROLE and cfg['role_sha256'] == ROLE_SHA, 'unpinned_role')
    require(cfg['variables'] == ['samplingRate'] and cfg['descriptor_read'] is False,
            'scalar_only_contract')
    require(cfg['expected_hz'] == 250 and cfg['limits'] == LIMITS, 'changed_budget_or_expected')
    require(cfg['deadline_utc'] == DEADLINE, 'changed_deadline')
    require(cfg['producer_sha256'] == sha(__file__), 'producer_not_frozen')


def inspect(cfg, loader=None, header_loader=None):
    """Test-injectable boundary; production caller validates frozen config first."""
    require(cfg['variables'] == ['samplingRate'] and cfg['descriptor_read'] is False,
            'scalar_only_contract')
    path, role_path = Path(cfg['mat_path']), Path(cfg['role_path'])
    require(path.is_file() and not path.is_symlink(), 'regular_MAT_required')
    require(str(path.resolve()) == cfg['mat_path'], 'canonical_MAT_path_required')
    require(path.stat().st_size == cfg['mat_bytes'], 'MAT_size_mismatch')
    require(role_path.is_file() and not role_path.is_symlink(), 'regular_role_required')
    require(role_path.stat().st_size < 8192, 'role_size_cap')
    require(sha(role_path) == cfg['role_sha256'], 'role_hash_mismatch')
    role = json.loads(role_path.read_bytes())
    require(role['subject'] == 'S001' and role['subject_role'] == 'development_only_all_records',
            'S001_development_role_required')
    require(role['member_selection'] == 'lexicographic_first_MAT_across_both_archives'
            and role['member'] == 'EEG-SSVEP-Part1/S001a.mat', 'selection_mismatch')
    require(role['mat_path'] == cfg['mat_path'] and role['mat_sha256'] == cfg['mat_sha256']
            and role['bytes'] == cfg['mat_bytes'], 'role_binding_mismatch')
    require(sha(path) == cfg['mat_sha256'], 'MAT_hash_mismatch')
    if loader is None or header_loader is None:
        from scipy.io import loadmat, whosmat
        loader = loader or loadmat
        header_loader = header_loader or whosmat
    headers = [(shape, cls) for name, shape, cls in header_loader(path)
               if name == 'samplingRate']
    require(len(headers) == 1, 'samplingRate_missing_or_duplicate')
    shape, cls = headers[0]
    require(tuple(shape) == (1, 1) and cls in NUMERIC, 'unsupported_scalar_header')
    loaded = loader(path, variable_names=['samplingRate'], squeeze_me=False,
                    struct_as_record=True, verify_compressed_data_integrity=True)
    require('samplingRate' in loaded and set(loaded) <= {
        '__header__', '__version__', '__globals__', 'samplingRate'}, 'unexpected_loaded_keys')
    value = loaded['samplingRate']
    require(value.shape == (1, 1) and value.dtype.kind in 'iuf', 'unsupported_scalar_value')
    hz = float(value.item())
    require(math.isfinite(hz) and hz > 0, 'nonfinite_or_nonpositive_samplingRate')
    return {'status': 'MATCH' if hz == cfg['expected_hz'] else 'MISMATCH',
            'stored_samplingRate_hz': hz, 'expected_release_hz': cfg['expected_hz'],
            'mat_sha256': cfg['mat_sha256'], 'subject': 'S001', 'scope': SCOPE}


def worker_claim(config, output):
    require(output == RUN / 'child.json', 'unpinned_worker_output')
    require(not output.exists() and not (RUN / 'terminal.json').exists(), 'worker_already_terminal')
    started = json.loads((RUN / 'started.json').read_bytes())
    require(started['status'] == 'STARTED' and started['manifest_sha256'] == sha(config)
            and started['producer_sha256'] == sha(__file__), 'worker_start_binding_mismatch')
    write_json(RUN / 'worker_claim.json', {'claimed_utc': now(), 'attempts': 1})


def child(config, output):
    worker_claim(config, output)  # Exclusive before any MAT metadata access, including retries.
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = '1'
    for limit, value in ((resource.RLIMIT_AS, LIMITS['address_space_bytes']),
                         (resource.RLIMIT_CPU, LIMITS['cpu_seconds']),
                         (resource.RLIMIT_FSIZE, LIMITS['each_output_bytes'])):
        resource.setrlimit(limit, (value, value))
    report = {'started_utc': now(), 'scope': SCOPE}
    try:
        cfg = json.loads(Path(config).read_bytes())
        validate_manifest(cfg)
        require(datetime.now(timezone.utc) < datetime.fromisoformat(DEADLINE), 'deadline')
        report.update(inspect(cfg))
        require(datetime.now(timezone.utc) < datetime.fromisoformat(DEADLINE), 'deadline_after_read')
    except Exception as error:
        report.update(status='STOPPED', error_type=type(error).__name__,
                      reason=str(error) if isinstance(error, Stop) else 'details_not_exported')
    report['ended_utc'] = now()
    write_json(output, report)


def validate_child(report, cfg):
    require(report.get('scope') == SCOPE, 'child_scope_mismatch')
    require(report.get('status') in ('MATCH', 'MISMATCH'), 'child_stopped_or_invalid')
    hz = report.get('stored_samplingRate_hz')
    require(type(hz) in (int, float) and math.isfinite(hz) and hz > 0, 'invalid_child_scalar')
    require(report['status'] == ('MATCH' if hz == cfg['expected_hz'] else 'MISMATCH'),
            'child_status_value_mismatch')
    require(report.get('mat_sha256') == cfg['mat_sha256'] and report.get('subject') == 'S001',
            'child_binding_mismatch')


def run(config, output, runner=None):
    cfg = json.loads(Path(config).read_bytes())
    validate_manifest(cfg)
    remaining = (datetime.fromisoformat(DEADLINE) - datetime.now(timezone.utc)).total_seconds()
    require(remaining > 0, 'deadline_before_attempt')
    output = Path(output)
    require(output.resolve() == RUN, 'unpinned_run_output')
    output.mkdir()  # Exclusive single-attempt marker, including failed/abandoned runs.
    write_json(output / 'started.json', {'status': 'STARTED', 'started_utc': now(),
               'manifest_sha256': sha(config), 'producer_sha256': sha(__file__),
               'limits': LIMITS, 'attempts': 1, 'scope': SCOPE})
    terminal = {'status': 'STOPPED', 'attempts': 1, 'scope': SCOPE}
    runner = runner or subprocess.run
    try:
        with (output / 'stdout.log').open('xb') as stdout, (output / 'stderr.log').open('xb') as stderr:
            remaining = (datetime.fromisoformat(DEADLINE) - datetime.now(timezone.utc)).total_seconds()
            require(remaining > 0, 'deadline_before_worker')
            completed = runner([sys.executable, str(Path(__file__).resolve()), '--worker',
                                str(Path(config).resolve()), str(output.resolve() / 'child.json')],
                               stdout=stdout, stderr=stderr,
                               timeout=min(LIMITS['wall_seconds'], remaining), check=False)
        terminal['child_returncode'] = completed.returncode
        require(completed.returncode == 0, 'child_nonzero')
        require(datetime.now(timezone.utc) < datetime.fromisoformat(DEADLINE), 'deadline_after_worker')
        for name in ('stdout.log', 'stderr.log', 'child.json'):
            require((output / name).stat().st_size <= LIMITS['each_output_bytes'], 'child_output_cap')
        report = json.loads((output / 'child.json').read_bytes())
        validate_child(report, cfg)
        terminal.update(status=report['status'], stored_samplingRate_hz=report['stored_samplingRate_hz'],
                        child_sha256=sha(output / 'child.json'))
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
        return 0 if result['status'] in ('MATCH', 'MISMATCH') else 1


if __name__ == '__main__':
    sys.exit(main())

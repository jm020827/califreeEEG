"""Generated fixtures/spies only; no real participant values."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.io import savemat

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/analysis/probe_mamem_i_scalar_metadata_v1.py'
spec = importlib.util.spec_from_file_location('mamem_scalar', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def fixture(tmp_path):
    path, role = tmp_path / 'generated.mat', tmp_path / 'role.json'
    savemat(path, {'samplingRate': np.array([[250.]]), 'eeg': np.zeros((2, 3)),
                   'DIN_1': np.zeros((4, 2))}, do_compression=True)
    data = dict(subject='S001', subject_role='development_only_all_records',
                member_selection='lexicographic_first_MAT_across_both_archives',
                member='EEG-SSVEP-Part1/S001a.mat', mat_path=str(path),
                mat_sha256=m.sha(path), bytes=path.stat().st_size)
    role.write_text(json.dumps(data))
    cfg = dict(mat_path=str(path), mat_sha256=m.sha(path), mat_bytes=path.stat().st_size,
               role_path=str(role), role_sha256=m.sha(role), expected_hz=250,
               variables=['samplingRate'], descriptor_read=False)
    return cfg, data


def test_generated_mat_real_scipy_whitelist(fixture):
    assert m.inspect(fixture[0])['stored_samplingRate_hz'] == 250


@pytest.mark.parametrize('hz,status', [(250, 'MATCH'), (256, 'MISMATCH')])
def test_scalar_and_whitelist(fixture, hz, status):
    def loader(path, **kwargs):
        assert kwargs == dict(variable_names=['samplingRate'], squeeze_me=False,
                              struct_as_record=True, verify_compressed_data_integrity=True)
        return {'samplingRate': np.array([[hz]])}
    result = m.inspect(fixture[0], loader, lambda p: [('samplingRate', (1, 1), 'double')])
    assert result['status'] == status
    assert not result['scope']['physical_sampling_clock_measured']


@pytest.mark.parametrize('value', [0, -1, np.nan, np.inf, 1j, '250'])
def test_bad_values(fixture, value):
    with pytest.raises(m.Stop):
        m.inspect(fixture[0], lambda *a, **k: {'samplingRate': np.array([[value]])},
                  lambda p: [('samplingRate', (1, 1), 'double')])


@pytest.mark.parametrize('headers', [[], [('samplingRate', (1, 1), 'cell')],
    [('samplingRate', (2, 1), 'double')], [('samplingRate', (1, 1), 'double')] * 2])
def test_bad_header_before_value_loader(fixture, headers):
    def forbidden(*a, **k):
        pytest.fail('loader invoked before valid header')
    with pytest.raises(m.Stop):
        m.inspect(fixture[0], forbidden, lambda p: headers)


@pytest.mark.parametrize('extra', ['eeg', 'DIN_1'])
def test_extra_materialized_key_rejected(fixture, extra):
    with pytest.raises(m.Stop, match='unexpected_loaded_keys'):
        m.inspect(fixture[0], lambda *a, **k: {'samplingRate': np.array([[250]]), extra: []},
                  lambda p: [('samplingRate', (1, 1), 'double')])


@pytest.mark.parametrize('change', ['size', 'hash', 'role_hash', 'role', 'path', 'descriptor'])
def test_bindings_before_any_loader(fixture, change):
    cfg, data = fixture
    if change == 'size':
        cfg['mat_bytes'] += 1
    elif change == 'hash':
        cfg['mat_sha256'] = data['mat_sha256'] = '0' * 64
    elif change == 'role_hash':
        cfg['role_sha256'] = '0' * 64
    elif change == 'role':
        data['subject_role'] = 'independent_efficacy'
    elif change == 'path':
        data['mat_path'] = 'different.mat'
    else:
        cfg['descriptor_read'] = True
    if change in ('role', 'path', 'hash'):
        Path(cfg['role_path']).write_text(json.dumps(data))
        cfg['role_sha256'] = m.sha(cfg['role_path'])
    def forbidden(*a, **k):
        pytest.fail('any loader invoked before binding checks')
    with pytest.raises(m.Stop):
        m.inspect(cfg, forbidden, forbidden)


@pytest.mark.parametrize('mode', ['timeout', 'nonzero', 'missing', 'stopped', 'bad_value', 'output_cap'])
def test_durable_terminal_and_no_retry(tmp_path, monkeypatch, mode):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'expected_hz': 250, 'mat_sha256': 'generated'}))
    monkeypatch.setattr(m, 'validate_manifest', lambda cfg: None)
    monkeypatch.setattr(m, 'DEADLINE', '2099-01-01T00:00:00+00:00')
    output, calls = tmp_path / 'run', []
    monkeypatch.setattr(m, 'RUN', output)
    def runner(cmd, **kwargs):
        calls.append(cmd)
        assert kwargs['timeout'] <= 60 and not kwargs['check']
        assert (output / 'started.json').exists()
        if mode == 'timeout':
            raise subprocess.TimeoutExpired(cmd, 60)
        if mode not in ('missing', 'nonzero'):
            report = dict(status='STOPPED' if mode == 'stopped' else 'MATCH', scope=m.SCOPE,
                          stored_samplingRate_hz=251 if mode == 'bad_value' else 250,
                          mat_sha256='generated', subject='S001')
            m.write_json(output / 'child.json', report)
        if mode == 'output_cap':
            kwargs['stdout'].write(b'x' * 32769)
        return SimpleNamespace(returncode=-9 if mode == 'nonzero' else 0)
    result = m.run(config, output, runner)
    assert result['status'] == 'STOPPED'
    assert json.loads((output / 'terminal.json').read_text()) == result
    with pytest.raises(FileExistsError):
        m.run(config, output, runner)
    assert len(calls) == 1


def test_manifest_rejects_extra_scope():
    cfg = json.loads((SCRIPT.parents[2] / 'configs/mamem_i_scalar_adapter_v1.json').read_text())
    cfg['producer_sha256'] = m.sha(SCRIPT)
    m.validate_manifest(cfg)
    cfg['variables'] = ['samplingRate', 'DIN_1']
    with pytest.raises(m.Stop, match='scalar_only_contract'):
        m.validate_manifest(cfg)


def test_worker_requires_parent_and_claims_once(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'RUN', tmp_path)
    config = tmp_path / 'config.json'
    config.write_text('{}')
    output = tmp_path / 'child.json'
    with pytest.raises(FileNotFoundError):
        m.worker_claim(config, output)
    m.write_json(tmp_path / 'started.json', dict(status='STARTED', manifest_sha256=m.sha(config),
                                               producer_sha256=m.sha(SCRIPT)))
    m.worker_claim(config, output)
    with pytest.raises(FileExistsError):
        m.worker_claim(config, output)
    m.write_json(tmp_path / 'terminal.json', {'status': 'STOPPED'})
    with pytest.raises(m.Stop, match='worker_already_terminal'):
        m.worker_claim(config, output)


def test_child_enforces_limits_before_inspect(fixture, tmp_path, monkeypatch):
    cfg, _ = fixture
    config, output = tmp_path / 'config.json', tmp_path / 'child.json'
    config.write_text(json.dumps(cfg))
    limits = []
    monkeypatch.setattr(m, 'worker_claim', lambda *a: None)
    monkeypatch.setattr(m, 'validate_manifest', lambda cfg: None)
    monkeypatch.setattr(m, 'DEADLINE', '2099-01-01T00:00:00+00:00')
    monkeypatch.setattr(m.resource, 'setrlimit', lambda *args: limits.append(args))
    def inspect(cfg):
        assert limits == [(m.resource.RLIMIT_AS, (2147483648, 2147483648)),
                          (m.resource.RLIMIT_CPU, (30, 30)),
                          (m.resource.RLIMIT_FSIZE, (32768, 32768))]
        assert all(m.os.environ[k] == '1' for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'])
        return {'status': 'MATCH'}
    monkeypatch.setattr(m, 'inspect', inspect)
    m.child(config, output)
    assert json.loads(output.read_text())['status'] == 'MATCH'


@pytest.mark.parametrize('hz,status', [(250, 'MATCH'), (256, 'MISMATCH')])
def test_parent_success_status_preserved(tmp_path, monkeypatch, hz, status):
    config, output = tmp_path / 'config.json', tmp_path / 'run'
    config.write_text(json.dumps({'expected_hz': 250, 'mat_sha256': 'generated'}))
    monkeypatch.setattr(m, 'validate_manifest', lambda cfg: None)
    monkeypatch.setattr(m, 'DEADLINE', '2099-01-01T00:00:00+00:00')
    monkeypatch.setattr(m, 'RUN', output)
    def runner(cmd, **kwargs):
        m.write_json(output / 'child.json', dict(status=status, scope=m.SCOPE,
                     stored_samplingRate_hz=hz, mat_sha256='generated', subject='S001'))
        return SimpleNamespace(returncode=0)
    result = m.run(config, output, runner)
    assert result['status'] == status and result['stored_samplingRate_hz'] == hz
    assert result['child_sha256'] == m.sha(output / 'child.json')


def test_preparation_deadline_stops_before_worker(tmp_path, monkeypatch):
    config, output = tmp_path / 'config.json', tmp_path / 'run'
    config.write_text('{}')
    monkeypatch.setattr(m, 'validate_manifest', lambda cfg: None)
    monkeypatch.setattr(m, 'DEADLINE', '2099-01-01T00:00:00+00:00')
    monkeypatch.setattr(m, 'RUN', output)
    original_write = m.write_json
    def delayed_write(path, data):
        original_write(path, data)
        if path.name == 'started.json':
            monkeypatch.setattr(m, 'DEADLINE', '2000-01-01T00:00:00+00:00')
    monkeypatch.setattr(m, 'write_json', delayed_write)
    result = m.run(config, output, lambda *a, **k: pytest.fail('late worker launched'))
    assert result['status'] == 'STOPPED' and result['reason'] == 'deadline_before_worker'


def test_late_read_never_promoted(tmp_path, monkeypatch):
    config, output = tmp_path / 'config.json', tmp_path / 'child.json'
    config.write_text('{}')
    monkeypatch.setattr(m, 'worker_claim', lambda *a: None)
    monkeypatch.setattr(m, 'validate_manifest', lambda cfg: None)
    monkeypatch.setattr(m.resource, 'setrlimit', lambda *a: None)
    monkeypatch.setattr(m, 'DEADLINE', '2099-01-01T00:00:00+00:00')
    def late_inspect(cfg):
        monkeypatch.setattr(m, 'DEADLINE', '2000-01-01T00:00:00+00:00')
        return {'status': 'MATCH'}
    monkeypatch.setattr(m, 'inspect', late_inspect)
    m.child(config, output)
    result = json.loads(output.read_text())
    assert result['status'] == 'STOPPED' and result['reason'] == 'deadline_after_read'

"""Generated arrays/files only; real MAMEM cannot enter this suite."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.io import savemat

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/analysis/probe_mamem_i_256row_v1.py'
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location('mamem_256', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def din_array(first=11, last=900):
    din = np.empty((4, 80), dtype=object)
    for index in np.ndindex(din.shape):
        din[index] = np.array([[np.nan]])
    din[3, 0], din[3, 71] = np.array([[first]]), np.array([[last]])
    return din


def eeg_array():
    return np.arange(257)[:, None] + np.arange(1000)[None, :] / 10.


def test_index_containment_inclusive_endpoint():
    window = m.window_from_din(din_array(11, 760), 1000, [4, 80])
    assert window['start0'] == 260 and window['end0'] == 760
    with pytest.raises(m.Stop, match='not_contained'):
        m.window_from_din(din_array(11, 759), 1000, [4, 80])


@pytest.mark.parametrize('first,last', [(0, 900), (-1, 900), (1.5, 900), (np.nan, 900),
    (np.inf, 900), (11, 1001), (900, 11), (11, 11), (700, 1000), ('x', 900), (1j, 900)])
def test_bad_endpoints(first, last):
    with pytest.raises(m.Stop):
        m.window_from_din(din_array(first, last), 1000, [4, 80])


@pytest.mark.parametrize('poison', [np.nan, np.inf, -np.inf, 1e300, -1e300, 7.])
def test_row_and_time_poison_invariant(poison):
    eeg = eeg_array()
    window = m.window_from_din(din_array(), 1000, [4, 80])
    baseline = m.select_copy(eeg, window, [257, 1000])
    report = m.summarize(baseline)
    eeg[256, :] = poison  # label/stim row has no effect, even on QC.
    eeg[:256, :260] = poison
    eeg[:256, 760:] = poison
    selected = m.select_copy(eeg, window, [257, 1000])
    assert np.array_equal(selected, baseline)
    assert m.summarize(selected) == report


def test_order_copy_and_positive_sensitivity():
    eeg = eeg_array()
    window = m.window_from_din(din_array(), 1000, [4, 80])
    selected = m.select_copy(eeg, window, [257, 1000])
    assert selected[0, 0] == eeg[0, 260] and selected[255, -1] == eeg[255, 759]
    assert not np.shares_memory(selected, eeg)
    old_hash = m.summarize(selected)['selected_sha256']
    selected[0, 0] += 1
    assert selected[0, 0] != eeg[0, 260]
    assert m.summarize(selected)['selected_sha256'] != old_hash


def test_selected_nonfinite_and_all_constant():
    selected = np.ones((256, 500)) * 1e300
    result = m.summarize(selected)
    assert result['status'] == 'INPUT_ALL_CONSTANT' and result['constant_channels'] == 256
    selected[0, 0] = -1e300
    assert m.summarize(selected)['constant_channels'] == 255  # no variance overflow.
    selected[0, 0] = np.nan
    with pytest.raises(m.Stop, match='selected_nonfinite'):
        m.summarize(selected)


@pytest.mark.parametrize('kind', ['transpose', 'float32', 'integer', 'short', 'window', 'negative'])
def test_bad_eeg_contract(kind):
    eeg = eeg_array()
    window = m.window_from_din(din_array(), 1000, [4, 80])
    if kind == 'transpose': eeg = eeg.T
    if kind == 'float32': eeg = eeg.astype('f4')
    if kind == 'integer': eeg = eeg.astype('i8')
    if kind == 'short': eeg = eeg[:256]
    if kind == 'window': window['end0'] += 1
    if kind == 'negative': window.update(start0=-1, end0=499)
    with pytest.raises(m.Stop): m.select_copy(eeg, window, [257, 1000])


@pytest.fixture
def fixture(tmp_path):
    path = tmp_path / 'generated.mat'
    eeg = np.zeros((257, 1000))
    eeg[0, :] = np.arange(1000)
    din = din_array()
    savemat(path, {'eeg': eeg, 'DIN_1': din, 'samplingRate': np.array([[123.]])}, do_compression=True)
    cfg = dict(m.PINS, mat_path=str(path), mat_sha256=m.sha(path), mat_bytes=path.stat().st_size,
               eeg_shape=[257, 1000], din_shape=[4, 80])
    role = dict(subject='S001', subject_role='development_only_all_records',
                member='EEG-SSVEP-Part1/S001a.mat',
                member_selection='lexicographic_first_MAT_across_both_archives',
                mat_path=str(path), mat_sha256=cfg['mat_sha256'], bytes=cfg['mat_bytes'])
    prior = dict(subject='S001', mat_sha256=cfg['mat_sha256'], events_in_first_group_prefix=72,
                 group_complete=True, boundary_observed=True)
    sr = dict(subject='S001', mat_sha256=cfg['mat_sha256'], status='MATCH', stored_samplingRate_hz=250)
    for key, data in [('role', role), ('din_report', prior), ('sampling_receipt', sr)]:
        target = tmp_path / (key + '.json')
        target.write_text(json.dumps(data))
        cfg[key + '_path'], cfg[key + '_sha256'] = str(target), m.sha(target)
    return cfg, eeg, din


def test_generated_mat_whitelist_real_scipy(fixture):
    cfg, _, _ = fixture
    result = m.inspect(cfg)
    assert result['status'] == 'INPUT_INTEGRITY_PASS' and result['constant_channels'] == 255
    assert result['window']['nominal_sampling_hz'] == 250  # does not read generated123scalar.


def test_loader_order_and_exact_cells(fixture):
    cfg, eeg, din = fixture
    calls = []
    def loader(path, **kwargs):
        name, = kwargs['variable_names']
        assert kwargs == dict(variable_names=[name], squeeze_me=False, struct_as_record=True,
                              verify_compressed_data_integrity=True)
        calls.append(name)
        return {name: din if name == 'DIN_1' else eeg}
    headers = lambda p: [('eeg', (257, 1000), 'double'), ('DIN_1', (4, 80), 'cell')]
    result = m.inspect(cfg, loader, headers)
    assert calls == ['DIN_1', 'eeg']
    assert result['scope']['DIN_numeric_cells'] == [[3, 0], [3, 71]]


@pytest.mark.parametrize('which', ['extra', 'duplicate', 'invalid_endpoint'])
def test_whitelist_or_header_stops_before_eeg(fixture, which):
    cfg, eeg, din = fixture
    calls = []
    headers = [('eeg', (257, 1000), 'double'), ('DIN_1', (4, 80), 'cell')]
    if which == 'duplicate': headers.append(headers[0])
    if which == 'invalid_endpoint': din[3, 0] = np.array([[0.]])
    def loader(path, **kwargs):
        name, = kwargs['variable_names']
        calls.append(name)
        assert name != 'eeg'
        return {'DIN_1': din, **({'eeg': eeg} if which == 'extra' else {})}
    with pytest.raises(m.Stop): m.inspect(cfg, loader, lambda p: headers)
    assert calls in ([], ['DIN_1'])


@pytest.mark.parametrize('field', ['mat_bytes', 'mat_sha256', 'role', 'din_report', 'sampling_receipt'])
def test_bad_bindings_before_load(fixture, field):
    cfg, _, _ = fixture
    if field == 'mat_bytes': cfg[field] += 1
    elif field == 'mat_sha256': cfg[field] = '0' * 64
    else: cfg[field + '_sha256'] = '0' * 64
    def forbidden(*a, **k): pytest.fail('loader called before valid binding')
    with pytest.raises(m.Stop): m.inspect(cfg, forbidden, forbidden)


@pytest.mark.parametrize('field', ['role', 'din_report', 'sampling_receipt'])
def test_valid_hash_wrong_semantics_rejected(fixture, field):
    cfg, _, _ = fixture
    target = Path(cfg[field + '_path'])
    data = json.loads(target.read_text())
    data['subject'] = 'S002'
    target.write_text(json.dumps(data))
    cfg[field + '_sha256'] = m.sha(target)
    with pytest.raises(m.Stop): m.validate_input(cfg)


@pytest.mark.parametrize('mode', ['pass', 'allconstant', 'timeout', 'nonzero', 'missing', 'bad_status', 'outputcap'])
def test_parent_receipts_and_single_attempt(tmp_path, monkeypatch, mode):
    config, output = tmp_path / 'config.json', tmp_path / 'run'
    cfg = dict(mat_sha256='generated', eeg_shape=[257, 1000])
    config.write_text(json.dumps(cfg))
    monkeypatch.setattr(m, 'validate_manifest', lambda c: None)
    monkeypatch.setattr(m, 'DEADLINE', '2099-01-01T00:00:00+00:00')
    monkeypatch.setattr(m, 'RUN', output)
    calls = []
    def runner(cmd, **kwargs):
        calls.append(cmd)
        assert (output / 'started.json').exists() and kwargs['timeout'] <= 60
        if mode == 'timeout': raise subprocess.TimeoutExpired(cmd, 60)
        if mode not in ('missing', 'nonzero'):
            count = 256 if mode == 'allconstant' else 0
            report = dict(scope=m.SCOPE, subject='S001', mat_sha256='generated',
                selected_shape=[256, 500], all_finite=True, constant_channels=count,
                status='INPUT_ALL_CONSTANT' if count == 256 else 'INPUT_INTEGRITY_PASS',
                max_abs_stored_units=1., selected_sha256='a' * 64,
                SSVEP_presence_verified=False, voltage_unit='unverified_stored_units',
                hash_encoding='little-endian float64 C-order',
                window=m.window_from_din(din_array(), 1000, [4, 80]))
            if mode == 'bad_status': report['constant_channels'] = 256
            m.write_json(output / 'child.json', report)
        if mode == 'outputcap': kwargs['stderr'].write(b'x' * 32769)
        return SimpleNamespace(returncode=-9 if mode == 'nonzero' else 0)
    result = m.run(config, output, runner)
    expected = {'pass': 'INPUT_INTEGRITY_PASS', 'allconstant': 'INPUT_ALL_CONSTANT'}.get(mode, 'STOPPED')
    assert result['status'] == expected
    assert json.loads((output / 'terminal.json').read_text()) == result
    with pytest.raises(FileExistsError): m.run(config, output, runner)
    assert len(calls) == 1


def test_worker_claim_before_access_and_manifest_pin(tmp_path, monkeypatch):
    cfg = json.loads((SCRIPT.parents[2] / 'configs/mamem_i_256row_adapter_v1.json').read_text())
    cfg['producer_sha256'] = m.sha(SCRIPT)
    m.validate_manifest(cfg)
    cfg['rows'] = [0, 257]
    with pytest.raises(m.Stop, match='changed_frozen'): m.validate_manifest(cfg)
    config, output = tmp_path / 'config.json', tmp_path / 'child.json'
    config.write_text('{}')
    monkeypatch.setattr(m, 'RUN', tmp_path)
    with pytest.raises(FileNotFoundError): m.worker_claim(config, output)
    m.write_json(tmp_path / 'started.json', dict(status='STARTED', manifest_sha256=m.sha(config),
                 producer_sha256=m.sha(SCRIPT), helper_sha256=m.sha(m.common.__file__)))
    m.worker_claim(config, output)
    with pytest.raises(FileExistsError): m.worker_claim(config, output)


def test_only_two_DIN_cells_accessed():
    class Tracked(np.ndarray):
        def __getitem__(self, key):
            assert key in ((3, 0), (3, 71))
            calls.append(key)
            return super().__getitem__(key)
    calls = []
    din = din_array().view(Tracked)
    m.window_from_din(din, 1000, [4, 80])
    assert calls == [(3, 0), (3, 71)]


def test_decode_scope_trace(fixture):
    access = []
    m.inspect(fixture[0], access=access)
    assert access == ['DIN_1_load_requested', 'DIN_1_decoded_whitelist_validated',
                      'two_sample_cells_and_window_validated', 'eeg_load_requested',
                      'eeg_decoded_whitelist_validated', 'selected_256x500_copied_before_QC']


def test_child_limits_before_load_and_late_stop(tmp_path, monkeypatch):
    config, output = tmp_path / 'config.json', tmp_path / 'child.json'
    config.write_text('{}')
    monkeypatch.setattr(m, 'worker_claim', lambda *a: None)
    monkeypatch.setattr(m, 'validate_manifest', lambda *a: None)
    monkeypatch.setattr(m, 'DEADLINE', '2099-01-01T00:00:00+00:00')
    limits = []
    monkeypatch.setattr(m.resource, 'setrlimit', lambda *a: limits.append(a))
    def inspect(cfg, access):
        assert limits == [(m.resource.RLIMIT_AS, (2147483648, 2147483648)),
                          (m.resource.RLIMIT_CPU, (30, 30)),
                          (m.resource.RLIMIT_FSIZE, (32768, 32768))]
        assert all(m.os.environ[k] == '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'))
        access.append('generated_access')
        monkeypatch.setattr(m, 'DEADLINE', '2000-01-01T00:00:00+00:00')
        return {'status': 'INPUT_INTEGRITY_PASS'}
    monkeypatch.setattr(m, 'inspect', inspect)
    m.child(config, output)
    result = json.loads(output.read_text())
    assert result['status'] == 'STOPPED' and result['reason'] == 'deadline'
    assert result['access_stages'] == ['generated_access']


def test_preparation_delay_stops_before_worker(tmp_path, monkeypatch):
    config, output = tmp_path / 'config.json', tmp_path / 'run'
    config.write_text('{}')
    monkeypatch.setattr(m, 'validate_manifest', lambda *a: None)
    monkeypatch.setattr(m, 'DEADLINE', '2099-01-01T00:00:00+00:00')
    monkeypatch.setattr(m, 'RUN', output)
    original_write = m.write_json
    def delayed_write(path, data):
        original_write(path, data)
        if path.name == 'started.json':
            monkeypatch.setattr(m, 'DEADLINE', '2000-01-01T00:00:00+00:00')
    monkeypatch.setattr(m, 'write_json', delayed_write)
    result = m.run(config, output, lambda *a, **k: pytest.fail('late launch'))
    assert result['status'] == 'STOPPED' and result['reason'] == 'deadline_before_worker'


@pytest.mark.parametrize('mode', ['disk', 'path'])
def test_reserve_and_fixed_output_before_attempt(tmp_path, monkeypatch, mode):
    config, output = tmp_path / 'config.json', tmp_path / 'run'
    config.write_text('{}')
    monkeypatch.setattr(m, 'validate_manifest', lambda *a: None)
    monkeypatch.setattr(m, 'DEADLINE', '2099-01-01T00:00:00+00:00')
    monkeypatch.setattr(m, 'RUN', output if mode == 'disk' else tmp_path / 'other')
    if mode == 'disk':
        monkeypatch.setattr(m.shutil, 'disk_usage', lambda p: SimpleNamespace(free=1))
    with pytest.raises(m.Stop): m.run(config, output)
    assert not output.exists()


def test_deadline_after_DIN_never_loads_EEG(fixture, monkeypatch):
    cfg, eeg, din = fixture
    calls = []
    original_window = m.window_from_din
    def late_window(*args):
        result = original_window(*args)
        monkeypatch.setattr(m, 'DEADLINE', '2000-01-01T00:00:00+00:00')
        return result
    monkeypatch.setattr(m, 'window_from_din', late_window)
    def loader(path, **kwargs):
        name, = kwargs['variable_names']
        calls.append(name)
        assert name == 'DIN_1'
        return {'DIN_1': din}
    with pytest.raises(m.Stop, match='deadline'):
        m.inspect(cfg, loader, lambda p: [('eeg', (257, 1000), 'double'), ('DIN_1', (4, 80), 'cell')])
    assert calls == ['DIN_1']

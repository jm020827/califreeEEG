from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from cfeg.analysis import metadata_calibration_v2 as analysis
from cfeg.data import metadata_calibration_v2_requests as requests


@pytest.fixture(autouse=True)
def _exercise_retired_algorithms_without_operational_authority(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "deny_v2_terminal_operational_action", lambda _action: None)
    monkeypatch.setattr(requests, "deny_v2_terminal_operational_action", lambda _action: None)


@dataclass(frozen=True)
class FakeBundle:
    root: Path
    contract: requests.V2ExternalContract
    development: requests.ValidatedExternalAsset
    independent: requests.ValidatedExternalAsset
    secret: bytes


@pytest.fixture(scope="module")
def fake_bundle(tmp_path_factory: pytest.TempPathFactory) -> FakeBundle:
    base_contract = requests.load_v2_external_contract()
    root = tmp_path_factory.mktemp("v2-external") / "beta_v1"
    root.mkdir()
    labels = np.tile(np.arange(40, dtype=np.int64), 4)
    h5_labels = labels.copy()
    # Deliberately poison every query y value. Prediction must never index it.
    h5_labels[120:] = 99
    signal = np.zeros((160, 64, 400), dtype=np.float32)
    time = np.arange(400, dtype=np.float32) / 200.0
    for row in range(160):
        for channel_index, position in enumerate(requests.V2_CANONICAL_CHANNEL_POSITIONS):
            signal[row, position] = (
                np.sin(2.0 * np.pi * (8.0 + 0.2 * (row % 40)) * time + channel_index * 0.01)
                + row * 1.0e-5
            )
    mask = np.ones((160, 64), dtype=bool)
    with h5py.File(root / "signals.h5", "w") as handle:
        handle.create_dataset("x", data=signal, compression="gzip", compression_opts=1)
        handle.create_dataset("channel_mask", data=mask, compression="gzip")
        handle.create_dataset("y", data=h5_labels, compression="gzip")

    rows: list[dict[str, object]] = []
    index = 0
    channel_ids = np.arange(1, 65, dtype=np.int64)
    for block in range(1, 5):
        for label in range(40):
            rows.append(
                {
                    "sample_id": f"fake-{index:03d}",
                    "h5_index": index,
                    "dataset_id": "beta",
                    "subject_id": "sub001",
                    "session_id": "session01",
                    "run_id": f"block{block:02d}",
                    "label": label,
                    "stimulus_frequency_hz": 8.0 + 0.2 * label,
                    "sfreq_processed": 200.0,
                    "n_channels_used": 64,
                    "canonical_channel_ids": channel_ids,
                }
            )
            index += 1
    pd.DataFrame(rows).to_parquet(root / "manifest.parquet", index=False)
    class_map = {
        str(label): {"label": label, "stimulus_frequency_hz": 8.0 + 0.2 * label}
        for label in range(40)
    }
    (root / "class_map.json").write_text(json.dumps(class_map, sort_keys=False), encoding="utf-8")
    (root / "asset_info.json").write_text(
        json.dumps(
            {
                "dataset_id": "beta",
                "canonical_class_frequencies": list(requests.V2_CODEBOOK_HZ),
            }
        ),
        encoding="utf-8",
    )

    original = requests._DATASET_SPECS["beta_v1"]
    requests._DATASET_SPECS["beta_v1"] = {
        **original,
        "n_subjects": 1,
        "manifest_sha256": requests._sha256_file(root / "manifest.parquet"),
        "signals_sha256": requests._sha256_file(root / "signals.h5"),
        "excluded_exposed": (),
        "development_count": 1,
        "independent_gate_count": 1,
    }
    allocation = {
        "allocation": {
            "beta_v1": {
                "excluded_exposed": [],
                "development": ["sub001"],
                "independent_gate": ["sub001"],
            }
        }
    }
    contract = requests.V2ExternalContract(
        plan_sha256=base_contract.plan_sha256,
        plan_content_sha256=base_contract.plan_content_sha256,
        allocation_file_sha256=base_contract.allocation_file_sha256,
        allocation_sha256=base_contract.allocation_sha256,
        plan=base_contract.plan,
        allocation=allocation,
        filterbank=base_contract.filterbank,
        asset_bindings={
            "beta_v1": (
                requests._DATASET_SPECS["beta_v1"]["manifest_sha256"],
                requests._DATASET_SPECS["beta_v1"]["signals_sha256"],
            )
        },
    )
    try:
        development = requests.validate_external_asset(
            root,
            dataset_id="beta_v1",
            phase="development",
            contract=contract,
        )
        selection_payload = _selection_receipt_for_contract(
            contract,
            requests.candidate_grid(contract)[0],
        )
        selection_path = root.parent / "selection.json"
        requests.write_json_exclusive(selection_path, selection_payload)
        selection_authorization = requests.validate_independent_selection_receipt(
            selection_path,
            contract=contract,
        )
        independent = requests.validate_external_asset(
            root,
            dataset_id="beta_v1",
            phase="independent_gate",
            contract=contract,
            selection_authorization=selection_authorization,
        )
    finally:
        requests._DATASET_SPECS["beta_v1"] = original
    yield FakeBundle(root, contract, development, independent, b"s" * 32)


def _fake_fbcca(
    signals: np.ndarray,
    *,
    sfreq: float,
    filterbank: object,
    workers: int,
) -> np.ndarray:
    del sfreq, filterbank, workers
    base = np.linspace(0.0, 1.0, 40, dtype=np.float64)
    return np.stack([np.roll(base, index % 40) for index in range(len(signals))])


def _score_cache(
    fake_bundle: FakeBundle,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    independent: bool = False,
) -> requests.FBCCAScoreCache:
    monkeypatch.setattr(requests, "_map_fbcca", _fake_fbcca)
    asset = fake_bundle.independent if independent else fake_bundle.development
    return requests.build_fbcca_score_cache(
        asset,
        participant=asset.participant("sub001"),
        contract=fake_bundle.contract,
        token_secret=fake_bundle.secret,
        cache_dir=tmp_path / "cache",
        workers=1,
    )


def _selection_receipt(
    fake_bundle: FakeBundle,
    candidate: requests.V2CandidateSpec,
) -> dict[str, object]:
    return _selection_receipt_for_contract(fake_bundle.contract, candidate)


def _selection_receipt_for_contract(
    contract: requests.V2ExternalContract,
    candidate: requests.V2CandidateSpec,
) -> dict[str, object]:
    grid = requests.candidate_grid(contract)
    payload: dict[str, object] = {
        "schema": "cfeg.metadata-calibration-v2-development-selection-completion.v1",
        "candidate_id": requests.V2_CANDIDATE_ID,
        "phase": "development_selection",
        "status": "selected_candidate_frozen",
        "plan_sha256": contract.plan_sha256,
        "plan_content_sha256": contract.plan_content_sha256,
        "allocation_sha256": contract.allocation_sha256,
        "allocation_file_sha256": contract.allocation_file_sha256,
        "candidate_grid_sha256": requests._canonical_json_sha256(
            {"candidates": [value.as_dict() for value in grid]}
        ),
        "participant_delta_sha256": "d" * 64,
        "criterion_order": list(requests._SELECTION_CRITERION_ORDER),
        "candidate_summaries": [{} for _ in range(12)],
        "selected_candidate": candidate.as_dict(),
        "k1_global_enabled": True,
        "candidate_and_parameters_immutable": True,
        "n_candidates": 12,
        "n_participants": 42,
        "independent_gate_authorized": True,
        "held_access_authorized": False,
    }
    payload["completion_receipt_sha256"] = requests._canonical_json_sha256(payload)
    return payload


def _synthetic_bound_score_cache(
    contract: requests.V2ExternalContract,
    tmp_path: Path,
    *,
    phase: str,
    subject_id: str | None = None,
) -> requests.FBCCAScoreCache:
    subject = subject_id or contract.subjects("beta_v1", phase)[0]  # type: ignore[arg-type]
    query_tokens = tuple(
        "qv2_"
        + requests._canonical_json_sha256(
            {"synthetic_query": index, "phase": phase, "subject": subject}
        )
        for index in range(40)
    )
    metadata = {
        "schema": requests.V2_SCORE_CACHE_SCHEMA,
        "plan_sha256": contract.plan_sha256,
        "allocation_sha256": contract.allocation_sha256,
        "dataset_id": "beta_v1",
        "phase": phase,
        "subject_id": subject,
        "asset_manifest_sha256": contract.asset_hashes("beta_v1")[0],
        "asset_signals_sha256": contract.asset_hashes("beta_v1")[1],
        "preprocessing_sha256": "1" * 64,
        "filterbank_sha256": requests.V2_FILTERBANK_SHA256,
        "score_schema": "strict_fbcca_bound_filterbank_raw_scores_v1",
        "codebook_hz": requests.V2_CODEBOOK_HZ,
        "channel_names": requests.V2_CHANNEL_NAMES,
        "support_partition_sha256": "2" * 64,
        "query_partition_sha256": "3" * 64,
        "query_token_commitment_sha256": requests._canonical_json_sha256(
            {"query_tokens": query_tokens}
        ),
        "query_labels_loaded": False,
        "hdf5_y_loaded_for_query": False,
    }
    values = {
        "support_scores": np.tile(np.linspace(0.0, 1.0, 40), (120, 1)),
        "support_labels": np.tile(np.arange(40), 3),
        "support_blocks": np.repeat(np.arange(1, 4), 40),
        "query_scores": np.tile(np.linspace(0.0, 1.0, 40), (40, 1)),
        "query_tokens": np.asarray(query_tokens, dtype="U68"),
    }
    cache_path = tmp_path / phase / "fbcca" / f"{requests._canonical_json_sha256(metadata)}.npz"
    loaded = requests._load_or_create_npz(cache_path, metadata=metadata, compute=lambda: values)
    return requests._score_cache_from_values(cache_path, metadata, loaded)


def _runner_development_frame(
    contract: analysis.V2ContractBinding,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in analysis.v2_candidate_grid(contract):
        for dataset_id in analysis.V2_EXTERNAL_DATASETS:
            asset = contract.asset_binding(dataset_id)
            for subject_id in contract.subject_ids(dataset_id, "development"):
                a0 = {0: 0.50, 1: 0.50, 3: 0.50}
                aq = {0: 0.50, 1: 0.52, 3: 0.54}
                a0_eauc = sum(analysis.V2_EAUC_WEIGHTS[k] * a0[k] for k in analysis.V2_BUDGETS)
                aq_eauc = sum(analysis.V2_EAUC_WEIGHTS[k] * aq[k] for k in analysis.V2_BUDGETS)
                rows.append(
                    {
                        "candidate_id": analysis.V2_CANDIDATE_ID,
                        "plan_sha256": contract.plan_sha256,
                        "allocation_sha256": contract.allocation_sha256,
                        **asset,
                        "dataset_id": dataset_id,
                        "cohort": "development",
                        **candidate.as_dict(),
                        "subject_id": subject_id,
                        **{f"a0_k{k}": a0[k] for k in analysis.V2_BUDGETS},
                        **{f"aq_k{k}": aq[k] for k in analysis.V2_BUDGETS},
                        **{f"delta_k{k}": aq[k] - a0[k] for k in analysis.V2_BUDGETS},
                        "a0_eauc": a0_eauc,
                        "aq_eauc": aq_eauc,
                        "eauc_delta": aq_eauc - a0_eauc,
                    }
                )
    return pd.DataFrame.from_records(rows)


def test_frozen_contract_replays_exact_allocation_and_candidate_grid() -> None:
    contract = requests.load_v2_external_contract()
    assert contract.allocation_sha256 == requests.V2_ALLOCATION_SHA256
    assert contract.plan["method_revision"] == "r6_pre_outcome_explicit_filterbank_weights"
    assert [len(contract.subjects(name, "development")) for name in requests._DATASET_SPECS] == [
        23,
        19,
    ]
    assert len(requests.candidate_grid(contract)) == 12


def test_r6_runtime_filterbank_weights_are_bit_exact() -> None:
    contract = requests.load_v2_external_contract()
    parameters = requests.resolve_filterbank_parameters(contract.filterbank, 200.0)
    observed = np.asarray(parameters["weights"], dtype=np.float64)
    expected = np.asarray(requests.V2_SUBBAND_WEIGHTS, dtype=np.float64)
    assert np.array_equal(observed, expected)
    assert observed[4].hex() == "0x1.88f5406f833c4p-2"


def test_inventory_is_exact_disjoint_and_query_blind(fake_bundle: FakeBundle) -> None:
    asset = fake_bundle.development
    participant = asset.participant("sub001")
    assert len(participant.support_rows) == 120
    assert len(participant.query_h5_indices) == 40
    assert {row.h5_index for row in participant.support_rows}.isdisjoint(
        participant.query_h5_indices
    )
    for block in (1, 2, 3):
        assert sorted(row.label for row in participant.support_rows if row.block == block) == list(
            range(40)
        )
    receipt = requests.build_inventory_receipt(asset, contract=fake_bundle.contract)
    assert receipt["support_labels_loaded"] is True
    assert receipt["query_labels_loaded"] is False
    assert receipt["query_hdf5_y_loaded"] is False
    assert receipt["n_support_rows"] == 120
    assert receipt["n_query_rows"] == 40


@pytest.mark.parametrize(
    ("dataset_id", "phase"),
    [("wang_v1", "development"), ("beta_v1", "held")],
)
def test_dataset_and_phase_boundary_fail_closed(
    fake_bundle: FakeBundle,
    dataset_id: str,
    phase: str,
) -> None:
    with pytest.raises(PermissionError):
        requests.validate_external_asset(
            fake_bundle.root,
            dataset_id=dataset_id,
            phase=phase,  # type: ignore[arg-type]
            contract=fake_bundle.contract,
        )


def test_independent_inventory_requires_selection_before_asset_open(
    fake_bundle: FakeBundle,
) -> None:
    with pytest.raises(PermissionError, match="typed development authorization"):
        requests.validate_external_asset(
            fake_bundle.root,
            dataset_id="beta_v1",
            phase="independent_gate",
            contract=fake_bundle.contract,
        )


def test_asset_hash_drift_fails_before_inventory(
    fake_bundle: FakeBundle,
) -> None:
    drifted = replace(
        fake_bundle.contract,
        asset_bindings={"beta_v1": (fake_bundle.development.manifest_sha256, "0" * 64)},
    )
    with pytest.raises(ValueError, match="signals SHA-256"):
        requests.validate_external_asset(
            fake_bundle.root,
            dataset_id="beta_v1",
            phase="development",
            contract=drifted,
        )


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/source39/request.json",
        "/tmp/wearable_v3/cache",
        "/tmp/held60/inventory.json",
        "/tmp/exposed/results.json",
    ],
)
def test_forbidden_path_aliases_are_rejected(path: str) -> None:
    with pytest.raises(PermissionError):
        requests.write_json_exclusive(path, {})


def test_query_tokens_are_reproducible_opaque_and_secret_bound(
    fake_bundle: FakeBundle,
) -> None:
    participant = fake_bundle.development.participant("sub001")
    first = requests.deterministic_query_tokens(
        fake_bundle.secret, asset=fake_bundle.development, participant=participant
    )
    second = requests.deterministic_query_tokens(
        fake_bundle.secret, asset=fake_bundle.development, participant=participant
    )
    changed = requests.deterministic_query_tokens(
        b"x" * 32, asset=fake_bundle.development, participant=participant
    )
    assert first == second
    assert first != changed
    assert len(set(first)) == 40
    assert all(value.startswith("qv2_") and "sub001" not in value for value in first)


def test_fbcca_cache_never_reads_poisoned_query_y_and_is_reproducible(
    fake_bundle: FakeBundle,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = 0

    def counted(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return _fake_fbcca(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(requests, "_map_fbcca", counted)
    asset = fake_bundle.development
    participant = asset.participant("sub001")
    first = requests.build_fbcca_score_cache(
        asset,
        participant=participant,
        contract=fake_bundle.contract,
        token_secret=fake_bundle.secret,
        cache_dir=tmp_path / "cache",
    )
    second = requests.build_fbcca_score_cache(
        asset,
        participant=participant,
        contract=fake_bundle.contract,
        token_secret=fake_bundle.secret,
        cache_dir=tmp_path / "cache",
    )
    with h5py.File(fake_bundle.root / "signals.h5", "r") as handle:
        assert np.equal(handle["y"][120:], 99).all()
    assert calls == 1
    assert first.path == second.path
    assert np.array_equal(first.query_scores, second.query_scores)
    assert stat.S_IMODE(first.path.stat().st_mode) == 0o400


def test_multiprocess_fbcca_is_ordered_and_restores_thread_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = requests.load_v2_external_contract()
    rng = np.random.default_rng(19)
    signals = rng.normal(size=(2, 8, 400))
    expected = requests._map_fbcca(
        signals,
        sfreq=200.0,
        filterbank=contract.filterbank,
        workers=1,
    )
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    observed = requests._map_fbcca(
        signals,
        sfreq=200.0,
        filterbank=contract.filterbank,
        workers=2,
    )
    assert np.array_equal(observed, expected)
    assert os.environ["OPENBLAS_NUM_THREADS"] == "7"


def test_partial_cache_and_metadata_collision_fail_closed(tmp_path: Path) -> None:
    partial = tmp_path / "partial.npz"
    partial.write_bytes(b"not-a-complete-npz")
    with pytest.raises(ValueError, match="partial or unreadable"):
        requests._load_or_create_npz(
            partial,
            metadata={"schema": "expected"},
            compute=lambda: {"x": np.zeros(1)},
        )

    collision = tmp_path / "collision.npz"
    values = {"x": np.arange(3)}
    stored = {
        "schema": "wrong",
        "payload_sha256": requests._array_mapping_sha256(values),
    }
    with collision.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(json.dumps(stored, sort_keys=True, separators=(",", ":"))),
            **values,
        )
    with pytest.raises(FileExistsError, match="collision"):
        requests._load_or_create_npz(
            collision,
            metadata={"schema": "expected"},
            compute=lambda: {"x": np.zeros(1)},
        )


def test_k1_requires_explicit_gate_and_request_has_no_query_identity_or_outcome(
    fake_bundle: FakeBundle,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = _score_cache(fake_bundle, monkeypatch, tmp_path)
    candidate = requests.candidate_grid(fake_bundle.contract)[0]
    active = requests.build_external_score_request(
        score_cache=cache,
        candidate=candidate,
        budget=1,
        contract=fake_bundle.contract,
        gate_enabled=True,
    )
    fallback = requests.build_external_score_request(
        score_cache=cache,
        candidate=candidate,
        budget=1,
        contract=fake_bundle.contract,
        gate_enabled=None,
    )
    assert active["schema"] == requests.V2_SCORE_REQUEST_SCHEMA
    assert active["development_selection_receipt"] is None
    assert len(active["support_labels"]) == 40
    assert fallback["support_labels"] is None
    assert fallback["support_fbcca_scores"] is None
    assert fallback["gate_enabled"] is None
    forbidden = {
        "query_labels",
        "query_sample_ids",
        "query_h5_indices",
        "sample_id",
        "h5_index",
        "trial_id",
        "source_file",
    }
    assert forbidden.isdisjoint(active)
    assert all(value.startswith("qv2_") for value in active["query_tokens"])


def test_development_request_is_consumed_by_final_runner_without_schema_translation(
    tmp_path: Path,
) -> None:
    producer_contract = requests.load_v2_external_contract()
    runner_contract = analysis.validate_v2_plan_and_allocation()
    cache = _synthetic_bound_score_cache(producer_contract, tmp_path, phase="development")
    candidate = requests.candidate_grid(producer_contract)[0]
    request = requests.build_external_score_request(
        score_cache=cache,
        candidate=candidate,
        budget=1,
        contract=producer_contract,
        gate_enabled=True,
    )
    receipt = analysis.score_v2_external_request(request, contract=runner_contract)
    assert receipt["status"] == "complete_query_outcome_free"
    assert receipt["development_selection_receipt_sha256"] is None
    assert receipt["operator_schema"] == "cfeg.metadata-calibration-v2-safe-operator.v3"
    assert receipt["query_outcomes_loaded"] is False


def test_independent_request_and_embedded_receipt_are_consumed_by_final_runner(
    tmp_path: Path,
) -> None:
    producer_contract = requests.load_v2_external_contract()
    runner_contract = analysis.validate_v2_plan_and_allocation()
    selection = analysis.select_v2_development_candidate(
        _runner_development_frame(runner_contract),
        contract=runner_contract,
        n_resamples=16,
    )
    selection_path = tmp_path / "selection.json"
    requests.write_json_exclusive(selection_path, selection)
    authorization = requests.validate_independent_selection_receipt(
        selection_path, contract=producer_contract
    )
    cache = _synthetic_bound_score_cache(
        producer_contract,
        tmp_path,
        phase="independent_gate",
    )
    request = requests.build_external_score_request(
        score_cache=cache,
        candidate=authorization.candidate,
        budget=1,
        contract=producer_contract,
        gate_enabled=True,
        selection_authorization=authorization,
    )
    receipt = analysis.score_v2_external_request(request, contract=runner_contract)
    assert receipt["development_selection_receipt_sha256"] == selection["completion_receipt_sha256"]
    assert receipt["status"] == "complete_query_outcome_free"
    assert receipt["held_access_authorized"] is False


def test_in_memory_score_cache_tampering_is_rejected(
    fake_bundle: FakeBundle,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = _score_cache(fake_bundle, monkeypatch, tmp_path)
    altered = cache.query_scores.copy()
    altered[0, 0] += 1.0
    forged = requests.FBCCAScoreCache(
        cache.metadata,
        cache.support_scores,
        cache.support_labels,
        cache.support_blocks,
        altered,
        cache.query_tokens,
        cache.path,
    )
    with pytest.raises(ValueError, match="[Ii]n-memory"):
        requests.build_external_score_request(
            score_cache=forged,
            candidate=requests.candidate_grid(fake_bundle.contract)[0],
            budget=1,
            contract=fake_bundle.contract,
            gate_enabled=True,
        )


def test_p2_uses_exact_seven_subbands_and_typed_provenance(
    fake_bundle: FakeBundle,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    score_cache = _score_cache(fake_bundle, monkeypatch, tmp_path)

    def fake_filterbank(
        value: np.ndarray, *, sfreq: float, filterbank: object
    ) -> tuple[np.ndarray, dict[str, object]]:
        del sfreq, filterbank
        return np.repeat(value[None], 7, axis=0), {"weights": list(requests.V2_SUBBAND_WEIGHTS)}

    monkeypatch.setattr(requests, "apply_filterbank", fake_filterbank)
    monkeypatch.setattr(
        requests,
        "template_residual_scores_from_subbands",
        lambda query, support, labels, weights: np.add.outer(
            np.arange(query.shape[1]), np.arange(40)
        ).astype(float),
    )
    asset = fake_bundle.development
    subbands = requests.build_subband_cache(
        asset,
        participant=asset.participant("sub001"),
        contract=fake_bundle.contract,
        token_secret=fake_bundle.secret,
        cache_dir=tmp_path / "cache",
    )
    template = requests.build_template_score_cache(subbands, budget=1, cache_dir=tmp_path / "cache")
    candidate = next(
        value
        for value in requests.candidate_grid(fake_bundle.contract)
        if value.operator == "filterbank_target_template_residual"
    )
    request = requests.build_external_score_request(
        score_cache=score_cache,
        candidate=candidate,
        budget=1,
        contract=fake_bundle.contract,
        gate_enabled=True,
        template_cache=template,
    )
    assert subbands.support_subbands.shape == (7, 120, 8, 400)
    assert request["template_subband_weights"] is None
    assert request["template_query_class_scores"] == template.query_class_scores.tolist()
    assert request["template_score_provenance"]["filterbank_sha256"] == (
        requests.V2_FILTERBANK_SHA256
    )


def test_r5_prequential_prefix_receipts_form_exact_chain(
    fake_bundle: FakeBundle,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    score_cache = _score_cache(fake_bundle, monkeypatch, tmp_path)
    candidate = requests.candidate_grid(fake_bundle.contract)[0]
    folds: list[object] = []

    def fake_operator(**kwargs: object) -> SimpleNamespace:
        fold = kwargs["fold_provenance"]
        folds.append(fold)
        depth = fold.evaluation_block - 1
        assert np.asarray(kwargs["support_fbcca_scores"]).shape == (depth * 40, 40)
        assert "support_depth" not in kwargs
        assert "gate_enabled" not in kwargs
        probabilities = np.full((40, 40), 1.0 / 40.0)
        return SimpleNamespace(
            base_probabilities=probabilities,
            fused_probabilities=probabilities,
            prequential_fold_provenance=fold,
        )

    assert requests._derive_prequential_gate_r5(
        fake_operator,
        score_cache=score_cache,
        candidate=candidate,
        contract=fake_bundle.contract,
    )
    assert [fold.evaluation_block for fold in folds] == [2, 3]
    assert folds[1].fit_block_partition_sha256s == (
        *folds[0].fit_block_partition_sha256s,
        folds[0].evaluation_partition_sha256,
    )


def test_independent_gate_requires_read_only_immutable_selection(
    fake_bundle: FakeBundle,
    tmp_path: Path,
) -> None:
    candidate = requests.candidate_grid(fake_bundle.contract)[0]
    receipt = _selection_receipt(fake_bundle, candidate)
    mutable = tmp_path / "mutable-selection.json"
    mutable.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(PermissionError, match="read-only"):
        requests.validate_independent_selection_receipt(mutable, contract=fake_bundle.contract)
    immutable = tmp_path / "selection.json"
    requests.write_json_exclusive(immutable, receipt)
    authorization = requests.validate_independent_selection_receipt(
        immutable, contract=fake_bundle.contract
    )
    assert authorization.candidate == candidate
    assert requests.resolve_phase_candidates(
        phase="independent_gate",
        contract=fake_bundle.contract,
        selection_receipt=immutable,
    ) == (candidate,)
    with pytest.raises(PermissionError):
        requests.resolve_phase_candidates(
            phase="independent_gate",
            contract=fake_bundle.contract,
        )


def test_independent_request_embeds_same_selected_candidate_receipt(
    fake_bundle: FakeBundle,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = requests.candidate_grid(fake_bundle.contract)[0]
    receipt = _selection_receipt(fake_bundle, candidate)
    receipt_path = tmp_path / "selection.json"
    requests.write_json_exclusive(receipt_path, receipt)
    authorization = requests.validate_independent_selection_receipt(
        receipt_path, contract=fake_bundle.contract
    )
    cache = _score_cache(fake_bundle, monkeypatch, tmp_path, independent=True)
    request = requests.build_external_score_request(
        score_cache=cache,
        candidate=candidate,
        budget=1,
        contract=fake_bundle.contract,
        gate_enabled=True,
        selection_authorization=authorization,
    )
    assert request["cohort"] == "independent_gate"
    assert request["development_selection_receipt"] == receipt
    other = requests.candidate_grid(fake_bundle.contract)[1]
    with pytest.raises(PermissionError, match="candidate/receipt"):
        requests.build_external_score_request(
            score_cache=cache,
            candidate=other,
            budget=1,
            contract=fake_bundle.contract,
            gate_enabled=True,
            selection_authorization=authorization,
        )


def test_query_labels_join_only_after_hashed_prediction_receipt(
    fake_bundle: FakeBundle,
) -> None:
    asset = fake_bundle.development
    participant = asset.participant("sub001")
    tokens = requests.deterministic_query_tokens(
        fake_bundle.secret, asset=asset, participant=participant
    )
    payload: dict[str, object] = {
        "status": "complete_query_outcome_free",
        "dataset_id": "beta_v1",
        "cohort": "development",
        "subject_id": "sub001",
        "query_tokens": list(tokens),
        "query_outcomes_loaded": False,
        "held_access_authorized": False,
    }
    payload["completion_receipt_sha256"] = requests._canonical_json_sha256(payload)
    joined = requests.evaluate_query_labels_after_predictions(
        [payload],
        asset=asset,
        participant=participant,
        token_secret=fake_bundle.secret,
    )
    assert joined["query_labels_loaded_only_after_predictions"] is True
    assert sorted(row["label"] for row in joined["rows"]) == list(range(40))
    with pytest.raises(ValueError, match="at least one"):
        requests.evaluate_query_labels_after_predictions(
            [],
            asset=asset,
            participant=participant,
            token_secret=fake_bundle.secret,
        )


def test_json_publication_is_exclusive_and_read_only(tmp_path: Path) -> None:
    target = tmp_path / "request.json"
    requests.write_json_exclusive(target, {"value": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1}
    assert stat.S_IMODE(target.stat().st_mode) == 0o400
    with pytest.raises(FileExistsError):
        requests.write_json_exclusive(target, {"value": 2})
    assert not any("staging" in value.name for value in tmp_path.iterdir())


def test_symlink_output_parent_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(PermissionError, match="symlink"):
        requests.write_json_exclusive(alias / "request.json", {"value": 1})
    assert not os.path.lexists(real / "request.json")

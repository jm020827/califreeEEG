"""New N1 source39 experiment role boundary and durable model binding.

Stored ZIP/NPY and selective JSON machinery is reused without altering old
classes/globals. The explicit generated profile is not human-run authority.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cfeg.analysis import task_trca_shape_archive as old
from cfeg.analysis.native_support_prefix import member_spans, pread_exact, require
from cfeg.analysis.task_trca_shape_inputs import RolePartition

ArchiveSpec = old.ArchiveSpec
SOURCE_IDS = old.SOURCE_IDS
SAMPLE_COUNTS = old.SAMPLE_COUNTS
INTERFACES = old.INTERFACES
SCHEMA = "task-trca-n1-integration-v1"
STUDY_ID = "task-trca-n1-source39-v1"
SCORE_SCHEMA = "component-time-centered-ensemble-pearson-v1"
FREEZE_SCHEMA = "cfeg.task_trca_n1_source39.all_models_frozen.v1"
GENERATED_IDS = (4, 6, 8, 11, 14, 21, 22, 25, 28)
_TOKEN_KEY = object()


@dataclass(frozen=True)
class RuntimeProfile:
    source_ids: tuple = SOURCE_IDS
    interfaces: tuple = (0, 1)
    samples: tuple = SAMPLE_COUNTS
    budgets: tuple = (3, 5)
    generated: bool = False
    seed: int | None = None

    def __post_init__(self):
        require(type(self.generated) is bool, "Explicit generated boolean required")
        expected = (
            (GENERATED_IDS, (0, 1), (17,), (3, 5), 20260914)
            if self.generated
            else (SOURCE_IDS, (0, 1), SAMPLE_COUNTS, (3, 5), None)
        )
        actual = (self.source_ids, self.interfaces, self.samples, self.budgets, self.seed)
        require(
            actual == expected and all(isinstance(v, tuple) for v in actual[:4]),
            "Only exact HUMAN or fixed GENERATED runtime profiles are allowed",
        )
        require(
            all(type(v) is int for values in actual[:4] for v in values),
            "Integer profile axes required",
        )
        require(self.seed is None or type(self.seed) is int, "Integer generated seed required")

    @property
    def conditions(self):
        return tuple((i, n, k) for i in self.interfaces for n in self.samples for k in self.budgets)

    def record(self):
        return {
            "source_ids": list(self.source_ids),
            "interfaces": list(self.interfaces),
            "samples": list(self.samples),
            "budgets": list(self.budgets),
            "generated": self.generated,
            "seed": self.seed,
        }


HUMAN_PROFILE = RuntimeProfile()
GENERATED_PROFILE = RuntimeProfile(GENERATED_IDS, (0, 1), (17,), (3, 5), True, 20260914)


def _profile(value):
    require(
        isinstance(value, RuntimeProfile) and value in (HUMAN_PROFILE, GENERATED_PROFILE),
        "Frozen RuntimeProfile required",
    )
    return value


class AllModelsFrozen:
    """Temporal-only token: old globally centered model tokens are rejected."""

    def __init__(self, key, receipt, manifest, files, partitions, profile):
        require(key is _TOKEN_KEY, "Use temporal verify_freeze")
        self._key, self._files, self._partitions = key, tuple(files), tuple(partitions)
        self.receipt_sha256, self.manifest_sha256, self.profile = receipt, manifest, profile

    def authorize(self, pid, partition, profile):
        require(
            self._key is _TOKEN_KEY and profile == self.profile, "Temporal freeze/profile mismatch"
        )
        expected = next((p for p in self._partitions if pid in p.evaluation_ids), None)
        if partition != expected:
            raise PermissionError("Exact frozen outer fit/evaluation partition required")
        for path, identity in self._files:
            require(
                path.resolve() == path
                and old._identity(path.stat(follow_symlinks=False)) == identity,
                "Frozen source/model/receipt changed or replaced",
            )


def verify_freeze(path, expected_sha256, expected_manifest_sha256, *, profile=HUMAN_PROFILE):
    """Bind all three new-schema model/source files to this output and manifest."""
    profile = _profile(profile)
    old._sha(expected_manifest_sha256)
    path = Path(path).absolute()
    require(path.name == "globalfreeze.json", "Fixed globalfreeze.json basename required")
    files, partitions = [], []
    with old._PinnedFile(path, expected_sha256, maximum_bytes=1024**2) as receipt:
        document = json.loads(
            pread_exact(receipt.fd, receipt.before.st_size, 0), object_pairs_hook=old._unique_pairs
        )
        require(
            isinstance(document, dict)
            and set(document)
            == {
                "schema",
                "status",
                "source_ids",
                "manifest_sha256",
                "query_access_count",
                "models",
            },
            "Exact temporal freeze envelope required",
        )
        require(
            document["schema"] == FREEZE_SCHEMA and document["status"] == "ALL_MODELS_FROZEN",
            "New temporal all-model freeze required; legacy freeze rejected",
        )
        require(
            document["source_ids"] == list(profile.source_ids)
            and all(type(v) is int for v in document["source_ids"]),
            "Freeze source IDs mismatch",
        )
        require(document["manifest_sha256"] == expected_manifest_sha256, "Freeze manifest mismatch")
        require(
            type(document["query_access_count"]) is int and document["query_access_count"] == 0,
            "No query decode before all-model freeze",
        )
        require(
            isinstance(document["models"], list) and len(document["models"]) == 3,
            "Exactly three frozen outer models required",
        )
        for fold, model in enumerate(document["models"]):
            require(
                isinstance(model, dict)
                and set(model)
                == {"fold_id", "fit_ids", "evaluation_ids", "path", "sha256", "bytes"},
                "Exact model freeze descriptor required",
            )
            fitting = tuple(pid for rank, pid in enumerate(profile.source_ids) if rank % 3 != fold)
            evaluation = profile.source_ids[fold::3]
            require(
                type(model["fold_id"]) is int
                and model["fold_id"] == fold
                and model["fit_ids"] == list(fitting)
                and model["evaluation_ids"] == list(evaluation)
                and all(type(v) is int for v in model["fit_ids"] + model["evaluation_ids"]),
                "Frozen outer partition mismatch",
            )
            expected_path = path.parent / f"model{fold}.json"
            require(
                model["path"] == str(expected_path), "Model must use exact output/fixed basename"
            )
            with old._PinnedFile(
                expected_path, model["sha256"], model["bytes"], 64 * 1024**2
            ) as pinned:
                payload = json.loads(
                    pread_exact(pinned.fd, pinned.before.st_size, 0),
                    object_pairs_hook=old._unique_pairs,
                )
                require(
                    isinstance(payload, dict)
                    and set(payload)
                    == {
                        "pipeline",
                        "selection",
                        "source_artifact",
                        "manifest_sha256",
                        "independent_source_audit",
                    },
                    "Exact temporal model payload required",
                )
                require(
                    payload["manifest_sha256"] == expected_manifest_sha256,
                    "Model manifest mismatch",
                )
                pipeline, selection = payload["pipeline"], payload["selection"]
                require(
                    pipeline.get("schema") == SCHEMA
                    and pipeline.get("score_schema") == SCORE_SCHEMA
                    and pipeline.get("fit_ids") == list(fitting),
                    "Legacy/wrong temporal model or fit IDs",
                )
                require(
                    selection.get("schema") == SCHEMA
                    and selection.get("score_schema") == SCORE_SCHEMA
                    and selection.get("outer_evaluation_ids") == list(evaluation),
                    "Model selection schema/role mismatch",
                )
                require(
                    payload["independent_source_audit"].get("status")
                    == "TEMPORAL_SOURCE_SELECTION_AUDIT_PASS",
                    "Independent source audit must pass before freeze",
                )
                source = payload["source_artifact"]
                require(
                    isinstance(source, dict)
                    and set(source) == {"path", "sha256", "bytes"}
                    and source["path"] == str(path.parent / f"source{fold}.npz"),
                    "Model source must use exact output/fixed basename",
                )
                with old._PinnedFile(
                    source["path"], source["sha256"], source["bytes"]
                ) as source_file:
                    files.append((source_file.path, old._identity(source_file.before)))
                files.append((pinned.path, old._identity(pinned.before)))
            partitions.append(RolePartition(fitting, (), evaluation))
        files.append((receipt.path, old._identity(receipt.before)))
    require(
        len({str(p) for p, _ in files}) == 7,
        "Distinct three source/three model/freeze files required",
    )
    return AllModelsFrozen(
        _TOKEN_KEY, expected_sha256, expected_manifest_sha256, files, partitions, profile
    )


class NativeArchive(old.NativeArchive):
    """Before-decode sink must durably persist or raise; rejected calls emit nothing."""

    def __init__(self, spec, partition, *, event_sink, profile=HUMAN_PROFILE):
        self.profile = _profile(profile)
        require(callable(event_sink), "Mandatory durable before-decode event sink required")
        require(
            spec.participant_id in self.profile.source_ids, "Participant outside runtime profile"
        )
        require(
            set(partition.fit_ids + partition.validation_ids + partition.evaluation_ids)
            <= set(self.profile.source_ids),
            "Roles outside runtime profile",
        )
        self._sink = event_sink
        super().__init__(spec, partition)

    def __enter__(self):
        self._file.__enter__()
        try:
            self._spans = member_spans(self._file.fd, self.profile.samples, self.spec.bytes)
        except BaseException:
            self._file.__exit__(None, None, None)
            raise
        return self

    def _context(self, interface, samples):
        require(
            type(interface) is int and interface in self.profile.interfaces,
            "Interface outside fixed profile",
        )
        require(
            type(samples) is int and samples in self.profile.samples, "Window outside fixed profile"
        )
        self._file.check()

    def _query_authority(self, freeze):
        if self.role != "evaluation" or not isinstance(freeze, AllModelsFrozen):
            raise PermissionError("Temporal query requires new verified freeze token")
        freeze.authorize(self.spec.participant_id, self.partition, self.profile)

    def _event(self, kind, interface, samples, blocks, freeze=None):
        value = {
            "phase": "before_decode",
            "participant_id": self.spec.participant_id,
            "role": self.role,
            "kind": kind,
            "interface": interface,
            "samples": samples,
            "blocks": list(blocks),
        }
        if freeze is not None:
            value["freeze_sha256"] = freeze.receipt_sha256
        self._sink(value)
        self.access_log.append(value)

    def support(self, interface, samples, k):
        require(
            type(k) is int and k in self.profile.budgets, "Support budget outside fixed profile"
        )
        self._context(interface, samples)
        self._event("support", interface, samples, range(k))
        return self._eeg(interface, samples, 0, k)

    def supervision(self, interface, samples):
        if self.role not in ("fit", "validation"):
            raise PermissionError("Evaluation block5 denied before decoding")
        self._context(interface, samples)
        self._event("source_supervision", interface, samples, (5,))
        return self._eeg(interface, samples, 5, 1)[0]

    def query(self, interface, samples, freeze=None):
        self._query_authority(freeze)
        self._context(interface, samples)
        self._event("query", interface, samples, range(6, 10), freeze)
        return self._eeg(interface, samples, 6, 4)

    def full_correlations(self, interface, samples, k, freeze=None):
        self._query_authority(freeze)
        require(type(k) is int and k in self.profile.budgets, "FULL budget outside fixed profile")
        self._context(interface, samples)
        self._event(f"full_k{k}", interface, samples, range(6, 10), freeze)
        data = old._npy_data(
            self._file.fd, self._spans[f"full_{samples}.npy"], (2, 2, 4, 12, 5, 12)
        )
        count = 4 * 12 * 5 * 12
        return old._decode_float64(
            self._file.fd, count, data + (interface * 2 + (k == 5)) * count * 8, (4, 12, 5, 12)
        )

    def a0_correlations(self, interface, samples, freeze=None):
        self._query_authority(freeze)
        self._context(interface, samples)
        self._event("a0", interface, samples, range(6, 10), freeze)
        data = old._npy_data(self._file.fd, self._spans[f"a0_{samples}.npy"], (2, 4, 12, 5, 12))
        count = 4 * 12 * 5 * 12
        return old._decode_float64(
            self._file.fd, count, data + interface * count * 8, (4, 12, 5, 12)
        )


class SupportMetadata(old.SupportMetadata):
    """Reuse the strict780-packet lexical index; only permitted numeric prefixes decode."""

    def __init__(
        self, path, sha256, expected_envelope, partition, *, event_sink, profile=HUMAN_PROFILE
    ):
        self.profile = _profile(profile)
        require(callable(event_sink), "Mandatory durable before-decode event sink required")
        require(
            set(partition.fit_ids + partition.validation_ids + partition.evaluation_ids)
            <= set(self.profile.source_ids),
            "Metadata roles outside runtime profile",
        )
        if self.profile.generated:
            require(
                "GENERATED" in expected_envelope.get("study_id", ""),
                "Generated metadata envelope required",
            )
        self._sink = event_sink
        super().__init__(path, sha256, expected_envelope, partition)

    def support(self, participant_id, interface, k):
        role = self.partition.role(participant_id)
        require(participant_id in self.profile.source_ids, "Metadata participant outside profile")
        require(
            type(interface) is int and interface in self.profile.interfaces,
            "Metadata interface outside profile",
        )
        require(
            type(k) is int and k in self.profile.budgets, "Metadata support budget outside profile"
        )
        self._file.check()
        event = {
            "phase": "before_decode",
            "participant_id": participant_id,
            "role": role,
            "kind": "metadata_support",
            "interface": interface,
            "blocks": list(range(k)),
        }
        self._sink(event)
        self.access_log.append(event)
        rows = [
            self._json.decode(self._packets[participant_id, interface, block]) for block in range(k)
        ]
        result = np.asarray(rows, dtype=np.float64)
        require(
            result.shape == (k, 8)
            and np.all(np.isnan(result) | (np.isfinite(result) & (result >= 0))),
            "Authorized impedance must be nonnegative finite or missing",
        )
        return np.frombuffer(result.tobytes(), dtype=np.float64).reshape(k, 8), self._orders[
            participant_id
        ]

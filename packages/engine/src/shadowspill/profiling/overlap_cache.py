"""Content-addressed paired evidence, separate from isolated memory profiles."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from shadowspill.artifacts._values import digest, plain
from shadowspill.ir import TaskOverlapCost
from shadowspill.store import (
    STORE_MODES,
    ArtifactStore,
    StoreMode,
    StorePolicy,
    atomic_json,
    digest_directory,
)
from shadowspill.store.artifacts import read_json

_SCHEMA = "shadowspill.overlap-measurement/v1"


def observed_cost(observation: Mapping[str, Any]) -> TaskOverlapCost | None:
    """Preserve the probe's conservative aggregation over supported ranks."""
    supported = [
        row["rates"]
        for row in observation.get("per_rank", ())
        if row["rates"] is not None
    ]
    if not supported:
        return None
    return TaskOverlapCost(
        observation["timing_class"],
        observation["bytes"],
        min(row[0] for row in supported),
        min(row[1] for row in supported),
    )


class OverlapProfileCache:
    """Reuse supported task/route evidence across candidates and processes.

    Identity includes finalized code, representative input lineage, numerical
    settings, allocator/stream policy, rank mapping, topology, backend, actual
    route geometry and sampling protocol. This never supplies task timings,
    allocation contracts, fixed memory or communication-memory evidence.

    Modes have the same meanings as TaskProfileCache. Refresh reuses new entries
    within this instance; caller-supplied task inputs disable reuse. Missing or
    unsupported observations are not cached. Every rank must agree on identical
    evidence before skipping a probe; rank zero alone persists the shared entry.
    """

    def __init__(
        self,
        root: str | Path | ArtifactStore | None = None,
        *,
        mode: StoreMode = "contribute",
    ) -> None:
        if mode not in STORE_MODES:
            raise ValueError(f"invalid overlap profile cache mode: {mode!r}")
        self.store = (
            root if isinstance(root, ArtifactStore) else ArtifactStore.resolve(root)
        )
        self.policy = StorePolicy.for_mode(mode)
        self._records: dict[str, dict[str, Any]] = {}

    def _path(self, key: str) -> Path:
        return digest_directory(self.store.overlap_profiles, key) / "measurement.json"

    def get(self, identity: Mapping[str, Any]) -> dict[str, Any] | None:
        key = digest(identity)
        if key in self._records:
            return plain(self._records[key])  # type: ignore[no-any-return]
        path = self._path(key)
        if not self.policy.read_enabled or not path.is_file():
            return None
        value = read_json(path)
        if value.get("schema") != _SCHEMA or value.get("identity") != plain(identity):
            raise ValueError(f"overlap measurement cache identity differs: {path}")
        observation = value.get("observation")
        if not isinstance(observation, dict) or digest(observation) != value.get(
            "observation_digest"
        ):
            raise ValueError(f"overlap measurement cache evidence is corrupt: {path}")
        self._validate(identity, observation)
        self._records[key] = observation
        self._record(key, path, "read")
        return plain(observation)  # type: ignore[no-any-return]

    def put(
        self,
        identity: Mapping[str, Any],
        observation: Mapping[str, Any],
        *,
        persist: bool = True,
    ) -> bool:
        if observation.get("status") != "measured":
            return False
        self._validate(identity, observation)
        key = digest(identity)
        value = plain(observation)
        self._records[key] = value
        if persist and self.policy.write_enabled:
            path = self._path(key)
            if self.policy.overwrite or not path.is_file():
                atomic_json(
                    path,
                    {
                        "schema": _SCHEMA,
                        "identity": plain(identity),
                        "observation": value,
                        "observation_digest": digest(value),
                    },
                )
                self._record(key, path, "write")
        return True

    def _record(self, key: str, path: Path, access: str) -> None:
        self.store.record(
            category="profiling",
            kind="overlap_measurement",
            digest=key,
            path=path,
            access=access,
            schema=_SCHEMA,
        )

    @staticmethod
    def _validate(identity: Mapping[str, Any], row: Mapping[str, Any]) -> None:
        route = identity["route"]
        expected_ranks = set(identity["task"]["environment"]["rank_assignment"])
        peers = row.get("per_rank", ())
        if (
            row.get("status") != "measured"
            or row.get("timing_class") != route["timing_class"]
            or row.get("bytes") != route["bytes"]
            or len(peers) != len(expected_ranks)
            or {peer["logical_rank"] for peer in peers} != expected_ranks
            or observed_cost(row) is None
        ):
            raise ValueError("overlap cache requires compatible supported evidence")
        for peer in peers:
            if peer["rates"] is not None:
                TaskOverlapCost(route["timing_class"], route["bytes"], *peer["rates"])
                count = peer.get("requested_trials")
                if (
                    type(count) is not int
                    or count < 1
                    or any(
                        len(peer.get("trials", {}).get(mode, ())) != count
                        for mode in ("task", "transfer", "both")
                    )
                ):
                    raise ValueError("overlap cache is missing paired trial evidence")

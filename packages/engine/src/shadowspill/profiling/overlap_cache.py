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
    """A store of overlap measurements, reused across candidates and processes.

    Pass one to `measure_overlap`, the same instance across a candidate
    sweep. A measurement is reused only for the same finalized code, input
    lineage, numerical settings, allocator and stream policy, rank layout,
    topology, backend, transfer geometry and sampling protocol, and only when
    every rank holds the same evidence; rank zero alone writes shared
    entries. It never supplies task timings, allocation evidence or fixed
    memory. Missing or unsupported observations are not stored, and explicit
    ``inputs`` disable reuse.

    Parameters
    ----------
    root : str or pathlib.Path or ArtifactStore, optional
        The store, or its root directory. By default the user's store under
        ``~/.cache/shadowspill``.
    mode : str, default "contribute"
        ``"contribute"``, ``"reuse"``, ``"require"`` or ``"refresh"``, as for
        `TaskProfileCache`.

    Attributes
    ----------
    store : ArtifactStore
        The store the measurements live in.
    policy : StorePolicy
        What ``mode`` permits.

    Raises
    ------
    ValueError
        If ``mode`` is none of these.
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
        """Look up the observation of one task and transfer route.

        Parameters
        ----------
        identity : mapping
            The observation's identity, as `measure_overlap` builds it.

        Returns
        -------
        dict or None
            A copy of the observation; None on a miss.

        Raises
        ------
        ValueError
            If the stored entry's identity differs, it is corrupt, or its
            evidence does not fit ``identity``.
        """
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
        """Record an observation in this instance, and on disk if allowed.

        An entry on disk is replaced only in ``"refresh"`` mode.

        Parameters
        ----------
        identity : mapping
            The observation's identity, as for `get`.
        observation : mapping
            The paired measurement, with every rank's trials.
        persist : bool, default True
            Whether this instance may write it to disk; `measure_overlap`
            lets rank zero alone.

        Returns
        -------
        bool
            Whether it was recorded; only a ``"measured"`` observation is.

        Raises
        ------
        ValueError
            If the observation does not fit ``identity`` or lacks paired
            trial evidence.
        """
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

"""Portable communication assumptions, separate from task profiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, ClassVar

from shadowspill.artifacts._values import freeze, plain, seconds
from shadowspill.artifacts.archive import Artifact


@dataclass(frozen=True)
class CommunicationInfo(Artifact):
    """Transfer and collective costs for planning, measured or supplied.

    `measure_communication` returns one; one written by hand plans with
    assumed rates. It is kept apart from task profiles, so a change of
    interconnect or backend replaces it alone. A fetch copies a value from the
    spill pool to the device, an eviction back. A collective with a matching
    measurement takes its measured time; any other is estimated as a ring
    from the scale-up and scale-out rates.

    Parameters
    ----------
    fetch_bytes_per_second : float
        Fetch copy rate, in bytes per second. At least one.
    evict_bytes_per_second : float
        Eviction copy rate, in bytes per second. At least one.
    fetch_concurrent_bytes_per_second : float, optional
        Fetch rate while an eviction runs too. By default the solo rate, as
        for independent full-duplex lanes.
    evict_concurrent_bytes_per_second : float, optional
        Eviction rate while a fetch runs too. By default the solo rate.
    fetch_latency_seconds : float, default 0.0
        Fixed time added to each fetch, in seconds.
    evict_latency_seconds : float, default 0.0
        Fixed time added to each eviction, in seconds.
    topology : mapping of str to sequence of int, default {}
        The logical ranks on each node, by node name. A rank is on at most
        one node.
    metadata : mapping, default {}
        Provenance, such as the measured ``spill_kind``, ``backend_setup``,
        ``profiling_options`` and ``host_placement``. ``timing_kind`` is
        ``"estimated"`` for edited or assumed timings.
    scale_up_bytes_per_second : float, optional
        Per-rank collective rate between ranks on one node, in bytes per
        second. At least one.
    scale_out_bytes_per_second : float, optional
        Per-rank collective rate between ranks on different nodes, in bytes
        per second. At least one.
    scale_up_latency_seconds : float, optional
        Time of each ring step within a node, in seconds.
    scale_out_latency_seconds : float, optional
        Time of each ring step across nodes, in seconds.
    measurements : sequence of mapping, default ()
        Measured collectives, each with its ``operation``, ``ranks``,
        ``tensor_bytes``, ``dtype`` and ``duration_seconds``.
    fixed_device_bytes : int, optional
        Device memory the communicators retain on each worker, in bytes.

    Raises
    ------
    ValueError
        If a rate is below one byte per second, a rate or latency is negative
        or not finite, a rank appears twice in ``topology``, a measurement is
        malformed, or measured communicator memory lacks its scope.
    """

    artifact_kind: ClassVar[str] = "communication_info"
    fetch_bytes_per_second: float
    evict_bytes_per_second: float
    fetch_concurrent_bytes_per_second: float | None = None
    evict_concurrent_bytes_per_second: float | None = None
    fetch_latency_seconds: float = 0.0
    evict_latency_seconds: float = 0.0
    topology: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    scale_up_bytes_per_second: float | None = None
    scale_out_bytes_per_second: float | None = None
    scale_up_latency_seconds: float | None = None
    scale_out_latency_seconds: float | None = None
    measurements: tuple[Mapping[str, Any], ...] = ()
    fixed_device_bytes: int | None = None

    def __post_init__(self) -> None:
        from .memory_records import validate_fixed_memory

        validate_fixed_memory(self.fixed_device_bytes, self.metadata)
        for direction in ("fetch", "evict"):
            for suffix in ("bytes_per_second", "concurrent_bytes_per_second"):
                key = f"{direction}_{suffix}"
                rate = getattr(self, key)
                if rate is not None:
                    seconds(rate, key)
                    if rate < 1:
                        raise ValueError(f"{key} must be at least one byte/second")
            seconds(getattr(self, f"{direction}_latency_seconds"), "latency_seconds")
        ranks = [rank for group in self.topology.values() for rank in group]
        if len(ranks) != len(set(ranks)) or any(
            type(r) is not int or r < 0 for r in ranks
        ):
            raise ValueError("topology must assign each logical rank to one node")
        object.__setattr__(self, "topology", freeze(self.topology))
        object.__setattr__(self, "metadata", freeze(self.metadata))
        for scope in ("scale_up", "scale_out"):
            rate = getattr(self, f"{scope}_bytes_per_second")
            latency = getattr(self, f"{scope}_latency_seconds")
            if rate is not None:
                seconds(rate, f"{scope}_bytes_per_second")
                if rate < 1:
                    raise ValueError(f"{scope} rate must be at least one byte/second")
            if latency is not None:
                seconds(latency, f"{scope}_latency_seconds")
        from .collectives import validate_measurement

        for measurement in self.measurements:
            validate_measurement(measurement)
        object.__setattr__(self, "measurements", freeze(tuple(self.measurements)))

    def to_dict(self) -> dict[str, Any]:
        """Return the costs as plain data.

        Returns
        -------
        dict
            Every field, by name.
        """
        return {
            name: plain(getattr(self, name))
            for name in self.__dataclass_fields__
            if name != "artifact_kind"
        }

    @cached_property
    def _measurement_index(
        self,
    ) -> dict[tuple[Any, ...], tuple[Mapping[str, Any], ...]]:
        from .collectives import measurement_index

        return measurement_index(self)

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, payloads: Mapping[str, bytes] | None = None
    ) -> CommunicationInfo:
        """Rebuild costs from `to_dict` data.

        Parameters
        ----------
        value : dict
            The costs, in `to_dict` form.
        payloads : mapping of str to bytes, optional
            Binary payloads; communication costs hold none, so it must be
            empty.

        Returns
        -------
        CommunicationInfo
            The costs.

        Raises
        ------
        ValueError
            If ``payloads`` is not empty, or a field is invalid.
        """
        if payloads:
            raise ValueError("CommunicationInfo cannot contain executable payloads")
        return cls(**value)

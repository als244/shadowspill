"""How one initialized runtime arranges its memory, and what it costs.

The pools a plan can be placed into, the routes between them, and the
bandwidth and latency the runtime measured for each route. Planning
reads all of it and has no other way to know any of it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemoryPool:
    """One pool of an open runtime, as planning sees it.

    `Runtime.pools` holds one for each configured pool, by name.

    Attributes
    ----------
    name : str
        The pool's name, as given to the runtime.
    pool_id : int
        The runtime's id for the pool: its position among the configured
        pools.
    kind : str
        ``"device"`` for the device pool; for a spill pool, its kind's name,
        such as ``"pinned_host"``.
    capacity : int
        Bytes the pool can lease. For the device pool, what its budget leaves
        once the memory the process already holds and the external headroom
        are set aside.
    physical_capacity : int
        For the device pool, its declared device budget, in bytes; for a
        spill pool, its capacity.
    device_ordinal : int or None
        The device pool's accelerator ordinal; None for a spill pool.
    addressable : bool
        Whether this process can address a lease in the pool directly. When
        it cannot, the framework keeps its own storage and the runtime copies
        bytes in and out.
    """

    name: str
    pool_id: int
    kind: str
    capacity: int
    physical_capacity: int | None
    device_ordinal: int | None
    addressable: bool = True


@dataclass(frozen=True, slots=True)
class RuntimeRoute:
    """One directed route of an open runtime, as planning sees it.

    `Runtime.routes` holds one for each configured route, by name.

    Attributes
    ----------
    name : str
        The route's name, as given to the runtime.
    route_id : int
        The runtime's id for the route: its position among the configured
        routes.
    source : str
        Name of the pool bytes move from.
    destination : str
        Name of the pool bytes move to.
    source_pool_id : int
        The source pool's id.
    destination_pool_id : int
        The destination pool's id.
    """

    name: str
    route_id: int
    source: str
    destination: str
    source_pool_id: int
    destination_pool_id: int


@dataclass(frozen=True, slots=True)
class TransferProfile:
    """What the runtime measured for moving bytes from one pool to another.

    One cell of `TransferCapabilities`, which has a cell for every ordered
    pair of pools, each pool to itself included.

    Attributes
    ----------
    source : str
        Name of the pool bytes move from.
    destination : str
        Name of the pool bytes move to.
    source_pool_id : int
        The source pool's id.
    destination_pool_id : int
        The destination pool's id.
    generation : int
        The matrix generation the cell was published in.
    latency_nanoseconds : int
        Median duration of one small copy, in nanoseconds; zero when not
        measured.
    bandwidth_bytes_per_second : int
        The rate planning and simulation use, in bytes per second: the
        concurrent rate when the route was measured with its reverse route,
        otherwise the solo rate. Zero when not measured; the largest 64-bit
        value from a pool to itself.
    solo_bandwidth_bytes_per_second : int
        The rate measured with no traffic on the reverse route, in bytes per
        second; zero when not measured.
    concurrent_bandwidth_bytes_per_second : int
        The rate measured while the reverse route was busy, in bytes per
        second; zero when not measured.
    solo_measurement_nanoseconds : int
        Median duration of one batch of large copies measured alone, in
        nanoseconds.
    concurrent_measurement_nanoseconds : int
        The same with the reverse route busy; zero when not measured.
    calibrated_timestamp_nanoseconds : int
        When the cell was last measured, on the runtime's monotonic clock, in
        nanoseconds; zero when never measured.
    small_copy_bytes : int
        The copy size latency was measured with, in bytes.
    large_copy_bytes : int
        The copy size bandwidth was measured with, in bytes.
    measured_copies : int
        Copies timed per measurement.
    available : bool
        Whether bytes can move between the pools: a route joins them, or they
        are the same pool.
    calibrated : bool
        Whether the cell has been measured, or joins a pool to itself.
    provenance : str
        ``"recalibration"`` for a measurement taken by
        `Runtime.calibrate_transfer_capabilities`; ``"initialization"`` for
        one taken when the runtime opened, or for a cell never measured.
    calibration_mode : str
        ``"identity"`` from a pool to itself, ``"bidirectional_concurrent"``
        for a route measured with its reverse route, ``"solo"`` otherwise.
    concurrent_route_count : int
        Routes moving bytes during the measurement: 0 from a pool to itself,
        1 for a route measured alone, 2 with its reverse route.
    """

    source: str
    destination: str
    source_pool_id: int
    destination_pool_id: int
    generation: int
    latency_nanoseconds: int
    bandwidth_bytes_per_second: int
    solo_bandwidth_bytes_per_second: int
    concurrent_bandwidth_bytes_per_second: int
    solo_measurement_nanoseconds: int
    concurrent_measurement_nanoseconds: int
    calibrated_timestamp_nanoseconds: int
    small_copy_bytes: int
    large_copy_bytes: int
    measured_copies: int
    available: bool
    calibrated: bool
    provenance: str
    calibration_mode: str
    concurrent_route_count: int

    def as_dict(self) -> dict[str, object]:
        """Return the cell as a JSON-compatible dictionary.

        Returns
        -------
        dict
            Every field by name.
        """
        return {
            "source": self.source,
            "destination": self.destination,
            "source_pool_id": self.source_pool_id,
            "destination_pool_id": self.destination_pool_id,
            "generation": self.generation,
            "latency_nanoseconds": self.latency_nanoseconds,
            "bandwidth_bytes_per_second": self.bandwidth_bytes_per_second,
            "solo_bandwidth_bytes_per_second": self.solo_bandwidth_bytes_per_second,
            "concurrent_bandwidth_bytes_per_second": (
                self.concurrent_bandwidth_bytes_per_second
            ),
            "solo_measurement_nanoseconds": self.solo_measurement_nanoseconds,
            "concurrent_measurement_nanoseconds": (
                self.concurrent_measurement_nanoseconds
            ),
            "calibrated_timestamp_nanoseconds": self.calibrated_timestamp_nanoseconds,
            "small_copy_bytes": self.small_copy_bytes,
            "large_copy_bytes": self.large_copy_bytes,
            "measured_copies": self.measured_copies,
            "available": self.available,
            "calibrated": self.calibrated,
            "provenance": self.provenance,
            "calibration_mode": self.calibration_mode,
            "concurrent_route_count": self.concurrent_route_count,
        }


@dataclass(frozen=True, slots=True)
class TransferCapabilities:
    """The transfer matrix an open runtime publishes, as an immutable snapshot.

    `Runtime.transfer_capabilities` and
    `Runtime.calibrate_transfer_capabilities` return it, with one
    `TransferProfile` for every ordered pair of pools. Each calibration
    publishes a new matrix with the next generation. Planning reads the
    matrix and measures nothing itself.

    Attributes
    ----------
    generation : int
        The number of calibrations the runtime has published; zero before
        the first.
    pool_names : tuple of str
        The pools, in id order.
    profiles : tuple of TransferProfile
        The cells, row by row: the cell from pool ``i`` to pool ``j`` is at
        position ``i * len(pool_names) + j``.
    digest : str
        SHA-256 of the generation, the pool names and the cells, in
        hexadecimal.
    """

    generation: int
    pool_names: tuple[str, ...]
    profiles: tuple[TransferProfile, ...]
    digest: str

    def route(self, source: str, destination: str) -> TransferProfile:
        """Look up the cell from one pool to another.

        Parameters
        ----------
        source : str
            Name of the pool bytes move from.
        destination : str
            Name of the pool bytes move to.

        Returns
        -------
        TransferProfile
            The cell for that ordered pair.

        Raises
        ------
        KeyError
            If either name is not in ``pool_names``.
        """
        try:
            source_id = self.pool_names.index(source)
            destination_id = self.pool_names.index(destination)
        except ValueError as exc:
            raise KeyError((source, destination)) from exc
        return self.profiles[source_id * len(self.pool_names) + destination_id]

    def as_dict(self) -> dict[str, object]:
        """Return the matrix as a JSON-compatible dictionary.

        Returns
        -------
        dict
            ``generation``, ``pool_names`` (a list), ``profiles`` (each cell
            as `TransferProfile.as_dict` gives it) and ``digest``.
        """
        return {
            "generation": self.generation,
            "pool_names": list(self.pool_names),
            "profiles": [profile.as_dict() for profile in self.profiles],
            "digest": self.digest,
        }


__all__ = [
    "MemoryPool",
    "RuntimeRoute",
    "TransferCapabilities",
    "TransferProfile",
]

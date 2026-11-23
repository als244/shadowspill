"""Public immutable simulator inputs, intervals, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from shadowspill.ir import MemoryActionKind, ResourceKind


def _require_non_negative(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive(value: int, name: str) -> None:
    _require_non_negative(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class DeviceSimulationConfig:
    """One device's capacity and copy rates, as the simulator prices them.

    A copy moves at its direction's solo rate while the opposite direction
    is idle on the same device, and at its concurrent rate while both move
    bytes. Equal solo and concurrent rates give a fixed rate.

    Parameters
    ----------
    device_id : str
        The device's identifier in the program. Non-empty, with no
        surrounding whitespace.
    capacity_bytes : int
        Device memory the program may occupy, in bytes, shared residency
        included. At least zero.
    fetch_solo_bandwidth_bytes_per_second : int
        Fetch rate while no eviction moves bytes, in bytes per second.
        Positive.
    fetch_concurrent_bandwidth_bytes_per_second : int
        Fetch rate while an eviction moves bytes, in bytes per second.
        Positive.
    evict_solo_bandwidth_bytes_per_second : int
        Eviction and write-back rate while no fetch moves bytes, in bytes per
        second. Positive.
    evict_concurrent_bandwidth_bytes_per_second : int
        Eviction and write-back rate while a fetch moves bytes, in bytes per
        second. Positive.
    fetch_latency_ns : int, default 0
        Fetch latency, in nanoseconds. At least zero. Kept as calibration
        metadata; simulated timing does not use it.
    evict_latency_ns : int, default 0
        Eviction latency, in nanoseconds. At least zero. Kept as calibration
        metadata; simulated timing does not use it.

    Raises
    ------
    ValueError
        If a field is empty, not an integer or out of range.
    """

    device_id: str
    capacity_bytes: int
    fetch_solo_bandwidth_bytes_per_second: int
    fetch_concurrent_bandwidth_bytes_per_second: int
    evict_solo_bandwidth_bytes_per_second: int
    evict_concurrent_bandwidth_bytes_per_second: int
    fetch_latency_ns: int = 0
    evict_latency_ns: int = 0

    def __post_init__(self) -> None:
        if not self.device_id or self.device_id.strip() != self.device_id:
            raise ValueError("device_id must be a non-empty normalized string")
        _require_non_negative(self.capacity_bytes, "capacity_bytes")
        for direction in ("fetch", "evict"):
            for mode in ("solo", "concurrent"):
                name = f"{direction}_{mode}_bandwidth_bytes_per_second"
                _require_positive(getattr(self, name), name)
        _require_non_negative(self.fetch_latency_ns, "fetch_latency_ns")
        _require_non_negative(self.evict_latency_ns, "evict_latency_ns")


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """The machine a schedule is simulated on: its devices and spill capacity.

    Parameters
    ----------
    devices : tuple of DeviceSimulationConfig
        One entry per device, each with its own ``device_id``. Non-empty.
        `simulate` requires exactly the program's devices.
    spill_capacity_bytes : int
        Spill memory the program may occupy, in bytes, shared residency
        included. At least zero.

    Raises
    ------
    ValueError
        If ``devices`` is not a non-empty tuple of `DeviceSimulationConfig`,
        names a device twice, or ``spill_capacity_bytes`` is negative.
    """

    devices: tuple[DeviceSimulationConfig, ...]
    spill_capacity_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.devices, tuple) or not self.devices:
            raise ValueError("devices must be a non-empty tuple")
        seen: set[str] = set()
        for index, device in enumerate(self.devices):
            if not isinstance(device, DeviceSimulationConfig):
                raise ValueError(f"devices[{index}] has an invalid type")
            if device.device_id in seen:
                raise ValueError(f"devices[{index}] duplicates {device.device_id!r}")
            seen.add(device.device_id)
        _require_non_negative(self.spill_capacity_bytes, "spill_capacity_bytes")

    @classmethod
    def single_device(
        cls,
        device_id: str,
        *,
        device_capacity_bytes: int,
        spill_capacity_bytes: int,
        fetch_solo_bandwidth_bytes_per_second: int,
        fetch_concurrent_bandwidth_bytes_per_second: int,
        evict_solo_bandwidth_bytes_per_second: int,
        evict_concurrent_bandwidth_bytes_per_second: int,
        fetch_latency_ns: int = 0,
        evict_latency_ns: int = 0,
    ) -> SimulationConfig:
        """Build a configuration with one device.

        Parameters
        ----------
        device_id : str
            The device's identifier in the program.
        device_capacity_bytes : int
            Its memory, as `DeviceSimulationConfig.capacity_bytes`.
        spill_capacity_bytes : int
            Spill memory, in bytes, as in `SimulationConfig`.
        fetch_solo_bandwidth_bytes_per_second : int
            As in `DeviceSimulationConfig`.
        fetch_concurrent_bandwidth_bytes_per_second : int
            As in `DeviceSimulationConfig`.
        evict_solo_bandwidth_bytes_per_second : int
            As in `DeviceSimulationConfig`.
        evict_concurrent_bandwidth_bytes_per_second : int
            As in `DeviceSimulationConfig`.
        fetch_latency_ns : int, default 0
            As in `DeviceSimulationConfig`.
        evict_latency_ns : int, default 0
            As in `DeviceSimulationConfig`.

        Returns
        -------
        SimulationConfig
            The configuration.

        Raises
        ------
        ValueError
            If a value is out of range.
        """
        return cls(
            devices=(
                DeviceSimulationConfig(
                    device_id=device_id,
                    capacity_bytes=device_capacity_bytes,
                    fetch_solo_bandwidth_bytes_per_second=(
                        fetch_solo_bandwidth_bytes_per_second
                    ),
                    fetch_concurrent_bandwidth_bytes_per_second=(
                        fetch_concurrent_bandwidth_bytes_per_second
                    ),
                    evict_solo_bandwidth_bytes_per_second=(
                        evict_solo_bandwidth_bytes_per_second
                    ),
                    evict_concurrent_bandwidth_bytes_per_second=(
                        evict_concurrent_bandwidth_bytes_per_second
                    ),
                    fetch_latency_ns=fetch_latency_ns,
                    evict_latency_ns=evict_latency_ns,
                ),
            ),
            spill_capacity_bytes=spill_capacity_bytes,
        )


@dataclass(frozen=True, slots=True)
class TaskPhysicalDelta:
    """How one task changes its device's physical pool use as it starts and ends.

    A `SimulationAdmission` lists these for the tasks whose deltas it sets.

    Parameters
    ----------
    task_id : str
        The task. Non-empty, with no surrounding whitespace.
    start_bytes : int
        Change in physical bytes in use when the task starts. Negative frees.
    completion_bytes : int
        Change in physical bytes in use when the task completes. Negative
        frees.

    Raises
    ------
    ValueError
        If ``task_id`` is empty or padded with whitespace, or a delta is not an
        integer.
    """

    task_id: str
    start_bytes: int
    completion_bytes: int

    def __post_init__(self) -> None:
        if not self.task_id or self.task_id.strip() != self.task_id:
            raise ValueError("task_id must be a non-empty normalized string")
        if isinstance(self.start_bytes, bool) or not isinstance(self.start_bytes, int):
            raise ValueError("start_bytes must be an integer")
        if isinstance(self.completion_bytes, bool) or not isinstance(
            self.completion_bytes, int
        ):
            raise ValueError("completion_bytes must be an integer")


@dataclass(frozen=True, slots=True)
class ActionPhysicalDelta:
    """How one memory action changes its device's physical pool use.

    A `SimulationAdmission` lists these for the actions whose deltas it sets.
    The trigger delta is charged when the action is submitted, so a fetch
    reserves its destination before its bytes move.

    Parameters
    ----------
    action_index : int
        The action's position in the schedule's actions. At least zero.
    trigger_bytes : int
        Change in physical bytes in use when the action is triggered.
        Negative frees.
    completion_bytes : int
        Change in physical bytes in use when the action completes. Negative
        frees.

    Raises
    ------
    ValueError
        If ``action_index`` is negative, or a delta is not an integer.
    """

    action_index: int
    trigger_bytes: int
    completion_bytes: int

    def __post_init__(self) -> None:
        _require_non_negative(self.action_index, "action_index")
        if isinstance(self.trigger_bytes, bool) or not isinstance(
            self.trigger_bytes, int
        ):
            raise ValueError("trigger_bytes must be an integer")
        if isinstance(self.completion_bytes, bool) or not isinstance(
            self.completion_bytes, int
        ):
            raise ValueError("completion_bytes must be an integer")


@dataclass(frozen=True, slots=True)
class MemoryReuseDependency:
    """A completion that must precede the reuse of a physical byte range.

    The successor does not start until the predecessor has completed and
    freed the range. Exactly one predecessor, a task or an eviction, and
    exactly one successor, a task or an action, are named.

    Parameters
    ----------
    predecessor_action_index : int, optional
        The eviction, by position in the schedule's actions, whose
        completion frees the range.
    successor_task_id : str, optional
        The task that reuses the range.
    successor_action_index : int, optional
        The action, by position in the schedule's actions, that reuses the
        range.
    predecessor_task_id : str, optional
        The task whose completion frees the range.

    Raises
    ------
    ValueError
        If not exactly one predecessor or not exactly one successor is named,
        an index is negative, an identifier is empty or padded with whitespace,
        or a
        task would wait for its own completion.
    """

    predecessor_action_index: int | None = None
    successor_task_id: str | None = None
    successor_action_index: int | None = None
    predecessor_task_id: str | None = None

    def __post_init__(self) -> None:
        if (self.predecessor_action_index is None) == (
            self.predecessor_task_id is None
        ):
            raise ValueError(
                "exactly one memory-reuse predecessor task or action is required"
            )
        if self.predecessor_action_index is not None:
            _require_non_negative(
                self.predecessor_action_index, "predecessor_action_index"
            )
        if self.predecessor_task_id is not None and (
            not self.predecessor_task_id
            or self.predecessor_task_id.strip() != self.predecessor_task_id
        ):
            raise ValueError(
                "predecessor_task_id must be a non-empty normalized string"
            )
        if (
            self.predecessor_task_id is not None
            and self.predecessor_task_id == self.successor_task_id
        ):
            raise ValueError("a task cannot wait for its own completion")
        if (self.successor_task_id is None) == (self.successor_action_index is None):
            raise ValueError(
                "exactly one memory-reuse successor task or action is required"
            )
        if self.successor_task_id is not None and (
            not self.successor_task_id
            or self.successor_task_id.strip() != self.successor_task_id
        ):
            raise ValueError("successor_task_id must be a non-empty normalized string")
        if self.successor_action_index is not None:
            _require_non_negative(self.successor_action_index, "successor_action_index")


@dataclass(frozen=True, slots=True)
class SimulationAdmission:
    """Physical pool accounting for a simulation, fixed before timing is known.

    Given one, `simulate` charges each device's physical pool use against
    capacity in place of its object and workspace bytes: the use starts at
    ``initial_physical_bytes`` and changes by each task's and action's delta,
    and each reuse dependency orders a reuse after the completion that
    frees its range.

    Parameters
    ----------
    initial_physical_bytes : tuple of (str, int)
        Physical bytes in use on each device before the first task, as
        ``(device_id, bytes)`` pairs naming exactly the program's devices.
        Each at least zero.
    device_capacity_bytes : tuple of (str, int), default ()
        Capacity to charge against on each device, as ``(device_id, bytes)``
        pairs, in place of the configured capacity. Empty keeps the
        configuration's; otherwise it names the same devices as
        ``initial_physical_bytes``, each positive.
    task_deltas : tuple of TaskPhysicalDelta, default ()
        Deltas of the tasks listed, at most one per task.
    action_deltas : tuple of ActionPhysicalDelta, default ()
        Deltas of the actions listed, at most one per action.
    reuse_dependencies : tuple of MemoryReuseDependency, default ()
        Completions that must precede a reuse.
    unlisted_delta_bytes : int, optional
        The delta, at both boundaries, of every task and action the lists
        leave out. None gives each the delta its logical bytes imply, as a
        simulation without admission charges them. A fixed layout holds all
        of its memory from the start, so it passes zero and lists nothing.

    Raises
    ------
    ValueError
        If a device, task or action is listed twice, a value is out of range,
        or the capacities and initial bytes name different devices.
    """

    initial_physical_bytes: tuple[tuple[str, int], ...]
    device_capacity_bytes: tuple[tuple[str, int], ...] = ()
    task_deltas: tuple[TaskPhysicalDelta, ...] = ()
    action_deltas: tuple[ActionPhysicalDelta, ...] = ()
    reuse_dependencies: tuple[MemoryReuseDependency, ...] = ()
    #: The delta, at both boundaries, of every task and action the lists
    #: above leave out. `None` leaves each to its logical bytes, which is what
    #: the simulator charges without admission; a fixed layout holds all of
    #: its memory in its slice from the start, so for it every delta is zero
    #: and nothing need be listed.
    unlisted_delta_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.unlisted_delta_bytes is not None and (
            isinstance(self.unlisted_delta_bytes, bool)
            or not isinstance(self.unlisted_delta_bytes, int)
        ):
            raise ValueError("unlisted_delta_bytes must be an integer or None")
        _validate_unique_pairs(self.initial_physical_bytes, "initial_physical_bytes")
        for device_id, bytes_ in self.initial_physical_bytes:
            if not device_id or device_id.strip() != device_id:
                raise ValueError(
                    "initial physical device IDs must be normalized strings"
                )
            _require_non_negative(bytes_, "initial physical bytes")
        _validate_unique_pairs(self.device_capacity_bytes, "device_capacity_bytes")
        for device_id, bytes_ in self.device_capacity_bytes:
            if not device_id or device_id.strip() != device_id:
                raise ValueError(
                    "admission capacity device IDs must be normalized strings"
                )
            _require_positive(bytes_, "admission device capacity bytes")
        if self.device_capacity_bytes and {
            item[0] for item in self.device_capacity_bytes
        } != {item[0] for item in self.initial_physical_bytes}:
            raise ValueError(
                "admission device capacities and initial physical bytes must "
                "name the same devices"
            )
        _validate_unique_values(
            tuple(item.task_id for item in self.task_deltas), "task_deltas"
        )
        _validate_unique_values(
            tuple(item.action_index for item in self.action_deltas), "action_deltas"
        )


def _validate_unique_pairs(
    values: tuple[tuple[str, int], ...],
    field: str,
) -> None:
    _validate_unique_values(tuple(item[0] for item in values), field)


def _validate_unique_values(values: tuple[object, ...], field: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field} contains duplicate identities")


class TransferDirection(StrEnum):
    """Which way a simulated copy moves, and so which lane it occupies.

    Attributes
    ----------
    FETCH : str
        Spill to device, on the fetch lane.
    EVICT : str
        Device to spill, on the evict lane, which evictions and write-backs
        share.
    """

    FETCH = "fetch"
    EVICT = "evict"


@dataclass(frozen=True, slots=True)
class TaskInterval:
    """When one task ran in a simulation, and what it waited for.

    `SimulationResult.task_intervals` holds one per task that ran.

    Attributes
    ----------
    task_id : str
        The task.
    device_id : str
        The device it ran on.
    resource_kind : ResourceKind
        The kind of resource it ran on.
    resource_lane : int
        Its lane of that resource; tasks on one lane run one at a time, in
        program order.
    ready_ns : int
        When it could have started, in nanoseconds of simulated time: its
        dependencies had completed and it was next on its lane.
    start_ns : int
        When it started, in nanoseconds.
    end_ns : int
        When it completed, in nanoseconds.
    workspace_bytes : int
        Workspace it held while it ran, in bytes.
    stall_reasons : tuple of str
        What it waited on, empty when nothing: ``"input-residency"`` (an
        input was not yet on the device), ``"device-capacity"`` (no room for
        its outputs and workspace), ``"memory-reuse"`` (a range it reuses was
        not yet freed) or ``"transfer-resource"`` (a phase waited for its
        queue).
    """

    task_id: str
    device_id: str
    resource_kind: ResourceKind
    resource_lane: int
    ready_ns: int
    start_ns: int
    end_ns: int
    workspace_bytes: int
    stall_reasons: tuple[str, ...] = ()

    @property
    def stall_ns(self) -> int:
        """Nanoseconds from ``ready_ns`` to ``start_ns``."""
        return self.start_ns - self.ready_ns


@dataclass(frozen=True, slots=True)
class TransferInterval:
    """When one copy moved in a simulation, and what it waited for.

    `SimulationResult.transfer_intervals` holds one per fetch, eviction and
    write-back that ran.

    Attributes
    ----------
    alias_group_id : str
        The alias group copied.
    trigger_task_id : str
        The task whose boundary triggered the action.
    device_id : str
        The device it moved to or from.
    direction : TransferDirection
        Which lane it occupied.
    kind : MemoryActionKind
        The action that issued it, which tells a write-back from an eviction
        on the evict lane.
    sequence : int
        Its submission order among copies in the same direction on the same
        device, from zero.
    ready_ns : int
        When the action was first submitted, at its trigger, in nanoseconds
        of simulated time.
    start_ns : int
        When its bytes started moving, in nanoseconds.
    end_ns : int
        When it completed, in nanoseconds.
    bytes : int
        Bytes moved.
    stall_reasons : tuple of str
        What it waited on, empty when nothing: ``"device-capacity"`` (no room
        for a fetch to land), ``"host-capacity"`` (no room in the spill
        pool), ``"source-readiness"`` (its source copy was not ready),
        ``"memory-reuse"`` (the range it lands in was not yet freed),
        ``"lane-busy"`` (the lane was carrying another copy) or
        ``"transfer-resource"`` (a timed phase waited for its resource).
    """

    alias_group_id: str
    trigger_task_id: str
    device_id: str
    direction: TransferDirection
    # The action behind the copy; a write-back shares the evict lane with
    # evictions and is told apart by this.
    kind: MemoryActionKind
    sequence: int
    ready_ns: int
    start_ns: int
    end_ns: int
    bytes: int
    stall_reasons: tuple[str, ...] = ()

    @property
    def stall_ns(self) -> int:
        """Nanoseconds from ``ready_ns`` to ``start_ns``."""
        return self.start_ns - self.ready_ns


@dataclass(frozen=True, slots=True)
class DeviceMemoryPeak:
    """The peak memory use of one device over a simulation.

    `SimulationResult.device_peaks` holds one per device. Each field is its
    own maximum, so the three need not occur at the same instant.

    Attributes
    ----------
    device_id : str
        The device.
    object_bytes : int
        Peak bytes of resident objects, shared residency included.
    workspace_bytes : int
        Peak bytes of task workspace.
    total_bytes : int
        Peak bytes in use, shared residency included: objects and workspace,
        or the physical pool use when the simulation was given a
        `SimulationAdmission`.
    """

    device_id: str
    object_bytes: int
    workspace_bytes: int
    total_bytes: int


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Memory use at one instant of a simulation.

    `SimulationResult.memory_timeline` holds these; `simulate` records none.

    Attributes
    ----------
    time_ns : int
        The instant, in nanoseconds of simulated time.
    device_object_bytes : tuple of (str, int)
        Bytes of resident objects on each device, as ``(device_id, bytes)``
        pairs.
    device_workspace_bytes : tuple of (str, int)
        Bytes of task workspace on each device, as ``(device_id, bytes)``
        pairs.
    spill_bytes : int
        Bytes in spill.
    device_physical_bytes : tuple of (str, int)
        Physical pool bytes in use on each device, as ``(device_id, bytes)``
        pairs; empty when not recorded.
    """

    time_ns: int
    device_object_bytes: tuple[tuple[str, int], ...]
    device_workspace_bytes: tuple[tuple[str, int], ...]
    spill_bytes: int
    device_physical_bytes: tuple[tuple[str, int], ...] = ()


class SimulationInfeasibleError(ValueError):
    """Raised when the simulator cannot run a schedule to completion.

    `simulate` raises it for an action whose preconditions fail, a
    deadlock, a final residency not reached, or memory that never fits.
    Each argument is kept as the attribute of the same name.

    Parameters
    ----------
    message : str
        The description.
    kind : str
        What failed, as a stable word such as ``"task-input-deadlock"``,
        ``"invalid-fetch"``, ``"final-residency"`` or
        ``"fetch-device-capacity"``.
    time_ns : int
        When, in nanoseconds of simulated time.
    task_id : str, optional
        The task involved, if any.
    alias_group_ids : tuple of str, default ()
        The alias groups involved.
    location : str, optional
        The memory involved: ``"device:<device_id>"``, or ``"host"`` for
        spill. None when the failure names no memory.
    capacity_bytes : int, optional
        That memory's capacity, in bytes, for a capacity failure.
    used_bytes : int, optional
        Bytes in use there, for a capacity failure.
    requested_bytes : int, optional
        Bytes the failing request needed, for a capacity failure.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        time_ns: int,
        task_id: str | None = None,
        alias_group_ids: tuple[str, ...] = (),
        location: str | None = None,
        capacity_bytes: int | None = None,
        used_bytes: int | None = None,
        requested_bytes: int | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.time_ns = time_ns
        self.task_id = task_id
        self.alias_group_ids = alias_group_ids
        self.location = location
        self.capacity_bytes = capacity_bytes
        self.used_bytes = used_bytes
        self.requested_bytes = requested_bytes


@dataclass(frozen=True, slots=True)
class CapacityViolation:
    """One instant a simulated plan wanted more memory than its capacity.

    A fetch or task launch that does not fit waits for room, so a violation
    is stall time, not a rejection. `SimulationResult.capacity_violations`
    holds these, one per shortfall, at its first refusal.

    Attributes
    ----------
    reason : str
        What asked for too much: ``"initial-device-capacity"`` or
        ``"initial-spill-capacity"`` (over capacity before the first task),
        ``"fetch-device-capacity"`` (a
        fetch), ``"evict-spill-capacity"`` (an eviction or write-back) or
        ``"task-device-capacity"`` (a task launch).
    location : str
        The memory that came up short: ``"device"`` or ``"spill"``.
    time_ns : int
        When, in nanoseconds of simulated time.
    capacity_bytes : int
        That memory's capacity, in bytes.
    used_bytes : int
        Bytes in use there at that instant.
    requested_bytes : int
        Bytes the request needed.
    device_id : str
        The device involved, or ``"unknown"`` when none is.
    task_id : str or None
        The task involved, if any.
    alias_group_id : str or None
        The alias group involved, if any.
    """

    reason: str
    location: str
    time_ns: int
    capacity_bytes: int
    used_bytes: int
    requested_bytes: int
    device_id: str
    task_id: str | None = None
    alias_group_id: str | None = None

    @property
    def excess_bytes(self) -> int:
        """Bytes past the capacity at this instant, at least zero."""

        return max(0, self.used_bytes + self.requested_bytes - self.capacity_bytes)


# Not slotted, so the simulator can attach its own interval arrays
# without declaring a field. They are not data - `asdict` and every other
# field walk must not see them, or writing a result out would fail and its
# digest would change.
@dataclass(frozen=True)
class SimulationResult:
    """What one simulated schedule cost.

    `simulate` returns it.

    Attributes
    ----------
    makespan_ns : int
        Simulated time until the last task or copy completed, in
        nanoseconds.
    task_intervals : tuple of TaskInterval
        One per task that ran, in program order.
    transfer_intervals : tuple of TransferInterval
        One per copy, by start time.
    device_peaks : tuple of DeviceMemoryPeak
        One per device, in the program's device order.
    spill_peak_bytes : int
        Peak spill use, in bytes, shared residency included.
    spill_transient_peak_bytes : int
        Peak spill held by transient leases, in bytes. A lease is persistent
        when the initial or final residency places its alias in spill or the
        alias retains its spill copy; every other lease is transient. Shared
        residency is persistent.
    spill_transient_end_bytes : int
        Spill still held by transient leases when the schedule ends, in
        bytes.
    memory_timeline : tuple of MemorySnapshot
        Memory use over time; empty from `simulate`.
    capacity_violations : tuple of CapacityViolation
        The shortfalls the plan waited on.
    capacity_violation_count : int
        How many shortfalls there were. Larger than the length of
        ``capacity_violations`` when that list was truncated.
    """

    makespan_ns: int
    task_intervals: tuple[TaskInterval, ...]
    transfer_intervals: tuple[TransferInterval, ...]
    device_peaks: tuple[DeviceMemoryPeak, ...]
    spill_peak_bytes: int
    spill_transient_peak_bytes: int
    spill_transient_end_bytes: int
    memory_timeline: tuple[MemorySnapshot, ...] = ()
    #: Where the plan came up short and waited. Truncation is visible as
    #: `capacity_violation_count` exceeding the length of this tuple.
    capacity_violations: tuple[CapacityViolation, ...] = ()
    capacity_violation_count: int = 0

    @property
    def interval_arrays(self) -> object | None:
        """The simulator's own interval arrays, or None.

        Set on a result `simulate` produced, for a consumer that works by
        index rather than through the interval records; they hold nothing
        the records do not. None on any other result, on a copy made with
        ``dataclasses.replace``, and once the result is pickled.
        """

        return self.__dict__.get("_interval_arrays")

    def attach_interval_arrays(self, value: object) -> None:
        """Attach the arrays this result was decoded from.

        The simulator calls it; `interval_arrays` returns ``value`` after.

        Parameters
        ----------
        value : object
            The arrays.
        """

        object.__setattr__(self, "_interval_arrays", value)

    def __getstate__(self) -> dict[str, object]:
        """Leave the arrays behind: they address library memory, so they mean
        nothing once this result is written out or moved to another process.
        A restored result reports `None`, exactly as an uncompiled one does."""

        state = dict(self.__dict__)
        state.pop("_interval_arrays", None)
        return state

    def device_peak(self, device_id: str) -> DeviceMemoryPeak:
        """Find one device's peak memory use.

        Parameters
        ----------
        device_id : str
            The device.

        Returns
        -------
        DeviceMemoryPeak
            Its entry in ``device_peaks``.

        Raises
        ------
        KeyError
            If the result has no such device.
        """
        for peak in self.device_peaks:
            if peak.device_id == device_id:
                return peak
        raise KeyError(device_id)


__all__ = [
    "ActionPhysicalDelta",
    "DeviceMemoryPeak",
    "DeviceSimulationConfig",
    "MemoryReuseDependency",
    "MemorySnapshot",
    "SimulationAdmission",
    "SimulationConfig",
    "SimulationInfeasibleError",
    "SimulationResult",
    "TaskInterval",
    "TaskPhysicalDelta",
    "TransferDirection",
    "TransferInterval",
]

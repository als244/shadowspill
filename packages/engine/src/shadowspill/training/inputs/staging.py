"""CPU proofs of complete-call input storage and serial physical staging bounds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import Any, Self

from shadowspill.parallelism import InputPlacement
from shadowspill_client import BatchConfig, Execution
from shadowspill_client.buffers import BufferBudget, BufferLease
from shadowspill_client.protocol.inputs import DTYPES, InputSchema
from shadowspill_client.protocol.payload_sizes import (
    MAX_TREE_DEPTH,
    TREE_NODE_BYTES,
    PayloadSize,
    manifest_json,
    metadata_json,
)
from shadowspill_client.protocol.tensors import storage_dtype

from .adapters import InputBinding
from .coverage_bounds import CoverageError, input_bounds
from .materialize import buffer_size
from .shapes import PackingSchedule
from .validation import Path
from .values import VALIDATION_SCRATCH_BYTES


class StagingLimitError(CoverageError):
    """A known candidate cannot fit the explicitly configured host bound."""


@dataclass(frozen=True)
class InputEnvelope:
    max_datums: int
    snapshot: PayloadSize


def input_envelope(
    schema: InputSchema, batch: BatchConfig, required_paths: Sequence[Path]
) -> InputEnvelope:
    """Bound every valid supplied tree, including known but unused leaves.

    A limit on an optional, unused leaf cannot bound a different supplied leaf
    with the same symbolic axis: the bounded leaf may be omitted. For each leaf,
    use only its own capacities and those guaranteed by the selected closure.
    Source tensors may use any accepted portable dtype, before deferred casts.
    No arrays or per-Datum objects are constructed while deriving the envelope.
    """
    definitions = {tuple(field.path): field for field in schema.fields}
    selected = dict.fromkeys(required_paths)
    count = input_bounds(schema, batch, selected).records
    trees: dict[str, Any] = {"inputs": {}, "loss_inputs": {}}
    nodes: set[Path] = {("inputs",), ("loss_inputs",)}
    edges: set[Path] = set()
    layouts = []
    largest = total = 0
    width = max(storage_dtype(dtype).itemsize for dtype in DTYPES)
    longest_dtype = max(DTYPES, key=len)
    largest_index = max(0, count * len(definitions) - 1)
    for path, field in definitions.items():
        if len(path) < 2 or path[0] not in trees:
            raise ValueError("input envelope requires combined model/loss paths")
        if len(path) - 1 > MAX_TREE_DEPTH or len(field.shape) > 32:
            raise CoverageError(
                f"field exceeds binary payload depth/rank limits: {path}"
            )
        bounds = input_bounds(schema, batch, {**selected, path: None})
        shape, summed = bounds.individual[path], bounds.total[path]
        elements = prod(shape)
        # An axis sum can tighten the product envelope. All alternatives are
        # upper bounds; no independence between aligned axes is assumed.
        totals = [bounds.records * elements]
        totals.extend(
            length * prod(shape[:axis] + shape[axis + 1 :])
            for axis, length in enumerate(summed)
        )
        total += width * min(totals)
        largest = max(largest, width * elements if bounds.records else 0)
        layouts.append({"dtype": longest_dtype, "shape": shape})
        tree = trees[path[0]]
        for i, key in enumerate(path[1:], 1):
            prefix = path[: i + 1]
            nodes.add(prefix)
            edges.add(prefix)
            if i == len(path) - 1:
                tree[key] = largest_index
            else:
                tree = tree.setdefault(key, {})
    # Repeating an upper-bound Datum/layout table needs only arithmetic. JSON
    # escaping, index digits, scalar/empty shapes and separators all count.
    header = (
        len(manifest_json([], []))
        + count * len(metadata_json(trees))
        + max(0, count - 1)
        + count * sum(len(metadata_json(layout)) for layout in layouts)
        + max(0, count * len(layouts) - 1)
    )
    tree_size = count * (
        TREE_NODE_BYTES * len(nodes)
        + sum(len(path[-1].encode("utf-8")) for path in edges)
    )
    try:
        snapshot = PayloadSize(total, header, tree_size, largest)
    except ValueError as error:
        raise CoverageError(
            f"advertised input metadata cannot fit the binary protocol: {error}"
        ) from error
    return InputEnvelope(count, snapshot)


class _ArenaBudget(BufferBudget):
    def reserve(self, size: int, *, timeout: float | None = None) -> BufferLease:
        # Backpressure occurs once, before any bytes of this call are owned.
        # Exceeding its admitted arena is an error, never a wait on itself.
        return super().reserve(size, timeout=0)


class StagingReservation:
    """One queue slot owns room for the complete input pipeline before copying.

    The inner budget accounts snapshot, validation/packing workspace, one packed
    buffer and one worker adapter. Stream physical buffers and cooperating
    workers serially; additional overlap requires separately admitted storage.
    The outer lease is released only after every inner owner has closed.
    """

    def __init__(
        self, budget: BufferBudget, size: int, *, timeout: float | None = None
    ) -> None:
        self._lease = budget.reserve(size, timeout=timeout)
        self.budget = _ArenaBudget(size, max_requests=5)

    def close(self) -> None:
        self.budget.close()
        if self.budget.used != (0, 0):
            raise RuntimeError("staging reservation still has live input owners")
        self._lease.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class InputStaging:
    envelope: InputEnvelope
    workspace_bytes: int
    buffer_peaks: tuple[int, ...]
    spool_bytes: int

    @property
    def staging_bytes(self) -> int:
        return max(
            self.envelope.snapshot.staging,
            self.envelope.snapshot.retained
            + self.workspace_bytes
            + max(VALIDATION_SCRATCH_BYTES, *self.buffer_peaks),
        )

    def reserve(
        self, budget: BufferBudget, *, timeout: float | None = None
    ) -> StagingReservation:
        return StagingReservation(budget, self.staging_bytes, timeout=timeout)


def admit_input_staging(
    schema: InputSchema,
    batch: BatchConfig,
    placement: InputPlacement,
    schedule: PackingSchedule,
    bindings: Mapping[tuple[int, int], InputBinding],
    execution: Execution,
) -> InputStaging:
    """Admit serial host staging separately from geometry and native pools.

    Every (buffer index, logical rank) adapter must have the exact prepared ABI.
    Count replicated input data once, but include the largest worker's declared
    generated metadata. This is CPU accounting, not measured RSS, GPU/spill
    admission, collective correctness or a proof of arbitrary model callbacks.
    """
    schedule.validate(schema, batch, placement)
    buffers = schedule.buffers
    envelope = input_envelope(schema, batch, tuple(buffers[0].fields))
    expected = {
        (index, rank)
        for index, (partition, _) in enumerate(schedule.locations)
        for rank in placement.partitions.groups[partition]
    }
    if bindings.keys() != expected:
        raise ValueError("input staging needs exactly every buffer/worker adapter")
    peaks = []
    for index, physical in enumerate(buffers):
        size = buffer_size(
            physical, schema, min(physical.max_datums, envelope.max_datums)
        )
        peak = size.staging
        partition, _ = schedule.locations[index]
        for rank in placement.partitions.groups[partition]:
            binding = bindings[index, rank]
            if (
                binding.layout.schema != schema
                or binding.layout.buffer != physical
                or binding.layout.placement != placement
                or binding.layout.logical_rank != rank
            ):
                raise ValueError("input staging adapter differs from its prepared ABI")
            peak = max(
                peak,
                size.retained + binding.staging_bytes,
                size.retained + binding.retained_bytes + binding.frame.write_bytes,
                binding.frame.size.staging,
            )
        peaks.append(peak)
    # Conservative container/shape allowances for complete validation plus the
    # finite packer's Datum x buffer compatibility matrix and group bookkeeping.
    # Reserve before validation/packing; keep it until input consumption ends.
    workspace = (
        512
        * (envelope.max_datums + 1)
        * (
            len(buffers)
            + 1
            + sum(len(f.path) + len(f.shape) + 2 for f in schema.fields)
        )
    )
    workspace += 512 * len(bindings)
    result = InputStaging(
        envelope,
        workspace,
        tuple(peaks),
        sum(binding.frame.size.wire for binding in bindings.values()),
    )
    if result.staging_bytes > execution.max_staging_bytes:
        raise StagingLimitError(
            f"{schedule.name}: input staging needs {result.staging_bytes} bytes; "
            f"configured max_staging_bytes is {execution.max_staging_bytes}"
        )
    return result

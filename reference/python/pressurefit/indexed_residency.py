"""Indexed projection for the PressureFit residency reducer."""

from __future__ import annotations

import ctypes
from array import array
from dataclasses import dataclass
from typing import Any

from shadowspill.ir import MemoryLocation
from shadowspill.planner.capi import (
    NO_INDEX,
    CResidencyOptions,
    CResidencyProblem,
    CResidencyResult,
    planner_api,
)
from shadowspill.planner.result import PressureFitInfeasibleError
from shadowspill.simulator import SimulationConfig
from shadowspill.status import ABI_VERSION, Status

from .facts import PlanningFacts
from .residency import ResidencyPlan, Span


def _array(ctype: Any, values: list[int]) -> Any:
    array_type = ctype * max(1, len(values))
    if not values:
        return array_type()
    if ctype is ctypes.c_uint8:
        payload: object = bytes(values)
    elif ctype is ctypes.c_int8:
        payload = array("b", values)
    elif ctype is ctypes.c_uint32:
        payload = array("I", values)
    elif ctype is ctypes.c_uint64:
        payload = array("Q", values)
    else:  # pragma: no cover - private helper accepts only the cases above.
        raise TypeError(f"unsupported indexed-array type: {ctype!r}")
    return array_type.from_buffer_copy(payload)


@dataclass(frozen=True, slots=True)
class CompiledResidencyTemplate:
    """Caller-owned buffers storing one immutable C residency problem."""

    problem: CResidencyProblem
    buffers: tuple[object, ...]
    facts: PlanningFacts
    device_ids: tuple[str, ...]
    seed: ResidencyPlan
    seed_resident: Any
    seed_breaks: Any


def index_residency_template(
    facts: PlanningFacts,
    config: SimulationConfig,
    seed: ResidencyPlan,
) -> CompiledResidencyTemplate:
    alias_count = len(facts.alias_ids)
    boundary_count = len(facts.tasks) + 1
    device_ids = tuple(facts.object_capacity_by_device)
    device_index = {device_id: index for index, device_id in enumerate(device_ids)}
    configured = {device.device_id: device for device in config.devices}
    cells = alias_count * boundary_count

    anchors = [0] * cells
    productions = [0] * cells
    latest_access = [NO_INDEX] * cells
    output_reservations = [0] * cells
    write_prefix = [0] * cells
    for alias in range(alias_count):
        row = alias * boundary_count
        for boundary in facts.anchors[alias]:
            anchors[row + boundary + 1] = 1
        for boundary in facts.production_boundaries[alias]:
            productions[row + boundary + 1] = 1
        for boundary, task in facts.access_events[alias]:
            position = row + boundary + 1
            previous = latest_access[position]
            latest_access[position] = (
                task if previous == NO_INDEX else max(previous, task)
            )
        writes = set(facts.write_boundaries[alias])
        seen_write = False
        for boundary in range(-1, facts.last_boundary + 1):
            seen_write = seen_write or boundary in writes
            write_prefix[row + boundary + 1] = int(seen_write)
    for task, aliases in enumerate(facts.output_reservations):
        for alias in aliases:
            output_reservations[alias * boundary_count + task] = 1

    device_priority_by_id = {
        device_id: rank for rank, device_id in enumerate(sorted(device_ids))
    }
    fetch_runtime: list[int] = []
    evict_runtime: list[int] = []
    for alias, size in enumerate(facts.alias_sizes):
        device = configured[facts.alias_devices[alias]]
        fetch_runtime.append(
            device.fetch_latency_ns
            + (size * 1_000_000_000 + device.fetch_bandwidth_bytes_per_second - 1)
            // device.fetch_bandwidth_bytes_per_second
        )
        evict_runtime.append(
            device.evict_latency_ns
            + (size * 1_000_000_000 + device.evict_bandwidth_bytes_per_second - 1)
            // device.evict_bandwidth_bytes_per_second
        )

    buffers: list[object] = []

    def keep(value: object) -> object:
        buffers.append(value)
        return value

    alias_sizes = keep(_array(ctypes.c_uint64, list(facts.alias_sizes)))
    alias_devices = keep(
        _array(
            ctypes.c_uint32,
            [device_index[device_id] for device_id in facts.alias_devices],
        )
    )
    retain_spill = keep(
        _array(ctypes.c_uint8, [int(value) for value in facts.alias_retain_spill_copy])
    )
    location_code = {
        None: -1,
        MemoryLocation.DEVICE: 0,
        MemoryLocation.SPILL: 1,
    }
    initial = keep(
        _array(
            ctypes.c_int8,
            [location_code[value] for value in facts.initial_locations],
        )
    )
    final = keep(
        _array(
            ctypes.c_int8,
            [location_code[value] for value in facts.final_locations],
        )
    )
    anchor_buffer = keep(_array(ctypes.c_uint8, anchors))
    production_buffer = keep(_array(ctypes.c_uint8, productions))
    access_buffer = keep(_array(ctypes.c_uint32, latest_access))
    reservation_buffer = keep(_array(ctypes.c_uint8, output_reservations))
    write_buffer = keep(_array(ctypes.c_uint8, write_prefix))
    first_input = keep(_array(ctypes.c_uint32, list(facts.first_input_tasks)))
    fetch_buffer = keep(_array(ctypes.c_uint64, fetch_runtime))
    evict_buffer = keep(_array(ctypes.c_uint64, evict_runtime))
    task_ends = keep(_array(ctypes.c_uint64, list(facts.task_ideal_end_ns)))
    capacities = keep(
        _array(
            ctypes.c_uint64,
            [facts.object_capacity_by_device[value] for value in device_ids],
        )
    )
    boundary_capacities = keep(
        _array(
            ctypes.c_uint64,
            [
                facts.object_capacity_by_boundary[device_id][boundary]
                for device_id in device_ids
                for boundary in range(boundary_count)
            ],
        )
    )
    priorities = keep(
        _array(
            ctypes.c_uint32,
            [device_priority_by_id[value] for value in device_ids],
        )
    )
    seed_resident_values, seed_break_values = _project_plan(
        seed,
        alias_count=alias_count,
        boundary_count=boundary_count,
    )
    seed_resident = keep(_array(ctypes.c_uint8, seed_resident_values))
    seed_breaks = keep(_array(ctypes.c_uint8, seed_break_values))
    problem = CResidencyProblem(
        abi_version=ABI_VERSION,
        alias_count=alias_count,
        boundary_count=boundary_count,
        device_count=len(device_ids),
        alias_size_bytes=alias_sizes,
        alias_device=alias_devices,
        alias_retain_spill_copy=retain_spill,
        initial_location=initial,
        final_location=final,
        anchors=anchor_buffer,
        productions=production_buffer,
        latest_access_task=access_buffer,
        output_reservations=reservation_buffer,
        write_prefix=write_buffer,
        first_input_task=first_input,
        fetch_runtime_ns=fetch_buffer,
        evict_runtime_ns=evict_buffer,
        task_ideal_end_ns=task_ends,
        device_capacity_bytes=capacities,
        boundary_capacity_bytes=boundary_capacities,
        device_priority=priorities,
    )
    return CompiledResidencyTemplate(
        problem,
        tuple(buffers),
        facts,
        device_ids,
        seed,
        seed_resident,
        seed_breaks,
    )


def _project_plan(
    plan: ResidencyPlan,
    *,
    alias_count: int,
    boundary_count: int,
) -> tuple[list[int], list[int]]:
    resident = [0] * (alias_count * boundary_count)
    breaks = [0] * (alias_count * boundary_count)
    for alias, spans in enumerate(plan.spans):
        row = alias * boundary_count
        for index, span in enumerate(spans):
            for boundary in range(span.start, span.end + 1):
                resident[row + boundary + 1] = 1
            if index + 1 < len(spans):
                breaks[row + span.end + 1] = 1
    return resident, breaks


def _decode_plan(
    template: CompiledResidencyTemplate,
    anchors: tuple[frozenset[int], ...],
    resident: Any,
    breaks: Any,
) -> ResidencyPlan:
    count = int(template.problem.boundary_count)
    result: list[tuple[Span, ...]] = []
    for alias in range(int(template.problem.alias_count)):
        row = alias * count
        spans: list[Span] = []
        index = 0
        while index < count:
            if not resident[row + index]:
                index += 1
                continue
            start = index
            while (
                index + 1 < count
                and resident[row + index + 1]
                and not breaks[row + index]
            ):
                index += 1
            spans.append(Span(start - 1, index - 1))
            index += 1
        result.append(tuple(spans))
    return ResidencyPlan(tuple(result), anchors)


def reduce_residency(
    template: CompiledResidencyTemplate,
    seed: ResidencyPlan,
    strategy: str,
    *,
    extra_pressure: dict[tuple[str, int], int] | None = None,
) -> ResidencyPlan:
    alias_count = int(template.problem.alias_count)
    boundary_count = int(template.problem.boundary_count)
    if seed != template.seed:
        raise ValueError("the residency template received a different seed")
    cell_count = alias_count * boundary_count
    extra_count = len(template.device_ids) * boundary_count
    extra_buffer = (ctypes.c_uint64 * max(1, extra_count))()
    device_index = {
        device_id: index for index, device_id in enumerate(template.device_ids)
    }
    for (device_id, boundary), value in (extra_pressure or {}).items():
        extra_buffer[device_index[device_id] * boundary_count + boundary + 1] = value
    output_resident = (ctypes.c_uint8 * max(1, cell_count))()
    output_breaks = (ctypes.c_uint8 * max(1, cell_count))()
    options = CResidencyOptions(
        minimize_transfer=int(strategy.endswith("transfer")),
        fetch_headroom=int(strategy.startswith("headroom")),
        seed_resident=template.seed_resident,
        seed_breaks=template.seed_breaks,
        extra_pressure_bytes=extra_buffer,
    )
    result = CResidencyResult(
        resident=output_resident,
        resident_capacity=cell_count,
        breaks=output_breaks,
        break_capacity=cell_count,
    )
    status = int(
        planner_api().shadowspill_reduce_residency(
            ctypes.byref(template.problem),
            ctypes.byref(options),
            ctypes.byref(result),
        )
    )
    if status == Status.ANALYTIC_INFEASIBLE:
        boundary = int(result.error_boundary)
        task_index = boundary + 1
        task_id = (
            template.facts.tasks[task_index].task_id
            if 0 <= task_index < len(template.facts.tasks)
            else None
        )
        device_id = template.device_ids[int(result.error_device)]
        raise PressureFitInfeasibleError(
            f"no legal residency cut can relieve {int(result.required_bytes)} "
            f"bytes at boundary {boundary} on {device_id!r}; capacity is "
            f"{int(result.capacity_bytes)}",
            kind="analytic_capacity",
            device_id=device_id,
            boundary_task_id=task_id,
            required_bytes=int(result.required_bytes),
            capacity_bytes=int(result.capacity_bytes),
        )
    if status != 0:
        encoded = planner_api().shadowspill_planner_status_string(status)
        message = encoded.decode("utf-8") if encoded else f"planner status {status}"
        raise RuntimeError(message)
    return _decode_plan(template, seed.anchors, output_resident, output_breaks)


__all__ = [
    "CompiledResidencyTemplate",
    "index_residency_template",
    "reduce_residency",
]

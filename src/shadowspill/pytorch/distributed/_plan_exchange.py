"""Exchange verified planning problems and portable schedules between ranks.

No compiled code, runtime tensors, or allocator bindings travel here. Received
schedules are reconstructed and admitted against each receiving rank's facts.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace

from shadowspill.planner.admission.refinement import FixedLayoutSelection
from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.plan_store import certified_result
from shadowspill.planner.program_inputs import MemoryBudgets, TransferBandwidths
from shadowspill.planner.serialization import _simulation_config_to_dict
from shadowspill.store import atomic_text

from . import current
from ._symmetry import Symmetry, _digest, verify


def shared_problem(
    problem, execution_budget, spill_budget, transfer, *, extra=None, incumbent=None
):
    bound = current()
    assert bound is not None
    config, facts = bound.control.run(
        "symmetry/machine_contract",
        lambda: problem.machine_inputs(
            execution_budget_bytes=execution_budget,
            spill_budget_bytes=spill_budget,
            transfer_bandwidths=transfer,
        ),
    )
    shared, evidence = verify(
        Symmetry(
            problem.program,
            config,
            facts,
            problem.initial_residency,
            problem.final_residency,
            problem.dynamic_scratch_reserve_bytes,
        ),
        extra=extra,
        incumbent=incumbent,
    )
    if shared is None:
        return problem, transfer, None, evidence
    device = shared.config.devices[0]
    transfer = TransferBandwidths(
        device.fetch_solo_bandwidth_bytes_per_second,
        device.fetch_concurrent_bandwidth_bytes_per_second,
        device.evict_solo_bandwidth_bytes_per_second,
        device.evict_concurrent_bandwidth_bytes_per_second,
        fetch_latency_ns=device.fetch_latency_ns,
        evict_latency_ns=device.evict_latency_ns,
    )
    return replace(problem, program=shared.program), transfer, shared, evidence


def received_plan(shared, payload, problem, budgets, transfer):
    started = time.perf_counter_ns()
    local = shared.receive(payload)
    return AnnotatedProgramPlan(
        program=problem,
        memory_budgets=MemoryBudgets(*budgets),
        transfer_bandwidths=transfer or problem.transfer_bandwidths,
        result=local.result,
        effective_facts=local.facts,
        fixed_layout=local.admission.layout,
        simulation_admission=local.admission.simulator_input,
        simulation=local.admission.simulation,
        attempts=local.attempts,
        plan_from_store=False,
        wall_time_ns=time.perf_counter_ns() - started,
    )


def pack(record: FixedLayoutSelection | AnnotatedProgramPlan) -> dict:
    """Portable logical schedules and evidence, without compiled code or bindings."""
    result = (
        certified_result(record.result, record.admission)
        if isinstance(record, FixedLayoutSelection)
        else record.result
    )
    facts = (
        record.facts
        if isinstance(record, FixedLayoutSelection)
        else record.effective_facts
    )
    layout = (
        record.admission.layout
        if isinstance(record, FixedLayoutSelection)
        else record.fixed_layout
    )
    return {
        "schedule": result.schedule.to_dict(),
        "choices": [item.to_dict() for item in result.selections],
        "config": _simulation_config_to_dict(result.simulation_config),
        "options": result.search_options.to_dict(),
        "simulation": asdict(result.simulation),
        "diagnostics": result.diagnostics.to_dict(),
        "resident_slice": result.resident_slice.to_dict(),
        "facts": facts.to_dict(),
        "scratch": layout.scratch_reserve_bytes,
        "resolutions": [
            {
                "selection_id": row.selection_id,
                "candidate_id": row.candidate_id,
                "schedule": row.schedule.to_dict(),
                "simulation": asdict(row.simulation),
                "choices": [item.to_dict() for item in row.selections],
                "resident_slice": row.resident_slice.to_dict(),
            }
            for row in result.resolutions
        ],
    }


def record_plan(store, record) -> str | None:
    """Keep received schedules, including every retained resolution, per rank."""
    if not store.plan_policy.write_enabled:
        return None
    program = record.result.program
    store.archive_program(program)
    payload = {"version": 1, "program": program.digest, "plan": pack(record)}
    identity = _digest(payload)
    path = store.planning / "distributed" / "plans" / f"{identity}.json"
    atomic_text(path, json.dumps(payload, sort_keys=True, separators=(",", ":")))
    store.record(
        category="planning",
        kind="distributed_plan",
        digest=identity,
        path=path,
        access="write",
        schema="distributed_plan/v1",
        dependencies=(program.digest,),
    )
    return str(path)

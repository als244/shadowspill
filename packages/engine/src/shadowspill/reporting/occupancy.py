"""Use the detailed occupancy renderer with the public, saved Plan artifacts."""

from collections.abc import Mapping
from typing import Any

from shadowspill.artifacts._values import plain
from shadowspill.artifacts.plan import Plan
from shadowspill.artifacts.task_names import profile_names, task_parts
from shadowspill.diagnostics.occupancy import attribute
from shadowspill.diagnostics.occupancy.alternatives import alternatives_data
from shadowspill.diagnostics.occupancy.page import page_data
from shadowspill.planning.serialization.simulation_result import simulation_result_value
from shadowspill.planning.training.order import coordinates


def occupancy_data(
    plan: Plan,
    *,
    description: str = "",
    diagnostics: Mapping[str, Any] | None = None,
    tokens_per_step: int | None = None,
) -> dict[str, Any]:
    """Attribute recorded schedules and physical leases; never simulate again.

    ``tokens_per_step`` is the step's token count for the summary's rate: what
    the Plan schedules for a predicted view, what the request supplied for a
    traced one.
    """
    communication = plan.communication_info
    rates = {}
    for direction in ("fetch", "evict"):
        solo = getattr(communication, f"{direction}_bytes_per_second")
        concurrent = getattr(communication, f"{direction}_concurrent_bytes_per_second")
        rates[f"{direction}_solo_bandwidth_bytes_per_second"] = solo
        rates[f"{direction}_concurrent_bandwidth_bytes_per_second"] = (
            solo if concurrent is None else concurrent
        )
        rates[f"{direction}_latency_ns"] = (
            getattr(communication, f"{direction}_latency_seconds") * 1e9
        )
    program = plan._costed_program.to_dict()
    result = attribute(
        {
            "schedule": plan.schedule.to_dict(),
            "selections": [choice.to_dict() for choice in plan.selections],
            "simulation_result": simulation_result_value(plan.timeline),
            "simulation": {
                "spill_capacity_bytes": plan.memory_budgets.spill_bytes,
                "devices": [rates],
            },
            "admission_certificate": {
                "layout": None
                if plan.memory_layout is None
                else plan.memory_layout.to_dict()
            },
        },
        program,
        diagnostics=diagnostics,
    )
    data = page_data(
        result,
        plan=description or plan.program.name or "Plan",
        tokens_per_step=tokens_per_step,
    )
    data["alternatives"] = alternatives_data(
        program,
        [
            {"group_id": choice.group_id, "option_id": choice.option_id}
            for choice in plan.selections
        ],
        positions=_positions(plan),
        labels={key: name.task for key, name in profile_names(plan.program).items()},
        covers=task_parts(plan.program),
        measured_ns=_measured(diagnostics),
    )
    data["configuration"] = execution_configuration(plan)
    data["allocation"] = plan.summary
    data["simulation_only_reasons"] = plan.simulation_only_reasons
    return data


def execution_configuration(plan: Plan) -> dict[str, Any]:
    """Describe the ordering and execution choices saved for this worker."""
    bindings = plan.program.bindings
    template = bindings.get("preparation_template", {})
    build = template.get("build", {})
    parallel = plan.program.parallelism
    architecture = (
        template.get("model_build", {}).get("config", {}).get("architecture", {})
    )
    return plain(
        {
            "ordering": bindings.get("ordering"),
            "chunking": build.get("chunking"),
            "model_chunks": {
                name: value
                for name, value in architecture.items()
                if "chunk" in name and type(value) is int and value > 0
            },
            "retention": build.get("graph_pair_policy"),
            "replicas": parallel.num_replicas if parallel else 1,
            "ranks_per_replica": parallel.ranks_per_replica if parallel else 1,
        }
    )


def _positions(plan: Plan) -> dict[str, tuple[int, int]] | None:
    """Each alternative group's microbatch and stage, from its tasks' names."""
    names = plan.program.bindings.get("task_names", {})
    placed: dict[str, tuple[int, int]] = {}
    for group in plan.program.task_alternative_groups:
        found = {
            coordinates(names.get(task, ""))
            for option in group.options
            for task in option.active_task_ids
        }
        place = next(iter(found)) if len(found) == 1 else None
        if place is None:
            return None
        placed[group.group_id] = place
    return placed


def _measured(diagnostics: Mapping[str, Any] | None) -> dict[str, list[int]]:
    """Each task's traced compute durations, in nanoseconds."""
    seen: dict[str, list[int]] = {}
    for record in (
        {} if diagnostics is None else diagnostics.get("tasks", {})
    ).values():
        compute = record.get("compute") or {}
        start = compute.get("compute_started_at_seconds")
        end = compute.get("compute_finished_at_seconds")
        if start is None or end is None or end < start:
            continue
        seen.setdefault(record["task_id"], []).append(round((end - start) * 1e9))
    return seen

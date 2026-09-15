"""The derived numbers the artifact reports, each with the reason it exists."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from shadowspill.ir import TaskAlternativeChoice, TaskAlternativeGroup


def recomputation_savings_bytes(
    groups: Sequence[TaskAlternativeGroup],
    selections: Sequence[TaskAlternativeChoice],
    alias_sizes: Mapping[str, int],
) -> tuple[int, int]:
    """Report maximum available and selected retained-byte savings."""

    selected_by_group = {item.group_id: item.option_id for item in selections}
    available = 0
    selected = 0
    for group in groups:
        reference = next(
            (item for item in group.options if item.option_id == "save"), None
        )
        if reference is None:
            continue
        reference_bytes = sum(
            alias_sizes[alias_id]
            for alias_id in set(reference.retained_alias_group_ids)
        )
        savings = {
            item.option_id: max(
                0,
                reference_bytes
                - sum(
                    alias_sizes[alias_id]
                    for alias_id in set(item.retained_alias_group_ids)
                ),
            )
            for item in group.options
        }
        available += max(savings.values(), default=0)
        selected += savings.get(selected_by_group.get(group.group_id, "save"), 0)
    return available, selected


def planning_breakdown(
    phase_seconds: dict[str, float], *, planning_seconds: float
) -> dict[str, float]:
    """Return non-overlapping public planning phases for matrix comparisons."""

    lowering_aot = phase_seconds.get("capture_lowering", 0.0)
    profiling = phase_seconds.get(
        "unique_stage_warmup_profiling",
        phase_seconds.get("structural_profiling", 0.0),
    )
    compilation = phase_seconds.get(
        "compiled_entrypoint_construction",
        phase_seconds.get("compilation", 0.0),
    )
    cached_warmup = phase_seconds.get("cached_entrypoint_warmup", 0.0)
    profile_orchestration = phase_seconds.get(
        "profile_cache_and_entrypoint_orchestration", 0.0
    )
    program_lowering = phase_seconds.get("program_lowering", 0.0)
    search = phase_seconds.get("search", 0.0)
    admission = (
        phase_seconds.get("admission_facts", 0.0)
        + phase_seconds.get("spill_admission", 0.0)
        + phase_seconds.get("slab_admission", 0.0)
    )
    classified = (
        lowering_aot
        + profiling
        + compilation
        + cached_warmup
        + profile_orchestration
        + program_lowering
        + search
        + admission
    )
    return {
        "lowering_aot": lowering_aot,
        "profiling": profiling,
        "compiled_entrypoint_construction": compilation,
        "cached_entrypoint_warmup": cached_warmup,
        "profile_cache_and_entrypoint_orchestration": profile_orchestration,
        "canonical_program_lowering": program_lowering,
        "search": search,
        "physical_admission": admission,
        "other": max(0.0, planning_seconds - classified),
        "total": planning_seconds,
    }


def state_tensor_at_path(state: object, path: str) -> torch.Tensor:
    """Resolve one compare_states() tensor path for failure diagnostics."""

    components = path.split("/")
    if not components or components[0] != "state":
        raise ValueError(f"invalid state metric path {path!r}")
    value = state
    for component in components[1:]:
        if isinstance(value, dict):
            if component in value:
                value = value[component]
            elif component.isdecimal() and int(component) in value:
                # Optimizer state_dict() keys are integer parameter ordinals,
                # while compare_states() renders every path component as text.
                value = value[int(component)]
            else:
                raise KeyError(f"state metric path component {component!r} is absent")
        elif isinstance(value, (list, tuple)):
            value = value[int(component)]
        else:
            raise ValueError(f"state metric path stops before {component!r}")
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"state metric path {path!r} does not resolve to a tensor")
    return value


def failure_tensor_values(
    names: list[str], reference: object, actual: object
) -> dict[str, dict[str, object]]:
    """Keep bounded concrete values for failed numerical comparisons."""

    result: dict[str, dict[str, object]] = {}
    for name in names:
        expected = state_tensor_at_path(reference, name).detach().cpu().reshape(-1)
        observed = state_tensor_at_path(actual, name).detach().cpu().reshape(-1)
        limit = min(64, expected.numel())
        result[name] = {
            "numel": expected.numel(),
            "truncated": expected.numel() > limit,
            "reference": expected[:limit].tolist(),
            "actual": observed[:limit].tolist(),
        }
    return result


def optimizer_steps(checkpoint: object) -> dict[str, int]:
    if not isinstance(checkpoint, dict):
        return {}
    optimizer = checkpoint.get("optimizer")
    if not isinstance(optimizer, dict):
        return {}
    state = optimizer.get("state")
    if not isinstance(state, dict):
        return {}
    result: dict[str, int] = {}
    for parameter_id, values in state.items():
        if not isinstance(values, dict):
            continue
        step = values.get("step")
        if isinstance(step, torch.Tensor) and step.numel() == 1:
            result[str(parameter_id)] = int(step)
    return result

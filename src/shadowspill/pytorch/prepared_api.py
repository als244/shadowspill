"""Search prepared programs, then admit the selected computation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from typing import Any

from torch import nn

from shadowspill.errors import AdmissionError
from shadowspill.planner import SearchOptions, StepDataOrdering
from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.program_inputs import TransferBandwidths
from shadowspill.runtime.configuration import resolve_physical_execution_budget
from shadowspill.search.planner import _Planner
from shadowspill.search.report import StepSearchReport
from shadowspill.step import StepProgram
from shadowspill.store import StoreMode

from .callables import PlannedTrainStep
from .prepared import PreparedStep
from .step_search.sweep import _Sweep


class _PreparedPrograms:
    def __init__(self, candidates: Mapping[str, PreparedStep]) -> None:
        self.candidates = candidates

    def programs(
        self,
        candidate: str,
        accumulation: int,
        orderings: Sequence[StepDataOrdering],
    ) -> tuple[tuple[StepProgram, ...], None]:
        return self.candidates[candidate].programs, None


def search_prepared_steps(
    candidates: Mapping[str, PreparedStep],
    *,
    budgets: Sequence[tuple[int, int]],
    search_options: SearchOptions | None = None,
    transfer_bandwidths: TransferBandwidths | None = None,
    artifact_store: str | PathLike[str] | None = None,
    plan_store: str | PathLike[str] | None = None,
    plan_store_mode: StoreMode = "contribute",
    incumbents: bool = True,
    keep_resolutions: bool = False,
    metadata: Mapping[str, Any] | None = None,
    verbose: bool = False,
    progress: Callable[[str], None] | None = None,
) -> StepSearchReport:
    """Search every prepared geometry and ordering at each budget, without a GPU.

    Returns the full selected plan per feasible budget in ``winner_plans``.
    The existing planner's incumbent propagation and plan store are shared with
    ``plan_step_search``. No export, materialization, or profiling runs here.
    """
    if not budgets or not candidates:
        raise ValueError("at least one candidate and budget are required")
    if any(not name or not step.programs for name, step in candidates.items()):
        raise ValueError("each named candidate must contain prepared programs")
    geometries = tuple(
        (name, len(step.signature_digests)) for name, step in candidates.items()
    )
    orderings = tuple(
        tuple(p.data_ordering for p in step.programs) for step in candidates.values()
    )
    sweep = _Sweep(
        ask=_Planner(
            transfer_bandwidths,
            search_options,
            artifact_store,
            plan_store,
            plan_store_mode,
            verbose,
            keep_resolutions,
        ),
        budgets=tuple(budgets),
        incumbents=incumbents,
        announce=progress or (lambda message: None),
        point_total=sum(len(items) for items in orderings) * len(budgets),
    )
    sweep.run(geometries, orderings, _PreparedPrograms(candidates))
    winners = sweep.winner_plans()
    return StepSearchReport(
        budgets=tuple(budgets),
        geometries=tuple(sweep.builds),
        points=tuple(sweep.points),
        metadata=dict(metadata or {}),
        search_options=search_options,
        transfer_bandwidths=transfer_bandwidths,
        winner_plans=winners,
    )


def admit_step(
    model: nn.Module,
    prepared_step: PreparedStep,
    selected_plan: AnnotatedProgramPlan,
    *,
    data_ordering: StepDataOrdering | None = None,
    **bindings: Any,
) -> PlannedTrainStep:
    """Bind a selected prepared computation to fresh state and a live runtime.

    ``bindings`` are ``plan_step``'s runtime, optimizer, example inputs, precision,
    and store arguments. The artifact supplies the objective; the selected plan
    supplies the ordering and transfer calibration. Admission validates the
    reconstructed program and certifies its physical layout without search.
    Budgets are whole-process physical caps. Equivalent orderings can lower to
    the same program; pass the search winner's ``data_ordering`` to retain its
    spelling, or use the first equivalent prepared ordering.
    """
    from .api import plan_step

    if bindings.get("distributed") is not None:
        raise NotImplementedError("persistent preparation currently requires one rank")
    matches = [
        p
        for p in prepared_step.programs
        if p.problem.digest == selected_plan.program.digest
        and (data_ordering is None or p.data_ordering == data_ordering)
    ]
    if not matches:
        raise ValueError("selected plan does not identify a prepared program")
    ordering = matches[0].data_ordering
    forbidden = {
        "objective",
        "depth",
        "breadth",
        "reverse_breadth",
        "pair_loss",
        "incumbent",
        "transfer_bandwidths",
        "selected_plan",
        "prepared_step",
    }
    if forbidden.intersection(bindings):
        raise TypeError(
            "admission gets its computation and selection from the artifacts"
        )
    physical = selected_plan.memory_budgets.execution_bytes
    spill = selected_plan.memory_budgets.spill_bytes
    for name, expected in (("execution_budget", physical), ("spill_budget", spill)):
        if bindings.get(name) is not None and bindings[name] != expected:
            raise AdmissionError(
                "selected plan budgets differ from the runtime request"
            )
    bindings["execution_budget"] = resolve_physical_execution_budget(
        physical, bindings["runtime"].pools[bindings["execution"]]
    )
    bindings["spill_budget"] = spill
    bindings.setdefault("search_options", selected_plan.result.search_options)
    bindings.setdefault("optimizer_ordering", matches[0].optimizer_ordering)
    for name, value in prepared_step.options.items():
        if name == "grad_dtype" and value is not None:
            import torch

            value = getattr(torch, value.removeprefix("torch."))
        if name != "partition" or value in {"auto", "whole"}:
            bindings.setdefault(name, value)
    return plan_step(
        model,
        objective=_prepared_objective,
        prepared_step=prepared_step,
        selected_plan=selected_plan,
        depth=ordering.depth,
        breadth=ordering.breadth,
        reverse_breadth=ordering.reverse_breadth,
        pair_loss=ordering.pair_loss,
        transfer_bandwidths=selected_plan.transfer_bandwidths,
        **bindings,
    )


def _prepared_objective(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("admission must never execute or export a Python objective")


__all__ = ["admit_step", "search_prepared_steps"]

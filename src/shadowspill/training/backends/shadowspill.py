"""Planning and execution over one explicitly entered ShadowSpill runtime."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Self

import torch
from torch import nn

from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.memory import SpillPool, pinned_host, transfer_route
from shadowspill.memory import device as device_pool
from shadowspill.planner import SearchOptions, StepDataOrdering
from shadowspill.pytorch import (
    Runtime,
    admit_step,
    import_model_state,
    plan_forward,
    plan_step,
    plan_step_search,
    prepare_step,
    release_model_state,
    search_prepared_steps,
)
from shadowspill.pytorch.accelerator import resolve_device
from shadowspill.pytorch.callables import PlannedForward, PlannedTrainStep
from shadowspill.pytorch.distributed import BoundDistributed, Distributed
from shadowspill.pytorch.failures import device_exhausted
from shadowspill.pytorch.graph_pairs import (
    GRAPH_PAIR_VARIANTS,
    validate_graph_pair_variants,
)
from shadowspill.pytorch.partition import PartitionPolicy
from shadowspill.pytorch.state.registry import registry_for
from shadowspill.pytorch.step_search.sweep import _refuse_state_left_behind
from shadowspill.search.geometries import default_orderings
from shadowspill.search.report import StepSearchPoint, StepSearchReport
from shadowspill.task.profiling import ProfilingOptions

from ..preparation import PreparedTraining
from ._spec import StepSpec

GIB = 1 << 30


class ShadowSpill:
    """Install pools before accelerator resources; share them across runners."""

    def __init__(
        self,
        *,
        execution_gib: float,
        spill_gib: float | None = None,
        spill_pool: SpillPool | None = None,
        device: str | int | torch.device | None = "auto",
        artifact_store: str | Path | None = None,
        search_options: SearchOptions | None = None,
        profiling_options: ProfilingOptions | None = None,
        partition: Literal["auto", "whole"] | PartitionPolicy = "auto",
        graph_pair_variants: tuple[str, ...] = GRAPH_PAIR_VARIANTS,
        fuse_narrow_gradient_accumulation: bool = True,
        orderings: Sequence[StepDataOrdering]
        | Callable[[int], Sequence[StepDataOrdering]]
        | None = None,
        external_headroom_gib: float = 0.5,
        reject_overbudget: bool = False,
        numa_binding: bool = True,
        control_group: Any = None,
        host_headroom_gib: float = 2.0,
        preparation_timeout: float = 1800.0,
        calibrate: bool = True,
    ) -> None:
        self.device = resolve_device(device)
        if spill_pool is None:
            if spill_gib is None:
                raise ValueError("supply spill_gib or spill_pool")
            spill_pool = pinned_host(capacity=int(spill_gib * GIB))
        spill_bytes = spill_pool.capacity if spill_gib is None else int(spill_gib * GIB)
        if not 0 < spill_bytes <= spill_pool.capacity:
            raise ValueError("spill_gib must be positive and fit spill_pool.capacity")
        self.spill_pool = spill_pool
        self.budget = (int(execution_gib * GIB), spill_bytes)
        self.calibrate = calibrate
        self.artifact_store = artifact_store
        self.search_options = search_options
        self.profiling_options = profiling_options
        self.partition = partition
        self.graph_pair_variants = validate_graph_pair_variants(graph_pair_variants)
        self.fuse_narrow_gradient_accumulation = fuse_narrow_gradient_accumulation
        self.orderings: Callable[[int], Sequence[StepDataOrdering]] | None
        if orderings is None or callable(orderings):
            self.orderings = orderings
        else:
            fixed = tuple(orderings)
            if not fixed or any(not isinstance(o, StepDataOrdering) for o in fixed):
                raise ValueError("orderings must contain StepDataOrdering values")

            def fixed_orderings(count: int) -> Sequence[StepDataOrdering]:
                if any(o.microbatches != count for o in fixed):
                    raise ValueError("fixed orderings must cover every microbatch")
                return fixed

            self.orderings = fixed_orderings
        self.external_headroom_bytes = int(external_headroom_gib * GIB)
        self.reject_overbudget = reject_overbudget
        self.numa_binding = numa_binding
        self.control_group = control_group
        self.host_headroom_bytes = int(host_headroom_gib * GIB)
        self.preparation_timeout = preparation_timeout
        self.runtime: Runtime | None = None
        self._models: weakref.WeakKeyDictionary[nn.Module, nn.Module] = (
            weakref.WeakKeyDictionary()
        )
        self._owned_models: list[nn.Module] = []
        self._sessions: list[_Step | _Forward] = []

    @property
    def is_open(self) -> bool:
        return self.runtime is not None

    def open(self, *, control_group: Any = None) -> Self:
        """Open pools explicitly, or let Trainer.prepare open them on first use."""
        if self.runtime is not None:
            raise RuntimeError("ShadowSpill backend is already entered")
        if control_group is not None:
            if (
                self.control_group is not None
                and self.control_group is not control_group
            ):
                raise ValueError("backend already names a different control group")
            self.control_group = control_group
        self.runtime = Runtime(
            numa_binding=self.numa_binding,
            control_group=self.control_group,
            host_headroom_bytes=self.host_headroom_bytes,
            preparation_timeout=self.preparation_timeout,
            calibrate=self.calibrate,
            pools={
                "execution": device_pool(
                    physical_capacity=self.budget[0],
                    device=self.device.index,
                    external_headroom=self.external_headroom_bytes,
                    reject_overbudget=self.reject_overbudget,
                ),
                "spill": self.spill_pool,
            },
            routes={
                "fetch": transfer_route(source="spill", destination="execution"),
                "evict": transfer_route(source="execution", destination="spill"),
            },
        )
        return self

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(self, *exception: object) -> None:
        self.close()

    def _distributed_binding(
        self,
        model: nn.Module,
        specification: Distributed,
        *,
        shard_optimizer: bool = True,
    ) -> BoundDistributed | None:
        if self.runtime is None:
            raise RuntimeError("enter the ShadowSpill context before preparing runners")
        bound = self.runtime._distributed_for(model, specification)
        if bound is not None:
            bound.shard_optimizer = shard_optimizer
        return bound

    def _import(
        self,
        model: nn.Module,
        distributed: Distributed | None = None,
        *,
        initialize: Callable[[nn.Module], None] | None = None,
    ) -> nn.Module:
        if self.runtime is None:
            raise RuntimeError("enter the ShadowSpill context before preparing runners")
        imported = self._models.get(model)
        if imported is None:
            imported = import_model_state(
                model,
                runtime=self.runtime,
                pool="spill",
                distributed=distributed,
                initialize=initialize,
            )
            self._models[model] = imported
            self._models[imported] = imported
            self._owned_models.append(imported)
        return imported

    def initialize_model(
        self, model: nn.Module, initialize: Callable[[nn.Module], None]
    ) -> nn.Module:
        """Run setup on pool-owned state, before capture or profiling."""
        if model in self._models:
            # Shared evaluation runners already use the imported model.
            from shadowspill.pytorch.state.initialization import pool_values

            assert self.runtime is not None
            with pool_values(self.runtime):
                initialize(self._models[model])
            return self._models[model]
        return self._import(model, initialize=initialize)

    def _planning_args(self) -> dict[str, Any]:
        return dict(
            runtime=self.runtime,
            execution="execution",
            spill="spill",
            execution_budget=self.budget[0],
            spill_budget=self.budget[1],
            execution_device=self.device,
            artifact_store=self.artifact_store,
            search_options=self.search_options,
            profiling_options=self.profiling_options,
            partition=self.partition,
        )

    def prepare_step(
        self,
        model: nn.Module,
        spec: StepSpec,
        examples: Mapping[str, Sequence[Sequence[Any]]],
    ) -> tuple[_Step, str]:
        if spec.distributed is None:
            prepared = self.prepare_artifacts(model, spec, examples)
            return self.admit_prepared(model, spec, examples, prepared)
        return self._prepare_distributed_step(model, spec, examples)

    def prepare_artifacts(
        self,
        model: nn.Module,
        spec: StepSpec,
        examples: Mapping[str, Sequence[Sequence[Any]]],
        *,
        budgets: Sequence[tuple[int, int]] | None = None,
    ) -> PreparedTraining:
        """Prepare each candidate once, and search all requested budgets."""
        if spec.distributed is not None:
            raise NotImplementedError(
                "persistent preparation currently requires one rank"
            )
        requested = tuple([self.budget] if budgets is None else budgets)
        if not requested or any(
            not (0 < execution <= self.budget[0] and 0 < spill <= self.budget[1])
            for execution, spill in requested
        ):
            raise ValueError("preparation budgets must be positive and fit the runtime")
        model = self._import(model)
        assert self.runtime is not None
        arguments = self._planning_args()
        arguments.pop("search_options")
        prepared = {}
        refused: list[StepSearchPoint] = []
        for name, inputs in examples.items():
            orderings = tuple((self.orderings or default_orderings)(len(inputs)))
            held = registry_for(self.runtime).values()
            try:
                prepared[name] = prepare_step(
                    model,
                    objective=spec.objective,
                    optimizer=spec.optimizer(model),
                    hyperparams=spec.hyperparams,
                    example_inputs=inputs,
                    master_dtype=spec.master_dtype,
                    grad_dtype=spec.grad_dtype,
                    parameter_metrics=spec.parameter_metrics,
                    fuse_narrow_gradient_accumulation=self.fuse_narrow_gradient_accumulation,
                    orderings=orderings,
                    graph_pair_variants=self.graph_pair_variants,
                    **arguments,
                )
            except Exception as error:
                if not device_exhausted(error):
                    raise
                _refuse_state_left_behind(self.runtime, held, name, len(inputs), error)
                print(
                    f"candidate {name}: device exhausted during preparation: {error}",
                    flush=True,
                )
                refused.extend(
                    StepSearchPoint(
                        name,
                        len(inputs),
                        ordering,
                        execution,
                        spill,
                        "infeasible",
                        None,
                        None,
                        str(error),
                        0.0,
                    )
                    for ordering in orderings
                    for execution, spill in requested
                )
        report = (
            search_prepared_steps(
                prepared,
                budgets=requested,
                artifact_store=self.artifact_store,
                search_options=self.search_options,
                progress=lambda message: print(message, flush=True),
            )
            if prepared
            else StepSearchReport(budgets=requested, geometries=(), points=())
        )
        if refused:
            report = replace(report, points=(*report.points, *refused))
        return PreparedTraining(prepared, report)

    def admit_prepared(
        self,
        model: nn.Module,
        spec: StepSpec,
        examples: Mapping[str, Sequence[Sequence[Any]]],
        prepared: PreparedTraining,
    ) -> tuple[_Step, str]:
        """Create just the callable selected for this backend's actual pools."""
        if spec.distributed is not None:
            raise NotImplementedError(
                "persistent preparation currently requires one rank"
            )
        model = self._import(model, spec.distributed)
        winner = prepared.search.winner(*self.budget)
        if winner is None:
            points = [
                p
                for p in prepared.search.points
                if (p.execution_budget_bytes, p.spill_budget_bytes) == self.budget
            ]
            if not points:
                raise ValueError(
                    "this budget has not been searched in the prepared artifacts"
                )
            message = "no prepared microbatch candidate fits the configured budgets"
            if any(p.status == "search_exhausted" for p in points):
                raise PlanSearchExhaustedError(message)
            if all(p.status == "infeasible" for p in points):
                raise PlanInfeasibleError(message, kind="prepared_candidates")
            raise RuntimeError(
                message + ": " + "; ".join(p.error or p.status for p in points)
            )
        chosen = winner.candidate
        call = admit_step(
            model,
            prepared.candidates[chosen],
            prepared.search.winner_plans[self.budget],
            data_ordering=winner.ordering,
            optimizer=spec.optimizer(model),
            hyperparams=spec.hyperparams,
            example_inputs=examples[chosen],
            master_dtype=spec.master_dtype,
            grad_dtype=spec.grad_dtype,
            parameter_metrics=spec.parameter_metrics,
            fuse_narrow_gradient_accumulation=self.fuse_narrow_gradient_accumulation,
            graph_pair_variants=self.graph_pair_variants,
            **self._planning_args(),
        )
        session = _Step(model, call, prepared.search)
        self._sessions.append(session)
        return session, chosen

    def _prepare_distributed_step(
        self,
        model: nn.Module,
        spec: StepSpec,
        examples: Mapping[str, Sequence[Sequence[Any]]],
    ) -> tuple[_Step, str]:
        model = self._import(model, spec.distributed)
        common: dict[str, Any] = dict(
            objective=spec.objective,
            distributed=spec.distributed,
            shard_optimizer=spec.shard_optimizer,
            optimizer=spec.optimizer(model),
            hyperparams=spec.hyperparams,
            master_dtype=spec.master_dtype,
            grad_dtype=spec.grad_dtype,
            parameter_metrics=spec.parameter_metrics,
            fuse_narrow_gradient_accumulation=self.fuse_narrow_gradient_accumulation,
            graph_pair_variants=self.graph_pair_variants,
            **self._planning_args(),
        )
        report = None
        chosen = next(iter(examples))
        selection: dict[str, Any] = {}
        if len(examples) > 1 or len(examples[chosen]) > 1:
            search_args = dict(common)
            for key in ("execution_budget", "spill_budget"):
                search_args.pop(key)
            report = plan_step_search(
                model,
                candidates=examples,
                budgets=[self.budget],
                orderings=self.orderings,
                progress=lambda s: print(s, flush=True),
                **search_args,
            )
            winner = report.winner(*self.budget)
            if winner is None:
                raise RuntimeError(
                    "no microbatch candidate fits the configured budgets"
                )
            chosen = winner.candidate
            selection = dict(
                depth=winner.ordering.depth,
                breadth=winner.ordering.breadth,
                reverse_breadth=winner.ordering.reverse_breadth,
                pair_loss=winner.ordering.pair_loss,
                incumbent=report.winner_plans[self.budget],
                transfer_bandwidths=report.planned_lanes,
            )
        call = plan_step(model, example_inputs=examples[chosen], **common, **selection)
        session = _Step(model, call, report)
        self._sessions.append(session)
        return session, chosen

    def prepare_forward(
        self,
        model: nn.Module,
        forward_fn: Callable[..., Any],
        example: Any,
        *,
        training: bool = False,
        distributed: Distributed | None = None,
    ) -> _Forward:
        model = self._import(model, distributed)
        owners = [
            session
            for session in self._sessions
            if not session.closed and session.model is model
        ]
        arguments = self._planning_args()
        if owners:
            arguments.pop("execution_budget")
            arguments["share_slab_with"] = owners[0].call
        call = plan_forward(
            model,
            forward_fn=forward_fn,
            distributed=distributed,
            example_inputs=[example],
            **arguments,
        )
        session = _Forward(model, call)
        self._sessions.append(session)
        return session

    def close(self) -> None:
        if self.runtime is None:
            return
        for session in reversed(self._sessions):
            session.close()
        for model in reversed(self._owned_models):
            release_model_state(model, runtime=self.runtime)
        self.runtime.close()
        self.runtime = None
        self._sessions.clear()
        self._owned_models.clear()
        self._models.clear()


class _Session[Call: (PlannedForward, PlannedTrainStep)]:
    def __init__(self, model: nn.Module, call: Call) -> None:
        self.model = model
        self.call: Call = call
        self.plan = call.plan_report
        self.closed = False

    def save_plan(self, path: Path) -> None:
        """Export the admitted plan using the neutral execution-plan schema."""
        path.write_text(self.plan.execution_plan.to_json() + "\n")

    def synchronize(self) -> None:
        self.call.synchronize()

    def close(self) -> None:
        if not self.closed:
            self.call.close()
            self.closed = True


class _Forward(_Session[PlannedForward]):
    def run(self, data: Any) -> Any:
        return self.call([data])


class _Step(_Session[PlannedTrainStep]):
    def __init__(
        self,
        model: nn.Module,
        call: PlannedTrainStep,
        planning: StepSearchReport | None,
    ) -> None:
        super().__init__(model, call)
        self.planning = planning

    def run(
        self,
        microbatches: Sequence[Sequence[Any]],
        hyperparams: Mapping[str, Any],
    ) -> tuple[Any, Any, Any]:
        result = self.call(microbatches, hyperparams=hyperparams)
        return result.objectives, result.metrics, result.parameter_metrics

    def diagnose(
        self, microbatches: Sequence[Sequence[Any]], directory: Path, *, warmup: int = 1
    ) -> dict[str, Any]:
        from ._diagnostics import diagnose

        return diagnose(self.call, microbatches, directory, warmup=warmup)

    def save(
        self,
        path: Path,
        *,
        weights: Literal["master", "compute"] = "master",
        include_optimizer: bool = True,
        frozen_state_id: str | None = None,
    ) -> None:
        self.call.save(
            path,
            weights=weights,
            include_optimizer=include_optimizer,
            frozen_state_id=frozen_state_id,
        )

    def load_state_dict(
        self, state: Mapping[str, Any], *, frozen_state_id: str | None = None
    ) -> None:
        self.call.load_state_dict(state, frozen_state_id=frozen_state_id)

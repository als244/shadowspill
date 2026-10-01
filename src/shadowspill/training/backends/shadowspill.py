"""Planning and execution over one explicitly entered ShadowSpill runtime."""

from __future__ import annotations

import tempfile
import weakref
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self

import torch
from torch import nn

from shadowspill.memory import device as device_pool
from shadowspill.memory import pinned_host, transfer_route
from shadowspill.planner import SearchOptions, StepDataOrdering
from shadowspill.pytorch import (
    Runtime,
    import_model_state,
    plan_forward,
    plan_step,
    plan_step_search,
    release_model_state,
)
from shadowspill.pytorch.accelerator import resolve_device
from shadowspill.pytorch.callables import PlannedForward, PlannedTrainStep
from shadowspill.pytorch.distributed import BoundDistributed, Distributed
from shadowspill.pytorch.partition import PartitionPolicy
from shadowspill.search.report import StepSearchReport
from shadowspill.task.profiling import ProfilingOptions

from ._spec import StepSpec

GIB = 1 << 30


class ShadowSpill:
    """Install pools before accelerator resources; share them across runners."""

    def __init__(
        self,
        *,
        execution_gib: float,
        spill_gib: float,
        device: str | int | torch.device | None = "auto",
        artifact_store: str | Path | None = None,
        search_options: SearchOptions | None = None,
        profiling_options: ProfilingOptions | None = None,
        partition: Literal["auto", "whole"] | PartitionPolicy = "auto",
        round_accumulation_once: bool = False,
        orderings: Callable[[int], Sequence[StepDataOrdering]] | None = None,
        external_headroom_gib: float = 0.5,
        reject_overbudget: bool = False,
        numa_binding: bool = True,
        control_group: Any = None,
        host_headroom_gib: float = 2.0,
        preparation_timeout: float = 1800.0,
    ) -> None:
        self.device = resolve_device(device)
        self.budget = (int(execution_gib * GIB), int(spill_gib * GIB))
        self.artifact_store = artifact_store
        self.search_options = search_options
        self.profiling_options = profiling_options
        self.partition = partition
        self.round_accumulation_once = round_accumulation_once
        self.orderings = orderings
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
        self._temporary_store: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> Self:
        if self.runtime is not None:
            raise RuntimeError("ShadowSpill backend is already entered")
        self.runtime = Runtime(
            numa_binding=self.numa_binding,
            control_group=self.control_group,
            host_headroom_bytes=self.host_headroom_bytes,
            preparation_timeout=self.preparation_timeout,
            pools={
                "execution": device_pool(
                    physical_capacity=self.budget[0],
                    device=self.device.index,
                    external_headroom=self.external_headroom_bytes,
                    reject_overbudget=self.reject_overbudget,
                ),
                "spill": pinned_host(capacity=self.budget[1]),
            },
            routes={
                "fetch": transfer_route(source="spill", destination="execution"),
                "evict": transfer_route(source="execution", destination="spill"),
            },
        )
        return self

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
        self, model: nn.Module, distributed: Distributed | None = None
    ) -> nn.Module:
        if self.runtime is None:
            raise RuntimeError("enter the ShadowSpill context before preparing runners")
        imported = self._models.get(model)
        if imported is None:
            imported = import_model_state(
                model, runtime=self.runtime, pool="spill", distributed=distributed
            )
            self._models[model] = imported
            self._models[imported] = imported
            self._owned_models.append(imported)
        return imported

    def _planning_args(self) -> dict[str, Any]:
        return dict(
            runtime=self.runtime,
            execution="execution",
            spill="spill",
            execution_budget=self.budget[0],
            spill_budget=self.budget[1],
            execution_device=self.device,
            artifact_store=(
                self._temporary_store.name
                if self._temporary_store
                else self.artifact_store
            ),
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
            round_accumulation_once=self.round_accumulation_once,
            **self._planning_args(),
        )
        report = None
        chosen = next(iter(examples))
        selection: dict[str, Any] = {}
        if len(examples) > 1 or len(examples[chosen]) > 1:
            if self.artifact_store is None:
                if self._temporary_store is None:
                    self._temporary_store = tempfile.TemporaryDirectory(
                        prefix="shadowspill-training-"
                    )
                common["artifact_store"] = self._temporary_store.name
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
        if self._temporary_store is not None:
            self._temporary_store.cleanup()
            self._temporary_store = None


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
        self, path: Path, *, weights: Literal["master", "compute"] = "master"
    ) -> None:
        self.call.save(path, weights=weights)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.call.load_state_dict(state)

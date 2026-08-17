"""Fresh-process accumulated-training parity and lifecycle canary."""

from __future__ import annotations

import ctypes
import gc
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn

from shadowspill.memory import device, pinned_host
from shadowspill.pytorch import (
    ObjectiveResult,
    Runtime,
    export_model_state,
    import_model_state,
    plan_step,
)
from shadowspill.pytorch.runtime_adapter.abi import AdapterStatistics
from shadowspill.pytorch.runtime_adapter.allocator import installed_allocator


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1024, 1024, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)


def _objective(
    model: nn.Module, value: torch.Tensor, target: torch.Tensor, label: str
) -> ObjectiveResult:
    error = model(value) - target
    loss = error.square().mean()
    return ObjectiveResult(loss, {"mean": error.detach().mean(), "label": label})


def _clone_model_state(state: object) -> dict[str, torch.Tensor]:
    if not isinstance(state, dict) or not isinstance(state.get("model"), dict):
        raise AssertionError("training checkpoint has an invalid model payload")
    return {
        name: value.clone()
        for name, value in state["model"].items()
        if isinstance(name, str) and isinstance(value, torch.Tensor)
    }


def _assert_bitwise(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> None:
    if set(left) != set(right):
        raise AssertionError("checkpoint replay changed model keys")
    for name in left:
        if not torch.equal(left[name], right[name]):
            raise AssertionError(f"checkpoint replay changed {name!r}")


def _statistics() -> AdapterStatistics:
    installed = installed_allocator()
    if installed is None:
        raise AssertionError("training did not install the allocator")
    result = AdapterStatistics()
    status = int(
        installed.library.shadowspill_pytorch_allocator_statistics(ctypes.byref(result))
    )
    if status != 0:
        raise AssertionError(f"statistics failed with status {status}")
    return result


def main(arguments: Iterable[str] | None = None) -> int:
    values = tuple(sys.argv[1:] if arguments is None else arguments)
    adapter = Path(values[0]).resolve()
    with tempfile.TemporaryDirectory() as cache:
        torch.manual_seed(127)
        model = _Model()
        reference = _Model()
        reference.load_state_dict(model.state_dict())
        example_inputs = [
            [torch.randn(3, 1024), torch.randn(3, 1024), "short"],
            [torch.randn(5, 1024), torch.randn(5, 1024), "long"],
        ]
        steps: list[list[list[object]]] = []
        for step in range(5):
            torch.manual_seed(1000 + step)
            steps.append(
                [
                    [torch.randn(3, 1024), torch.randn(3, 1024), "short"],
                    [torch.randn(5, 1024), torch.randn(5, 1024), "long"],
                ]
            )

        reference_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=0.003, foreach=False
        )

        constructed: list[torch.optim.AdamW] = []
        optimizer_calls: list[int] = []

        def optimizer_factory(
            parameters: Iterable[torch.nn.Parameter],
        ) -> torch.optim.AdamW:
            optimizer = torch.optim.AdamW(parameters, lr=0.003, foreach=False)
            constructed.append(optimizer)

            def count_actual_step(
                stepped: torch.optim.Optimizer,
                _args: tuple[object, ...],
                _kwargs: dict[str, object],
            ) -> None:
                if stepped is optimizer:
                    optimizer_calls.append(1)

            optimizer.register_step_post_hook(count_actual_step)
            return optimizer

        runtime = Runtime(
            pools={
                "execution": device(
                    physical_capacity=2 << 30,
                    provider_headroom=512 << 20,
                ),
                "spill": pinned_host(capacity=1 << 30),
            },
            library_path=adapter,
        )
        model = import_model_state(
            model,
            runtime=runtime,
            pool="spill",
            release_source=True,
        )
        parameter_ids = tuple(id(parameter) for parameter in model.parameters())
        planned = plan_step(
            model,
            objective=_objective,
            opt=optimizer_factory,
            example_inputs=example_inputs,
            runtime=runtime,
            execution="execution",
            spill="spill",
            planning_cachedir=cache,
            profiling_metadata=(
                {"batch_size": 3, "label": "short"},
                {"batch_size": 5, "label": "long"},
            ),
        )
        if len(constructed) != 1:
            raise AssertionError("optimizer factory was not invoked exactly once")
        plan_diagnostics = planned.plan_report.diagnostics
        if not plan_diagnostics.cache_artifacts:
            raise AssertionError("plan diagnostics omitted cache artifacts")
        if len(plan_diagnostics.profiling_metadata) != 2:
            raise AssertionError("plan diagnostics omitted profiling metadata")
        if (
            plan_diagnostics.measured_wall_time_ns
            + plan_diagnostics.unattributed_overhead_ns
            != plan_diagnostics.total_wall_time_ns
        ):
            raise AssertionError("plan diagnostic wall time does not reconcile")
        phase_names = {item.name for item in plan_diagnostics.phases}
        if (
            "pressurefit_simulation" not in phase_names
            or "compiled_entrypoint_construction" not in phase_names
            or "unique_stage_warmup_profiling" not in phase_names
        ):
            raise AssertionError("plan diagnostic omitted a required phase")
        if not plan_diagnostics.unique_stages:
            raise AssertionError("plan diagnostic omitted unique graph stages")
        for stage in plan_diagnostics.unique_stages:
            if tuple(pair.variant for pair in stage.graph_pairs) != (
                "save",
                "recompute",
            ):
                raise AssertionError("unique stage omitted a graph-pair choice")
            for pair in stage.graph_pairs:
                if pair.forward.runtime_ns <= 0 or pair.backward.runtime_ns <= 0:
                    raise AssertionError("graph pair omitted a profile runtime")
                for profile in (pair.forward, pair.backward):
                    if not profile.semantic_roots or not profile.compiled_roots:
                        raise AssertionError(
                            "graph pair omitted semantic/physical layout"
                        )
                    if not profile.allocation_contract_digest:
                        raise AssertionError(
                            "graph pair omitted its allocation contract"
                        )
                    if profile.semantic_contract_capture_ns <= 0:
                        raise AssertionError("graph pair omitted contract timing")
                    if profile.physical_profile_wall_time_ns <= 0:
                        raise AssertionError("graph pair omitted profiling timing")
        if planned.plan_report.initial_execution_plan is not None:
            raise AssertionError("preinitialized AdamW state emitted an initial plan")
        active = planned.plan_report.execution_plan.program.selected_tasks(
            planned.plan_report.execution_plan.selections
        )
        for execution_ordinal, task in enumerate(active):
            execution_task_id = f"execution_{execution_ordinal:06d}"
            task_diagnostic = plan_diagnostics.task(execution_task_id)
            if task_diagnostic.task_id != task.task_id:
                return 1
            if not task_diagnostic.selected:
                raise AssertionError("selected task is not marked selected")
            if task.phase != "optimizer" and (
                not task_diagnostic.semantic_contract_digest
                or not task_diagnostic.compiled_layout_digest
            ):
                raise AssertionError("selected task omitted lowering diagnostics")
            if task.phase != "optimizer" and (
                task_diagnostic.graph_pair_variant
                != task_diagnostic.chosen_graph_pair_variant
            ):
                raise AssertionError("selected task has the wrong graph-pair choice")
            if (
                task_diagnostic.execution_ordinal != execution_ordinal
                or task_diagnostic.execution_task_id
                != f"execution_{execution_ordinal:06d}"
                or not task_diagnostic.semantic_name
            ):
                raise AssertionError("selected task has no chronological identity")
        if tuple(task.phase for task in active) != (
            "forward",
            "backward",
            "forward",
            "backward",
            "optimizer",
        ):
            raise AssertionError("training plan has the wrong accumulated task order")

        checkpoint: dict[str, object] | None = None
        for step, microbatches in enumerate(steps):
            reference_optimizer.zero_grad(set_to_none=True)
            reference_losses: list[torch.Tensor] = []
            for value, target, label in microbatches:
                result = _objective(reference, value, target, label)
                result.loss.backward()
                reference_losses.append(result.loss.detach())
            reference_optimizer.step()
            actual = planned(microbatches, runtime_trace=True)
            if actual.diagnostics is None:
                raise AssertionError(
                    "runtime_trace=True omitted StepResult diagnostics"
                )
            diagnostics = actual.diagnostics.result()
            execution_timing = diagnostics.timing
            if step == 0:
                if execution_timing.trace_setup_seconds <= 0.0:
                    raise AssertionError("first trace omitted lazy setup time")
                if (
                    not diagnostics.runtime.events
                    or diagnostics.runtime.begin_timestamp_ns <= 0
                    or diagnostics.runtime.end_timestamp_ns
                    < diagnostics.runtime.begin_timestamp_ns
                    or diagnostics.runtime.event_overflow
                    or diagnostics.runtime.allocation_event_overflow
                ):
                    raise AssertionError("native runtime trace is incomplete")
                native_kinds = {
                    item.kind.name.lower() for item in diagnostics.runtime.events
                }
                if not {
                    "session_begin",
                    "session_end",
                    "before_task",
                    "after_task",
                    "action_queued",
                }.issubset(native_kinds):
                    raise AssertionError("native runtime trace omitted core events")
                if diagnostics.runtime.materialized_allocation_requests != (
                    diagnostics.runtime.allocation_requests
                    - diagnostics.runtime.zero_byte_allocation_requests
                ):
                    raise AssertionError("zero-byte allocation accounting is invalid")
                if diagnostics.allocator.live_allocations_before < 0 or (
                    diagnostics.allocator.live_allocations_after < 0
                ):
                    raise AssertionError("live-allocation accounting is invalid")
                if execution_timing.compute_seconds <= 0.0:
                    raise AssertionError(
                        "compute-only qualification timing is not positive"
                    )
                if execution_timing.optimizer_seconds <= 0.0:
                    raise AssertionError("optimizer timing is not positive")
                phases = dict(execution_timing.phase_gpu_seconds)
                if set(phases) != {"forward", "backward", "optimizer"}:
                    raise AssertionError("execution timing omitted a task phase")
                if len(execution_timing.tasks) != len(active):
                    raise AssertionError("execution timing omitted a selected task")
                if (
                    not diagnostics.summary.trace_complete
                    or diagnostics.summary.profiled_task_seconds <= 0.0
                    or diagnostics.summary.real_task_event_seconds <= 0.0
                    or diagnostics.summary.real_selected_span_seconds <= 0.0
                ):
                    raise AssertionError("step timing reconciliation is incomplete")
                if set(diagnostics.simulator_comparison) != set(execution_timing.tasks):
                    raise AssertionError("task simulator comparison is incomplete")
                expected_transfers = (
                    planned.plan_report.pressurefit_result.simulation.transfer_intervals
                )
                if len(diagnostics.transfers.simulator_comparison) != len(
                    expected_transfers
                ):
                    raise AssertionError("transfer simulator comparison is incomplete")
                for transfer in diagnostics.transfers.simulator_comparison.values():
                    if (
                        transfer.real_completion_timestamp_ns
                        < transfer.real_dispatch_timestamp_ns
                        or transfer.bytes <= 0
                    ):
                        raise AssertionError("transfer timing comparison is invalid")
                for execution_ordinal, task_timing in enumerate(
                    execution_timing.tasks.values()
                ):
                    execution_task_id = f"execution_{execution_ordinal:06d}"
                    if (
                        task_timing.execution_ordinal != execution_ordinal
                        or task_timing.execution_task_id != execution_task_id
                        or not task_timing.semantic_name
                    ):
                        raise AssertionError(
                            "step trace has no chronological task identity"
                        )
                    if diagnostics.tasks[execution_task_id] is not task_timing:
                        raise AssertionError("step task lookup is not canonical")
                    planned_task = plan_diagnostics.task(execution_task_id)
                    if (
                        planned_task.task_id != task_timing.task_id
                        or planned_task.execution_ordinal
                        != task_timing.execution_ordinal
                        or planned_task.semantic_name != task_timing.semantic_name
                    ):
                        raise AssertionError(
                            "plan and step task identities do not agree"
                        )
                    if task_timing.expected_profile_seconds <= 0.0:
                        raise AssertionError("task omitted expected profile time")
                    comparison = diagnostics.simulator_comparison[execution_task_id]
                    if (
                        comparison.task_id != task_timing.task_id
                        or comparison.simulated_end_ns < comparison.simulated_start_ns
                        or comparison.real_end_seconds < comparison.real_start_seconds
                    ):
                        raise AssertionError("task simulator timing is invalid")
                    if any(
                        timestamp <= 0
                        for timestamp in (
                            task_timing.before_task_enter_timestamp_ns,
                            task_timing.before_task_exit_timestamp_ns,
                            task_timing.after_task_enter_timestamp_ns,
                            task_timing.after_task_exit_timestamp_ns,
                            task_timing.before_readiness_waits_timestamp_ns,
                            task_timing.before_task_compute_timestamp_ns,
                            task_timing.after_task_compute_timestamp_ns,
                        )
                    ):
                        raise AssertionError("task omitted one of seven timestamps")
                    if (
                        task_timing.before_readiness_waits_seconds is None
                        or task_timing.before_task_compute_seconds is None
                        or task_timing.after_task_compute_seconds is None
                        or task_timing.readiness_wait_seconds is None
                        or task_timing.task_compute_seconds is None
                        or task_timing.native_before_task_enter_seconds is None
                        or task_timing.native_before_task_exit_seconds is None
                        or task_timing.native_after_task_enter_seconds is None
                        or task_timing.native_after_task_exit_seconds is None
                    ):
                        raise AssertionError("host callback timing omitted a boundary")
                    if (
                        not task_timing.before_readiness_waits_sequence
                        < task_timing.before_task_compute_sequence
                        < task_timing.after_task_compute_sequence
                    ):
                        raise AssertionError(
                            "host callback boundary order is invalid for "
                            f"{task_timing.task_id}: "
                            f"{task_timing.before_readiness_waits_sequence}, "
                            f"{task_timing.before_task_compute_sequence}, "
                            f"{task_timing.after_task_compute_sequence}"
                        )
            if actual.step_number != step + 1 or len(actual.objectives) != 2:
                raise AssertionError("StepResult has the wrong logical step")
            for loss, expected in zip(actual.objectives, reference_losses, strict=True):
                torch.testing.assert_close(loss.cpu(), expected, rtol=2e-5, atol=2e-6)
            if tuple(metric["label"] for metric in actual.metrics) != (
                "short",
                "long",
            ):
                raise AssertionError("static objective metrics changed")
            if step == 2:
                statistics_before_checkpoint = _statistics()
                checkpoint = planned.state_dict()
                statistics_after_checkpoint = _statistics()
                if (
                    statistics_after_checkpoint.allocation_callbacks
                    != statistics_before_checkpoint.allocation_callbacks
                    or statistics_after_checkpoint.free_callbacks
                    != statistics_before_checkpoint.free_callbacks
                ):
                    raise AssertionError(
                        "checkpointing manufactured CUDA placeholder allocations"
                    )
        if len(optimizer_calls) != 5 or checkpoint is None:
            raise AssertionError("optimizer mutation count differs from step count")
        uninterrupted = _clone_model_state(planned.state_dict())
        planned.load_state_dict(checkpoint)
        for microbatches in steps[3:]:
            replay_result = planned(microbatches, runtime_trace=True)
            if replay_result.diagnostics is None:
                raise AssertionError("replay omitted trace diagnostics")
            replay_result.diagnostics.result()
        replayed = _clone_model_state(planned.state_dict())
        _assert_bitwise(uninterrupted, replayed)
        if len(optimizer_calls) != 7:
            raise AssertionError("checkpoint replay did not run one update per call")
        optimizer_state = planned.state_dict()["optimizer"]
        if not isinstance(optimizer_state, dict):
            raise AssertionError("optimizer checkpoint is not a mapping")
        for parameter_state in optimizer_state["state"].values():
            for value in parameter_state.values():
                if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                    raise AssertionError("optimizer checkpoint retained CUDA storage")

        planned.close()
        planned.close()
        export_model_state(model, runtime=runtime, release_runtime=True)
        if tuple(id(parameter) for parameter in model.parameters()) != parameter_ids:
            raise AssertionError("training replaced a Parameter object")
        if any(parameter.device.type != "cpu" for parameter in model.parameters()):
            raise AssertionError("training close did not restore CPU state")
        closed_optimizer = planned.state_dict()["optimizer"]
        for parameter_state in closed_optimizer["state"].values():
            for value in parameter_state.values():
                if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                    raise AssertionError("close did not restore optimizer state to CPU")
        for actual, expected in zip(
            model.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=5e-5)
        statistics = _statistics()
        if statistics.callback_failures != 0 or statistics.pointer_lookup_failures != 0:
            raise AssertionError("training produced allocator callback failures")
        if statistics.cuda.device_allocations != 1:
            raise AssertionError("training grew the CUDA slab")
        if statistics.cuda.pinned_host_allocations != 1:
            raise AssertionError("training grew the configured spill pool")

        warm_model = _Model()
        warm_model = import_model_state(
            warm_model,
            runtime=runtime,
            pool="spill",
            release_source=True,
        )
        warm = plan_step(
            warm_model,
            objective=_objective,
            opt=optimizer_factory,
            example_inputs=example_inputs,
            runtime=runtime,
            execution="execution",
            spill="spill",
            planning_cachedir=cache,
            profiling_metadata=(
                {"batch_size": 3, "label": "short"},
                {"batch_size": 5, "label": "long"},
            ),
        )
        warm_diagnostics = warm.plan_report.diagnostics
        if (
            warm_diagnostics.aot_graph_pair_cache_hits == 0
            or warm_diagnostics.aot_graph_pair_cache_misses != 0
            or warm_diagnostics.profile_cache_hits == 0
            or warm_diagnostics.profile_cache_misses != 0
        ):
            raise AssertionError("second planning call did not reuse warm artifacts")
        warm_result = warm(steps[0])
        if not all(torch.isfinite(value).all() for value in warm_result.objectives):
            raise AssertionError("warm-cache execution produced a non-finite loss")
        warm.close()
        export_model_state(warm_model, runtime=runtime, release_runtime=True)
        del actual
        del loss
        del replay_result
        del warm_result
        gc.collect()
        torch.cuda.synchronize()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

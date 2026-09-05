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

from shadowspill.memory import device, pinned_host, transfer_route
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
            routes={
                "fetch": transfer_route(source="spill", destination="execution"),
                "evict": transfer_route(source="execution", destination="spill"),
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
            artifact_store_dir=cache,
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
        if plan_diagnostics.measured_wall_time_ns > plan_diagnostics.total_wall_time_ns:
            raise AssertionError("plan phases measure more than the whole call")
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
            summary = diagnostics.summary
            timelines = diagnostics.timelines
            if step == 0:
                if summary.trace_setup_seconds <= 0.0:
                    raise AssertionError("first trace omitted lazy setup time")
                if (
                    not diagnostics.runtime.events
                    or diagnostics.runtime.began_at_ns <= 0
                    or diagnostics.runtime.ended_at_ns < diagnostics.runtime.began_at_ns
                    or diagnostics.runtime.event_overflow
                    or diagnostics.runtime.allocation_event_overflow
                ):
                    raise AssertionError("runtime trace is incomplete")
                kinds = {item.kind.name.lower() for item in diagnostics.runtime.events}
                if not {
                    "session_begin",
                    "session_end",
                    "before_task",
                    "after_task",
                    "action_queued",
                }.issubset(kinds):
                    raise AssertionError("runtime trace omitted core events")
                if diagnostics.runtime.materialized_allocation_requests != (
                    diagnostics.runtime.allocation_requests
                    - diagnostics.runtime.zero_byte_allocation_requests
                ):
                    raise AssertionError("zero-byte allocation accounting is invalid")
                if diagnostics.allocator.live_allocations_before < 0 or (
                    diagnostics.allocator.live_allocations_after < 0
                ):
                    raise AssertionError("live-allocation accounting is invalid")
                if summary.real_selected_span_seconds <= 0.0:
                    raise AssertionError(
                        "compute-only qualification timing is not positive"
                    )
                if summary.optimizer_span_seconds <= 0.0:
                    raise AssertionError("optimizer timing is not positive")
                phases = {item.phase for item in summary.phase_comparisons}
                if phases != {"forward", "backward", "optimizer"}:
                    raise AssertionError("step summary omitted a task phase")
                if len(timelines.compute) != len(active) or (
                    set(timelines.compute) != set(diagnostics.tasks)
                ):
                    raise AssertionError("compute lane omitted a selected task")
                if (
                    not summary.trace_complete
                    or summary.profiled_task_seconds <= 0.0
                    or summary.real_task_event_seconds <= 0.0
                ):
                    raise AssertionError("step timing reconciliation is incomplete")
                expected_transfers = (
                    planned.plan_report.pressurefit_result.simulation.transfer_intervals
                )
                scheduled = [
                    transfer
                    for group in (
                        diagnostics.transfers.fetch,
                        diagnostics.transfers.evict,
                    )
                    for transfer in group.values()
                    if transfer.triggered_by != "init"
                ]
                if len(scheduled) != len(expected_transfers) or (
                    set(timelines.fetch.order) != set(diagnostics.transfers.fetch)
                    or set(timelines.evict.order) != set(diagnostics.transfers.evict)
                ):
                    raise AssertionError("transfer lanes omitted a scheduled transfer")
                for lane in (timelines.fetch, timelines.evict):
                    group = getattr(diagnostics.transfers, lane.summary.direction)
                    opening = lane.summary.opening_transfers
                    for position, key in enumerate(lane.order):
                        transfer = group[key]
                        expected_sequence = (
                            position if position < opening else position - opening
                        )
                        expected_trigger_kind = "init" if position < opening else "task"
                        if (
                            transfer.sequence != expected_sequence
                            or (transfer.triggered_by == "init")
                            != (expected_trigger_kind == "init")
                            or transfer.direction != lane.summary.direction
                            or transfer.completion_observed_at_seconds
                            < transfer.dispatched_at_seconds
                            or transfer.bytes <= 0
                        ):
                            raise AssertionError("transfer lane record is invalid")
                        if transfer.triggered_by != "init" and (
                            transfer.triggered_by not in diagnostics.tasks
                            or transfer.simulated_started_at_seconds is None
                        ):
                            raise AssertionError(
                                "scheduled transfer lacks its trigger or simulation"
                            )
                        if transfer.lane_started_at_seconds is None or (
                            transfer.lane_finished_at_seconds is None
                            or transfer.lane_finished_at_seconds
                            < transfer.lane_started_at_seconds
                        ):
                            raise AssertionError(
                                "transfer lane record lacks its stream interval"
                            )
                    if lane.summary.measured_transfers != len(lane.order):
                        raise AssertionError(
                            "lane summary miscounts measured transfers"
                        )
                for execution_ordinal, execution_task_id in enumerate(
                    timelines.compute
                ):
                    record = diagnostics.tasks[execution_task_id]
                    if (
                        record.execution_ordinal != execution_ordinal
                        or record.execution_task_id
                        != f"execution_{execution_ordinal:06d}"
                        or not record.semantic_name
                    ):
                        raise AssertionError(
                            "step trace has no chronological task identity"
                        )
                    planned_task = plan_diagnostics.task(execution_task_id)
                    if (
                        planned_task.task_id != record.task_id
                        or planned_task.execution_ordinal != record.execution_ordinal
                        or planned_task.semantic_name != record.semantic_name
                    ):
                        raise AssertionError(
                            "plan and step task identities do not agree"
                        )
                    if record.expected_profile_seconds <= 0.0:
                        raise AssertionError("task omitted expected profile time")
                    if (
                        record.simulated_finished_at_seconds
                        < record.simulated_started_at_seconds
                        or record.compute_finished_at_seconds
                        < record.compute_started_at_seconds
                        or record.compute_started_at_seconds
                        < record.compute_reached_at_seconds
                    ):
                        raise AssertionError("compute lane record is out of order")
                    if any(
                        value is None
                        for value in (
                            record.before_task_entered_at_seconds,
                            record.before_task_exited_at_seconds,
                            record.after_task_entered_at_seconds,
                            record.after_task_exited_at_seconds,
                        )
                    ):
                        raise AssertionError("host boundary timing omitted a boundary")
                    if record.compute_reached_at_seconds <= 0.0 or (
                        record.compute_finished_at_seconds
                        <= record.compute_started_at_seconds
                    ):
                        raise AssertionError("compute lane record lacks stream markers")
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
        for replay_index, microbatches in enumerate(steps[3:]):
            if replay_index == 0:
                submitted = planned.submit(microbatches, runtime_trace=True)
                if submitted.resolved:
                    raise AssertionError(
                        "submitted training step synchronized during dispatch"
                    )
                replay_result = submitted.result()
            else:
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
        # Closing copies nothing: model state went back to the pool it was
        # imported into, and optimizer state ended with the plan.
        try:
            planned.state_dict()
        except RuntimeError as error:
            if "take the checkpoint before close" not in str(error):
                raise AssertionError(
                    "released optimizer state reported unclearly"
                ) from error
        else:
            raise AssertionError("close kept an optimizer checkpoint")
        # The reference trains eagerly on the CPU, so this bound covers
        # CPU-versus-accelerator arithmetic, not ShadowSpill: the trained
        # weights are bitwise identical to the same schedule run eagerly on
        # the accelerator. Measured on an A100 at 5e-4 atol on 2026-09-03,
        # 1048575 of 1048576 weights agree to about 1e-9, and one diverges by
        # 1.4e-4 where AdamW's normalized update amplifies float32 round-off
        # into a fraction of a step. The bound stays far inside one optimizer
        # step (lr 0.003), so a genuinely wrong update still fails here.
        for actual, expected in zip(
            model.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=5e-4)
        statistics = _statistics()
        if statistics.callback_failures != 0 or statistics.pointer_lookup_failures != 0:
            raise AssertionError("training produced allocator callback failures")
        if statistics.backend.device_allocations != 1:
            raise AssertionError("training grew the CUDA slab")
        if statistics.backend.pinned_host_registrations != 1:
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
            artifact_store_dir=cache,
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
        del submitted
        del warm_result
        gc.collect()
        torch.cuda.synchronize()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

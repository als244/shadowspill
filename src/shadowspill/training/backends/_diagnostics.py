"""Uncounted startup invocations of an already prepared training step."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch

from shadowspill.pytorch.callables import PlannedTrainStep
from shadowspill.pytorch.spill import write_spill_tensor

from .. import _checkpoint


def diagnose(
    call: PlannedTrainStep,
    inputs: Sequence[Sequence[Any]],
    directory: Path,
    *,
    warmup: int,
) -> dict[str, Any]:
    """Run the real task schedule at zero LR; retain no optimizer snapshot.

    The optimizer must advertise zero_lr_preserves_state. Only mutable model
    buffers and RNG are saved separately; parameters, masters and moments stay
    in their normal allocations throughout. All ranks call this independently.
    """
    optimizer = call._executor.optimizer_state.optimizer
    if not getattr(optimizer, "zero_lr_preserves_state", False):
        raise ValueError(
            "startup diagnostics require an optimizer with "
            "zero_lr_preserves_state=True; ordinary AdamW still changes moments "
            "at zero LR"
        )
    held_rates = [group.get("lr") for group in optimizer.param_groups]
    if any(
        not isinstance(rate, torch.Tensor) or rate.device.type != "cpu"
        for rate in held_rates
    ):
        raise ValueError(
            "startup diagnostics require captured host-tensor learning rates"
        )
    rates = cast(list[torch.Tensor], held_rates)
    selection = _selection(call)
    state = call._state
    objects = state.bridge.objects
    parameter_aliases = {
        objects.alias_for_object(item.binding.object_id)
        for item in state._registrations()
        if item.binding.parameter
    }
    mutated = {
        objects.alias_for_object(mutation.object_id)
        for record in call._executor.run.execution
        if record.task.phase != "optimizer"
        for mutation in record.task.mutations
    }
    if mutated & parameter_aliases:
        raise ValueError(
            "startup diagnostics cannot preserve model parameters mutated "
            "outside optimizer tasks"
        )
    buffer_aliases = {
        objects.alias_for_object(item.binding.object_id)
        for item in state._registrations()
        if not item.binding.parameter
    } & mutated
    call.synchronize()
    buffers = state._read_model_aliases(aliases=buffer_aliases)
    rng = _checkpoint.rng_state(state.device)
    original_rates = [rate.clone() for rate in rates]
    original_step = call._step
    directory.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "version": 1,
        "training_step": original_step,
        "warmup_steps": warmup,
        "learning_rate": 0.0,
        "buffer_snapshot_bytes": sum(t.numel() for t in buffers.values()),
    }
    try:
        for rate in rates:
            rate.zero_()
        # Prepare trace resources before the warmup and measured invocation.
        call.prepare_runtime_trace()
        print(
            f"startup diagnostics: {warmup} warmup step(s), then one traced step; lr=0",
            flush=True,
        )
        for _ in range(warmup):
            result = call(inputs)
            call.synchronize()
            del result
        started = time.perf_counter()
        result = call(inputs, runtime_trace=True)
        call.synchronize()
        assert result.diagnostics is not None
        diagnostics = result.diagnostics.result().as_dict()
        record["traced_seconds"] = time.perf_counter() - started
        (directory / "step.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        del result
    finally:
        primary_error = sys.exception()
        try:
            try:
                call.synchronize()
                for alias, values in buffers.items():
                    write_spill_tensor(objects, alias, values)
                call.mark_cycle_end()
                # Diagnostic calls are not training throughput samples.
                call.invocation_timings()
            finally:
                for rate, original in zip(rates, original_rates, strict=True):
                    rate.copy_(original)
                call._step = original_step
                _checkpoint.restore_rng(rng, state.device)
        except Exception as cleanup_error:
            if primary_error is None:
                raise
            # An execution failure can close the runtime. Preserve that original
            # diagnosis instead of replacing it with a failed state restoration.
            primary_error.add_note(
                f"Startup diagnostic cleanup also failed: {cleanup_error}"
            )
    from shadowspill.diagnostics.occupancy import attribute, write_pages

    program = call.plan_report.program.to_dict()
    (directory / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (directory / "program.json").write_text(json.dumps(program, indent=2) + "\n")
    pages = write_pages(
        [
            attribute(selection, program),
            attribute(selection, program, diagnostics=diagnostics),
        ],
        directory / "timelines",
        title="Startup diagnostic step (LR=0)",
    )
    record["timeline"] = str(pages[0])
    (directory / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"startup diagnostics complete: {directory / 'summary.json'}", flush=True)
    return record


def _selection(call: PlannedTrainStep) -> dict[str, Any]:
    """Export actual admitted evidence, independent of intermediate candidates."""
    report = call.plan_report
    if report.physical_layout is None:
        raise ValueError("startup timelines require an admitted fixed physical layout")
    result = report.search_result
    return {
        "schema": "training_startup_selection/v1",
        "program_digest": report.program.digest,
        "schedule": report.execution_plan.schedule.to_dict(),
        "selections": [item.to_dict() for item in report.execution_plan.selections],
        "simulation": asdict(result.simulation_config),
        "simulation_result": asdict(result.simulation),
        "admission_certificate": {"layout": report.physical_layout.to_dict()},
    }

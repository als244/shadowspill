"""Profile full prepared training updates, with no capture of setup or warmup.

This experiment uses the Trainer's prepared callable directly so it can pass
the existing per-invocation ``profiler_annotations=True`` option. All model,
optimizer, schedule and data construction remains in the training recipe.
"""

import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist

from shadowspill.training._inputs import make_microbatches
from shadowspill.training._model import model_mode
from shadowspill.training.observations import StepObservations


def profile_trainer(trainer, source, *, rank_dir, steps, warmup, checkpoint, phase):
    rank = dist.get_rank()
    if checkpoint is not None:
        phase("profile_checkpoint_load", checkpoint=str(checkpoint))
        trainer.load(checkpoint)
        # Direct-call execution uses the same optional source capability as fit.
        if trainer._source_state is not None:
            source.load_state_dict(trainer._source_state)
            trainer._source_state = None
        phase("profile_checkpoint_ready", step=trainer.step_count)
    initial_step = trainer.step_count
    execution = trainer._require_prepared()
    call = execution.call
    candidate = trainer.candidates[trainer.selected_candidate]
    rows = []

    def update(kind, offset):
        step = initial_step + offset + 1
        values = {
            name: schedule(step - 1) for name, schedule in trainer.schedules.items()
        }
        with torch.cuda.nvtx.range(f"ep2/{kind}/step_{step:06d}/rank_{rank}"):
            with torch.cuda.nvtx.range("ep2/input_preparation"):
                data = next(source)
                batches = make_microbatches(candidate, data)
            started = time.perf_counter()
            with (
                model_mode(trainer.model, True),
                torch.cuda.nvtx.range("ep2/planned_training_step"),
            ):
                result = call(batches, hyperparams=values, profiler_annotations=True)
                call.synchronize()
            with torch.cuda.nvtx.range("ep2/collect_metrics"):
                observed = StepObservations.collect(
                    result.objectives, result.metrics, result.parameter_metrics
                )
                actual_step = result.step_number
                del result
            elapsed = time.perf_counter() - started
        loss = sum(observed.losses)
        if not math.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at profile update {step}")
        if actual_step != step:
            raise AssertionError((actual_step, step))
        row = dict(
            kind=kind,
            step=step,
            seconds=elapsed,
            loss=loss,
            hyperparams=values,
            microbatches=len(batches),
        )
        rows.append(row)
        phase("profile_step", **row)

    for offset in range(warmup):
        update("warmup", offset)
    call.synchronize()
    torch.cuda.synchronize(trainer.backend.device)
    dist.barrier()
    phase("profile_capture_start", steps=steps, first_step=initial_step + warmup + 1)
    # Each worker must activate its profiler before entering the first NVTX range.
    # A leader-only start can lose another worker's initial enclosing ranges.
    torch.cuda.cudart().cudaProfilerStart()
    dist.barrier()
    try:
        with torch.cuda.nvtx.range(f"ep2/profiled_training/rank_{rank}"):
            for offset in range(warmup, warmup + steps):
                update("training", offset)
            call.synchronize()
            call._finish_profiler_annotations()
        # Finish both ranks' kernels and asynchronous annotations before stopping.
        dist.barrier()
    finally:
        if rank == 0:
            torch.cuda.cudart().cudaProfilerStop()
    phase("profile_capture_stop")
    record = dict(
        passed=True,
        rank=rank,
        initial_step=initial_step,
        warmup_steps=warmup,
        captured_steps=steps,
        final_step=initial_step + warmup + steps,
        checkpoint=None if checkpoint is None else str(checkpoint),
        profiler_annotations=True,
        runtime_trace=False,
        updates=rows,
    )
    Path(rank_dir, "profile-result.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    phase("profile_complete", captured_steps=steps)

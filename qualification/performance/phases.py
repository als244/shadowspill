"""Warm/checkpoint and measure an already loaded Executor; no preparation work."""

import time
from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from qualification.pipeline.observe import Observe
from qualification.runtime_evidence import adapter_statistics, check_physical_budget
from shadowspill import Executor, load_checkpoint, save_checkpoint
from shadowspill.execution.tensors import TensorChunk

from .cases import FullModelManifest


@dataclass(frozen=True)
class WarmStep:
    checkpoint_seconds: float
    restore_seconds: float
    checkpoint_restored: bool | None
    warm_seconds: float
    warm_objectives: list[float]
    warm_diagnostics: dict[str, Any]
    execution_baseline: object


@dataclass(frozen=True)
class Measurements:
    group_seconds: list[float]
    host_group_seconds: list[float]
    group_tokens_per_second: list[float]
    cycle_seconds: list[float]
    dispatch_seconds: list[float]
    measured_objectives: list[list[float]]


def optimizer_steps(executor: Executor) -> int:
    steps = {}

    def collect(chunk: TensorChunk, value: torch.Tensor) -> None:
        if chunk.name.startswith("optimizer.") and chunk.name.endswith(".step"):
            steps[chunk.name] = int(value)

    executor.export_tensors(collect)
    if not steps or len(set(steps.values())) != 1:
        raise AssertionError(f"optimizer counters disagree or are absent: {steps}")
    return next(iter(steps.values()))


def warm_step(
    executor: Executor,
    observer: Observe,
    inputs: Sequence[tuple[Any, ...]],
    arguments: Namespace,
    directory: Path,
    physical_statuses: list[int],
) -> WarmStep:
    checkpoint = directory / "checkpoint.ssp"
    checkpoint_seconds = restore_seconds = 0.0
    restored = None
    if not arguments.skip_checkpoint:
        started = time.perf_counter()
        save_checkpoint(
            executor, checkpoint, metadata={"step": optimizer_steps(executor)}
        )
        checkpoint_seconds = time.perf_counter() - started
    result, summary, trace = observer.traced(inputs, step=0)
    losses = [float(value) for value in result.losses]
    del result
    executor.synchronize()
    physical_statuses.append(check_physical_budget())
    if not arguments.skip_checkpoint:
        started = time.perf_counter()
        metadata = load_checkpoint(executor, checkpoint)
        restored = optimizer_steps(executor) == metadata["step"]
        restore_seconds = time.perf_counter() - started
    observer.idle()
    return WarmStep(
        checkpoint_seconds,
        restore_seconds,
        restored,
        summary["host_seconds"],
        losses,
        trace,
        adapter_statistics(),
    )


def measure_groups(
    executor: Executor,
    observer: Observe,
    inputs: Sequence[tuple[Any, ...]],
    manifest: FullModelManifest,
    arguments: Namespace,
    physical_statuses: list[int],
) -> Measurements:
    measured = Measurements([], [], [], [], [], [])
    for group in range(arguments.groups):
        retained = []
        observer.idle()
        started = time.perf_counter()
        for step in range(arguments.steps_per_group):
            # Close each cycle where the next starts, including terminal copies
            # and caller dispatch costs. No compilation or tuning occurs here.
            observer.idle()
            observer.events[step].record()
            dispatched = time.perf_counter()
            retained.append(executor.run(args=inputs))
            measured.dispatch_seconds.append(time.perf_counter() - dispatched)
        observer.idle()
        observer.events[arguments.steps_per_group].record()
        observer.events[arguments.steps_per_group].synchronize()
        measured.host_group_seconds.append(time.perf_counter() - started)
        cycles = [
            observer.events[i].elapsed_time(observer.events[i + 1]) / 1000
            for i in range(arguments.steps_per_group)
        ]
        measured.cycle_seconds.extend(cycles)
        elapsed = sum(cycles)
        measured.group_seconds.append(elapsed)
        measured.group_tokens_per_second.append(
            manifest.tokens_per_step * len(cycles) / elapsed
        )
        for handle in retained:
            measured.measured_objectives.append(
                [float(value) for value in handle.result().losses]
            )
        del handle, retained
        executor.synchronize()
        observer.idle()
        physical_statuses.append(check_physical_budget())
        print(
            f"{manifest.identity} group {group + 1}/{arguments.groups}: "
            f"{elapsed / len(cycles):.6f}s/step, "
            f"{measured.group_tokens_per_second[-1]:.2f} tokens/s",
            flush=True,
        )
    return measured

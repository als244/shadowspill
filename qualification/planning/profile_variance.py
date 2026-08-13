"""Measure one structural Qwen task under controlled profiling conditions."""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from qualification.numerical.cases import build_case
from shadowspill.memory import device, pinned_host
from shadowspill.pytorch import Runtime
from shadowspill.pytorch.capture.aot import capture_training_objective
from shadowspill.pytorch.capture.fake import fake_cuda_inputs, fake_cuda_model
from shadowspill.pytorch.compilation.compiler import (
    compile_artifact,
    materialize_example_arguments,
)
from shadowspill.pytorch.materialization import representative_cpu_inputs
from shadowspill.pytorch.materialization.training import (
    representative_training_arguments,
)
from shadowspill.pytorch.partition import partition_training_capture
from shadowspill.pytorch.profiling.inputs import (
    materialize_representative_inputs,
)
from shadowspill.pytorch.runtime_adapter.abi import AdapterStatistics


def _arguments(tokens: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if tokens == 96:
        lengths = [17, 31, 48]
    else:
        first = tokens // 4
        lengths = [first, first, tokens - 2 * first]
    return (
        {"max_seq_len": max(192, tokens)},
        [{"token_shape": [1, tokens], "sequence_lengths": lengths}],
    )


def _capture_artifact(tokens: int, stage_index: int) -> Any:
    model_config, geometry = _arguments(tokens)
    case = build_case(
        "qwen35",
        model_implementation="pytorch",
        seed=20_260_811,
        model_config=model_config,
        data_geometry=geometry,
        case_factory=None,
        case_options={},
    )
    cpu_inputs = tuple(representative_cpu_inputs(case.microbatches[0]))
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    fake_model = fake_cuda_model(case.model, fake_mode, device_index=0)
    with fake_mode:
        capture = capture_training_objective(
            fake_model,
            case.objective,
            fake_cuda_inputs(cpu_inputs, fake_mode, device_index=0),
        )
    root_inputs = representative_training_arguments(
        capture,
        case.model,
        cpu_inputs,
    )
    with fake_mode:
        partitioned = partition_training_capture(
            capture,
            partition="auto",
            representative_root_inputs=root_inputs,
        )
    try:
        return partitioned.stages[stage_index].recompute_pair.forward
    except IndexError as exc:
        raise ValueError(
            f"stage index {stage_index} is outside {len(partitioned.stages)} stages"
        ) from exc


def _condition_device(stream: torch.cuda.Stream) -> tuple[int, ...]:
    left = torch.randn((2048, 2048), dtype=torch.bfloat16, device="cuda")
    right = torch.randn_like(left)
    output = torch.empty_like(left)
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    samples: list[int] = []
    for _ in range(64):
        start.record(stream)
        torch.mm(left, right, out=output)
        finish.record(stream)
        finish.synchronize()
        samples.append(round(start.elapsed_time(finish) * 1_000_000))
        if len(samples) >= 3:
            recent = samples[-3:]
            median = statistics.median(recent)
            if median > 0 and (max(recent) - min(recent)) / median <= 0.02:
                break
    del output, right, left
    stream.synchronize()
    return tuple(samples)


def _adapter_stats(runtime: Runtime | None) -> dict[str, int] | None:
    if runtime is None:
        return None
    value = AdapterStatistics()
    library = runtime._installed.library
    status = int(library.shadowspill_pytorch_allocator_statistics(ctypes.byref(value)))
    if status != 0:
        raise RuntimeError(f"allocator statistics failed with status {status}")
    native = value.runtime
    return {
        "allocation_calls": int(value.allocation_callbacks),
        "free_calls": int(value.free_callbacks),
        "zero_byte_allocation_calls": int(value.zero_size_allocation_callbacks),
        "record_stream_calls": int(value.record_stream_callbacks),
        "callback_failures": int(value.callback_failures),
        "physical_checks": int(value.physical_checks),
        "allocation_events": int(native.allocation_events),
        "requested_allocated_bytes": int(native.requested_allocated_bytes),
        "pending_retirements": int(native.pending_retirements),
    }


def _delta(
    before: dict[str, int] | None,
    after: dict[str, int] | None,
) -> dict[str, int] | None:
    if before is None or after is None:
        return None
    return {name: after[name] - value for name, value in before.items()}


def _dispersion(values: list[int]) -> dict[str, float | int]:
    median = float(statistics.median(values))
    deviations = [abs(value - median) for value in values]
    midpoint = len(values) // 2
    return {
        "count": len(values),
        "minimum_ns": min(values),
        "median_ns": median,
        "maximum_ns": max(values),
        "relative_mad": statistics.median(deviations) / median,
        "half_to_half_median_drift": abs(
            statistics.median(values[:midpoint]) - statistics.median(values[-midpoint:])
        )
        / median,
    }


def _measure(
    function: Any,
    arguments: tuple[object, ...],
    *,
    name: str,
    samples: int,
    profiler_annotations: bool,
    warmups: int,
    wait_idle: Any,
) -> dict[str, object]:
    stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    for _ in range(warmups):
        output = function(*arguments)
        del output
    stream.synchronize()
    wait_idle()
    cuda_ns: list[int] = []
    host_ns: list[int] = []
    for index in range(samples):
        if profiler_annotations:
            torch.cuda.nvtx.range_push(
                f"shadowspill.qualification.profile_variance.{name}.{index:04d}"
            )
        start.record(stream)
        host_started = time.perf_counter_ns()
        output = function(*arguments)
        host_ns.append(time.perf_counter_ns() - host_started)
        del output
        finish.record(stream)
        finish.synchronize()
        wait_idle()
        cuda_ns.append(round(start.elapsed_time(finish) * 1_000_000))
        if profiler_annotations:
            torch.cuda.nvtx.range_pop()
    return {
        "cuda": _dispersion(cuda_ns),
        "host_callable": _dispersion(host_ns),
        "cuda_samples_ns": cuda_ns,
        "host_callable_samples_ns": host_ns,
    }


def _device_state() -> dict[str, str] | None:
    """Read coarse device state outside measured regions when nvidia-smi exists."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,clocks.mem,power.draw,pstate,temperature.gpu",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    values = [item.strip() for item in completed.stdout.strip().split(",")]
    if len(values) != 5:
        return {"raw": completed.stdout.strip()}
    return dict(
        zip(
            ("sm_clock_mhz", "memory_clock_mhz", "power_w", "pstate", "temperature_c"),
            values,
            strict=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--allocator", choices=("standard", "shadowspill"), required=True
    )
    parser.add_argument("--tokens", type=int, choices=(96, 1024), default=96)
    parser.add_argument("--stage-index", type=int, default=3)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument(
        "--condition-device",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--profiler-annotations", action="store_true")
    arguments = parser.parse_args()
    if arguments.samples < 10:
        parser.error("--samples must be at least 10")

    runtime: Runtime | None = None
    if arguments.allocator == "shadowspill":
        runtime = Runtime(
            pools={
                "execution": device(physical_capacity=30 << 30),
                "spill": pinned_host(capacity=64 << 30),
            },
            calibrate=False,
        )
    capture_started = time.perf_counter_ns()
    artifact = _capture_artifact(arguments.tokens, arguments.stage_index)
    capture_ns = time.perf_counter_ns() - capture_started
    compile_started = time.perf_counter_ns()
    compiled = compile_artifact(artifact, device_ordinal=0)
    compilation_ns = time.perf_counter_ns() - compile_started
    representative = materialize_representative_inputs(artifact, device_ordinal=0)
    zero_arguments = materialize_example_arguments(
        artifact.example_arguments, device_ordinal=0
    )
    normal_arguments = representative.arguments
    stream = torch.cuda.current_stream()
    state_before = _device_state()
    conditioning = _condition_device(stream) if arguments.condition_device else ()
    state_after_conditioning = _device_state()
    library = None if runtime is None else runtime._installed.library

    def wait_idle() -> None:
        if library is not None:
            status = int(library.shadowspill_pytorch_allocator_wait_idle())
            if status != 0:
                raise RuntimeError(f"allocator failed to become idle: {status}")

    if library is not None and arguments.profiler_annotations:
        status = int(library.shadowspill_pytorch_profiler_annotations_set(1))
        if status != 0:
            raise RuntimeError(f"enabling runtime annotations failed: {status}")
    before = _adapter_stats(runtime)
    zero = _measure(
        compiled.function,
        zero_arguments,
        name="zero",
        samples=arguments.samples,
        profiler_annotations=arguments.profiler_annotations,
        warmups=arguments.warmups,
        wait_idle=wait_idle,
    )
    middle = _adapter_stats(runtime)
    normal = _measure(
        compiled.function,
        normal_arguments,
        name="normal",
        samples=arguments.samples,
        profiler_annotations=arguments.profiler_annotations,
        warmups=arguments.warmups,
        wait_idle=wait_idle,
    )
    after = _adapter_stats(runtime)
    if library is not None and arguments.profiler_annotations:
        status = int(library.shadowspill_pytorch_profiler_annotations_set(0))
        if status != 0:
            raise RuntimeError(f"disabling runtime annotations failed: {status}")

    result = {
        "schema": "shadowspill.profile_variance/v1",
        "allocator": arguments.allocator,
        "tokens": arguments.tokens,
        "stage_index": arguments.stage_index,
        "structural_abi_key": artifact.compatibility_digest,
        "semantic_contract_digest": artifact.storage_contract.compatibility_digest,
        "operator_count": len(artifact.operator_targets),
        "fx_node_count": len(tuple(artifact.graph_module.graph.nodes)),
        "capture_ns": capture_ns,
        "compilation_ns": compilation_ns,
        "warmups": arguments.warmups,
        "device_conditioning_samples_ns": conditioning,
        "device_state_before": state_before,
        "device_state_after_conditioning": state_after_conditioning,
        "device_state_after_measurement": _device_state(),
        "zero": zero,
        "normal": normal,
        "zero_allocator_delta": _delta(before, middle),
        "normal_allocator_delta": _delta(middle, after),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    del representative, normal_arguments, zero_arguments, compiled, artifact
    if runtime is not None:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Capture one unchanged ShadowSpill execution with internal and NSYS traces."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from qualification.model_state import externalize_case_model, relocate_case_model
from qualification.numerical.cases import build_case
from shadowspill.memory import device, pinned_host
from shadowspill.pytorch import (
    Runtime,
    plan_step,
)


def _event_bracket(
    training: object, microbatches: list[list[object]]
) -> dict[str, float]:
    stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    training._arm_selected_span_timing()  # type: ignore[attr-defined]
    wall_start = time.perf_counter()
    start.record(stream)
    training(microbatches, runtime_trace=False)  # type: ignore[operator]
    end.record(stream)
    selected_task_seconds = training._collect_selected_span_seconds()  # type: ignore[attr-defined]
    end.synchronize()
    return {
        "compute_seconds": float(start.elapsed_time(end)) / 1e3,
        "selected_task_seconds": selected_task_seconds,
        "host_and_compute_seconds": time.perf_counter() - wall_start,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--family", default="qwen35")
    parser.add_argument(
        "--model-implementation",
        choices=("pytorch", "mlops"),
        default="pytorch",
    )
    parser.add_argument("--device-budget", type=int, default=30 << 30)
    parser.add_argument("--trace-repetitions", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.trace_repetitions < 1:
        parser.error("--trace-repetitions must be positive")

    case = build_case(
        arguments.family,
        model_implementation=arguments.model_implementation,
        seed=20_260_811,
        model_config={},
        data_geometry=None,
        case_factory=None,
        case_options={},
    )
    with case.implementations():
        runtime = Runtime(
            pools={
                "execution": device(physical_capacity=arguments.device_budget),
                "spill": pinned_host(capacity=64 << 30),
            }
        )
        case = relocate_case_model(case, runtime=runtime)
        model = case.model
        planning_start = time.perf_counter()
        training = plan_step(
            model,
            objective=case.objective,
            opt=case.optimizer,
            example_inputs=case.microbatches,
            runtime=runtime,
            execution="execution",
            spill="spill",
        )
        planning_seconds = time.perf_counter() - planning_start

        # Warm the recurrent execution and the lazy trace resources outside NSYS.
        training(case.microbatches, runtime_trace=False)
        torch.cuda.synchronize()
        untraced = [_event_bracket(training, case.microbatches) for _ in range(3)]
        warm_trace = training(case.microbatches, runtime_trace=True)
        assert warm_trace.diagnostics is not None
        warm_trace.diagnostics.result()

        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("shadowspill.qualification.phase1_capture")
        traced_samples: list[dict[str, float]] = []
        traced_wall_start = time.perf_counter()
        diagnostics = None
        for _ in range(arguments.trace_repetitions):
            sample_wall_start = time.perf_counter()
            traced = training(
                case.microbatches,
                runtime_trace=True,
                profiler_annotations=True,
            )
            assert traced.diagnostics is not None
            diagnostics = traced.diagnostics.result()
            traced_samples.append(
                {
                    "selected_task_seconds": diagnostics.timing.compute_seconds,
                    "task_interval_sum_seconds": sum(
                        value for _, value in diagnostics.timing.phase_gpu_seconds
                    ),
                    "host_call_seconds": diagnostics.timing.host_call_seconds,
                    "wall_seconds": time.perf_counter() - sample_wall_start,
                }
            )
        assert diagnostics is not None
        traced_wall_seconds = time.perf_counter() - traced_wall_start
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()

        report = training.plan_report
        training.close()
        externalize_case_model(case, runtime=runtime)
        runtime.close()

    result = {
        "schema": "shadowspill.phase1_execution_control/v1",
        "family": arguments.family,
        "model_implementation": arguments.model_implementation,
        "device_budget_bytes": arguments.device_budget,
        "planning_seconds": planning_seconds,
        "untraced_steps": untraced,
        "untraced_compute_median_seconds": statistics.median(
            item["compute_seconds"] for item in untraced
        ),
        "untraced_selected_task_median_seconds": statistics.median(
            item["selected_task_seconds"] for item in untraced
        ),
        "traced_wall_seconds": traced_wall_seconds,
        "traced_samples": traced_samples,
        "trace": diagnostics.as_dict(),
        "plan": {
            "program_digest": report.execution_plan.program.digest,
            "schedule_digest": report.execution_plan.schedule.digest,
            "predicted_makespan_seconds": report.predicted_makespan_ns / 1e9,
            "predicted_device_peak_bytes": report.predicted_device_peak_bytes,
            "task_count": len(
                report.execution_plan.program.selected_tasks(
                    report.execution_plan.selections
                )
            ),
            "action_count": len(report.transfer_actions),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "trace"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

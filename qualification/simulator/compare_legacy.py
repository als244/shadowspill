#!/usr/bin/env python3
"""Compare randomized ShadowSpill schedules with the external legacy oracle."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from pathlib import Path

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    ObjectRole,
    ObjectSpec,
    Program,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    TaskProfile,
    TaskSpec,
)
from shadowspill.simulator import SimulationConfig, simulate

ORACLE_SOURCE = r"""
import json
import sys
from dataflow_sim.core.schema import TaskChain
from dataflow_sim.engine.simulator import run

cases = json.load(sys.stdin)
results = []
for case in cases:
    log = run(TaskChain.from_dict(case))
    intervals = [
        [item.task_id, int(item.start), int(item.end), item.track]
        for item in log.task_intervals
    ]
    intervals.sort(key=lambda item: (item[1], item[2], item[0]))
    results.append({
        "intervals": intervals,
        "device_peak": log.peak_fast_memory_bytes,
        "host_peak": log.peak_backing_memory_bytes,
    })
json.dump(results, sys.stdout, sort_keys=True)
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-python", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def make_case(rng: random.Random, index: int) -> dict[str, object]:
    activation = rng.randint(1, 4096)
    output = rng.randint(0, 2048)
    runtimes = [rng.randint(0, 10000) for _ in range(4)]
    workspaces = [rng.randint(0, 1024) for _ in range(4)]
    h2d_bandwidth = rng.randint(1, 128)
    d2h_bandwidth = rng.randint(1, 128)
    capacity = 64 + activation + output + max(workspaces)
    return {
        "id": index,
        "activation": activation,
        "output": output,
        "runtimes": runtimes,
        "workspaces": workspaces,
        "h2d_bandwidth": h2d_bandwidth,
        "d2h_bandwidth": d2h_bandwidth,
        "initial_memory": [{"id": "input", "size": 64, "location": "fast"}],
        "tasks": [
            {
                "id": "produce",
                "inputs": ["input"],
                "outputs": [{"id": "activation", "size": activation}],
                "runtime": runtimes[0],
                "workspace_bytes": workspaces[0],
                "offload_after": ["activation"],
            },
            {
                "id": "middle",
                "inputs": ["input"],
                "outputs": [],
                "runtime": runtimes[1],
                "workspace_bytes": workspaces[1],
                "prefetch_after": ["activation"],
            },
            {
                "id": "spacer",
                "inputs": ["input"],
                "outputs": [],
                "runtime": runtimes[2],
                "workspace_bytes": workspaces[2],
            },
            {
                "id": "consume",
                "inputs": ["activation"],
                "outputs": [{"id": "output", "size": output}],
                "runtime": runtimes[3],
                "workspace_bytes": workspaces[3],
                "releases_after": ["activation"],
            },
        ],
        "final_locations": {"output": "fast"},
        "fast_memory_capacity": capacity,
        "backing_memory_capacity": activation,
        "bandwidth_from_slow": h2d_bandwidth,
        "bandwidth_to_slow": d2h_bandwidth,
    }


def shadowspill_case(case: dict[str, object]) -> dict[str, object]:
    activation = int(case["activation"])
    output = int(case["output"])
    runtimes = [int(value) for value in case["runtimes"]]  # type: ignore[union-attr]
    workspaces = [int(value) for value in case["workspaces"]]  # type: ignore[union-attr]
    resource = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    program = Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=(
            AliasGroupSpec("input_storage", "cuda_0", 64),
            AliasGroupSpec("activation_storage", "cuda_0", activation),
            AliasGroupSpec("output_storage", "cuda_0", output),
        ),
        objects=(
            ObjectSpec("input", "input_storage", 0, 64, ObjectRole.INPUT),
            ObjectSpec(
                "activation",
                "activation_storage",
                0,
                activation,
                ObjectRole.ACTIVATION,
            ),
            ObjectSpec("output", "output_storage", 0, output, ObjectRole.OUTPUT),
        ),
        profiles=tuple(
            TaskProfile(f"profile_{i}", runtimes[i], workspaces[i], f"abi_{i}")
            for i in range(4)
        ),
        tasks=(
            TaskSpec(
                "produce",
                resource,
                "profile_0",
                inputs=("input",),
                outputs=("activation",),
            ),
            TaskSpec(
                "middle",
                resource,
                "profile_1",
                dependencies=("produce",),
                inputs=("input",),
            ),
            TaskSpec(
                "spacer",
                resource,
                "profile_2",
                dependencies=("middle",),
                inputs=("input",),
            ),
            TaskSpec(
                "consume",
                resource,
                "profile_3",
                dependencies=("produce", "spacer"),
                inputs=("activation",),
                outputs=("output",),
            ),
        ),
    )
    schedule = MemorySchedule(
        initial_residency=(ResidencySpec("input_storage", MemoryLocation.DEVICE),),
        actions=(
            MemoryAction("produce", "activation_storage", MemoryActionKind.OFFLOAD),
            MemoryAction("middle", "activation_storage", MemoryActionKind.PREFETCH),
            MemoryAction("consume", "activation_storage", MemoryActionKind.RELEASE),
        ),
        final_residency=(ResidencySpec("output_storage", MemoryLocation.DEVICE),),
    )
    config = SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=int(case["fast_memory_capacity"]),
        host_capacity_bytes=int(case["backing_memory_capacity"]),
        h2d_bandwidth_bytes_per_second=(
            int(case["bandwidth_from_slow"]) * 1_000_000_000
        ),
        d2h_bandwidth_bytes_per_second=(int(case["bandwidth_to_slow"]) * 1_000_000_000),
    )
    result = simulate(program, schedule, config=config)
    intervals = [
        [item.task_id, item.start_ns, item.end_ns, "compute"]
        for item in result.task_intervals
    ]
    intervals.extend(
        [
            (
                "from_slow:activation"
                if item.direction.value == "host_to_device"
                else "to_slow:activation"
            ),
            item.start_ns,
            item.end_ns,
            ("from_slow" if item.direction.value == "host_to_device" else "to_slow"),
        ]
        for item in result.transfer_intervals
    )
    intervals.sort(key=lambda item: (int(item[1]), int(item[2]), str(item[0])))
    return {
        "intervals": intervals,
        "device_peak": result.device_peak("cuda_0").total_bytes,
        "host_peak": result.host_peak_bytes,
    }


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    cases = [make_case(rng, index) for index in range(args.cases)]
    oracle_cases = [
        {
            key: value
            for key, value in case.items()
            if key
            not in {
                "id",
                "activation",
                "output",
                "runtimes",
                "workspaces",
                "h2d_bandwidth",
                "d2h_bandwidth",
            }
        }
        for case in cases
    ]
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONHOME", "PYTHONPATH"}
    }
    completed = subprocess.run(
        [str(args.legacy_python), "-c", ORACLE_SOURCE],
        cwd=args.legacy_root,
        env=environment,
        input=json.dumps(oracle_cases),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)
    expected = json.loads(completed.stdout)
    for index, (case, oracle) in enumerate(zip(cases, expected, strict=True)):
        actual = shadowspill_case(case)
        if actual != oracle:
            raise AssertionError(
                f"case {index} diverged\nexpected={oracle}\nactual={actual}"
            )
    print(f"{len(cases)} randomized external simulator comparisons passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare frozen PressureFit chains with the external legacy oracle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from shadowspill.planner import pressurefit  # noqa: E402
from tests.planner._examples import (  # noqa: E402
    training_chain_config,
    training_chain_initial,
    training_chain_program,
)

ORACLE_SOURCE = r"""
import json
import sys
from dataflow_sim.policies.pressurefit import plan_pressurefit_policy
from tests.dataflow_sim.chain_fixtures import build_bare_training_chain

results = []
for case in json.load(sys.stdin):
    chain, diagnostic = plan_pressurefit_policy(
        build_bare_training_chain(case["layers"]),
        fast_memory_capacity=case["capacity"],
        preplace="greedy",
    )
    actions = []
    for task in chain.tasks:
        actions.extend([task.id, "release", item] for item in task.releases_after)
        actions.extend([task.id, "offload", item.obj_id] for item in task.offload_after)
        actions.extend(
            [task.id, "prefetch", item.obj_id] for item in task.prefetch_after
        )
    results.append({
        "candidate": diagnostic.selected_candidate,
        "makespan_ns": int(diagnostic.selected_makespan_us * 1000),
        "initial_device": [
            item.id for item in chain.initial_memory if item.location == "fast"
        ],
        "actions": actions,
    })
json.dump(results, sys.stdout, sort_keys=True)
"""

CASES = (
    {"layers": 1, "capacity": 224},
    {"layers": 2, "capacity": 224},
    {"layers": 5, "capacity": 800},
    {"layers": 10, "capacity": 500},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-python", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONHOME", "PYTHONPATH"}
    }
    completed = subprocess.run(
        [str(args.legacy_python), "-c", ORACLE_SOURCE],
        cwd=args.legacy_root,
        env=environment,
        input=json.dumps(CASES),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)
    legacy_results = json.loads(completed.stdout)
    reports: list[dict[str, object]] = []
    for case, legacy in zip(CASES, legacy_results, strict=True):
        layers = case["layers"]
        result = pressurefit(
            training_chain_program(layers),
            initial_residency=training_chain_initial(layers),
            config=training_chain_config(case["capacity"]),
        )
        current = {
            "candidate": result.diagnostics.selected_candidate_id,
            "makespan_ns": result.simulation.makespan_ns,
            "initial_device": [
                item.alias_group_id
                for item in result.schedule.initial_residency
                if item.location.value == "device"
            ],
            "actions": [
                [
                    item.trigger_task_id,
                    item.kind.value,
                    item.alias_group_id,
                ]
                for item in result.schedule.actions
            ],
        }
        task_order = {
            task.task_id: index
            for index, task in enumerate(training_chain_program(layers).tasks)
        }

        def semantic_plan(
            value: dict[str, object],
            order: dict[str, int] = task_order,
        ) -> tuple[object, ...]:
            raw_actions = value.get("actions")
            if not isinstance(raw_actions, list):
                raise ValueError("plan actions must be a list")
            actions: list[tuple[str, str, str]] = []
            for raw_action in raw_actions:
                if (
                    not isinstance(raw_action, list)
                    or len(raw_action) != 3
                    or not all(isinstance(item, str) for item in raw_action)
                ):
                    raise ValueError("each plan action must contain three strings")
                trigger, kind, object_id = raw_action
                actions.append((trigger, kind, object_id))
            initial_device = value.get("initial_device")
            if not isinstance(initial_device, list) or not all(
                isinstance(item, str) for item in initial_device
            ):
                raise ValueError("plan initial_device must be a list of strings")
            makespan_ns = value.get("makespan_ns")
            if isinstance(makespan_ns, bool) or not isinstance(makespan_ns, int):
                raise ValueError("plan makespan_ns must be an integer")
            actions = sorted(
                actions,
                key=lambda item: (order[item[0]], item[1], item[2]),
            )
            return (
                value.get("candidate"),
                makespan_ns,
                tuple(sorted(initial_device)),
                tuple(actions),
            )

        exact_plan = semantic_plan(current) == semantic_plan(legacy)
        current_makespan = semantic_plan(current)[1]
        legacy_makespan = semantic_plan(legacy)[1]
        if not isinstance(current_makespan, int) or not isinstance(
            legacy_makespan, int
        ):
            raise AssertionError("validated plan makespans must be integers")
        if current_makespan > legacy_makespan:
            raise AssertionError(f"{layers}-layer makespan regressed")
        if layers in (5, 10) and not exact_plan:
            raise AssertionError(
                f"{layers}-layer plan diverged\nlegacy={legacy}\ncurrent={current}"
            )
        reports.append(
            {
                "layers": layers,
                "legacy_makespan_ns": legacy["makespan_ns"],
                "shadowspill_makespan_ns": current["makespan_ns"],
                "exact_plan": exact_plan,
            }
        )
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

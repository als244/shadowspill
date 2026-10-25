"""Finalize opaque Triton provider choices using one stage's ordinary tensors."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import torch
from torch.utils._pytree import tree_flatten

from shadowspill.artifacts._values import digest
from shadowspill.artifacts.program import Program
from shadowspill.artifacts.tasks import stage_jobs
from shadowspill.pytorch.executable import restore_executable
from shadowspill.pytorch.task_inputs import prepare_inputs
from shadowspill.pytorch.tuning import capture_choices


def finalize_providers(program: Program, device: torch.device) -> Program:
    dependencies = program.bindings["operator_dependencies"]
    pending = {
        key
        for key, values in dependencies.items()
        if any(not item["operator"].startswith("shadowspill::") for item in values)
    }
    if not pending:
        return program
    choices: dict[str, Any] = {}
    functions: dict[str, Any] = {}

    def invoke(key: str, arguments: Sequence[Any]) -> Any:
        if key not in functions:
            functions[key] = restore_executable(
                program.executables[key], dependencies=dependencies.get(key, ())
            )
        if key not in pending:
            return functions[key](*arguments)
        with capture_choices() as captured:
            result = functions[key](*arguments)
        choices[key] = captured
        pending.remove(key)
        return result

    with (
        torch.cuda.device(device),
        torch.random.fork_rng(devices=[device]),
        torch.no_grad(),
    ):
        for job in stage_jobs(program, None):
            if "forward" not in job:
                component = job["optimizer"] if "optimizer" in job else job["gradient"]
                key = component["profile_key"]
                if key in pending:
                    arguments = prepare_inputs(program, key, device)
                    invoke(key, arguments)
                    del arguments
                continue
            forward, backward = job["forward"], job.get("backward")
            key = forward["profile_key"]
            backward_key = None if backward is None else backward["profile_key"]
            if key not in pending and backward_key not in pending:
                continue
            arguments = prepare_inputs(program, key, device)
            leaves, _ = tree_flatten(invoke(key, arguments))
            if backward_key in pending:
                assert backward is not None and backward_key is not None
                values = {
                    slot["object_id"]: arguments[slot["leaf_index"]]
                    for slot in forward["inputs"]
                }
                values.update(
                    {
                        slot["object_id"]: leaves[slot["leaf_index"]]
                        for slot in forward["outputs"]
                    }
                )
                backward_arguments = prepare_inputs(
                    program,
                    backward_key,
                    device,
                    slots=backward["inputs"],
                    values=values,
                )
                invoke(backward_key, backward_arguments)
                del backward_arguments, values
            del arguments, leaves
        torch.cuda.synchronize(device)
    if pending:
        raise ValueError(
            f"provider finalization did not cover tasks: {sorted(pending)}"
        )
    return replace(
        program,
        task_definitions=tuple(
            replace(
                item,
                compatibility_digest=digest(
                    {
                        "task": item.compatibility_digest,
                        "triton_choices": choices[item.profile_key],
                    }
                ),
            )
            if item.profile_key in choices
            else item
            for item in program.task_definitions
        ),
        bindings={**program.bindings, "triton_choices": choices},
    )

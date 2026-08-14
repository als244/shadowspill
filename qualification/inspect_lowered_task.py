"""Inspect one lowered training task and its frontend-owned fixed inputs."""

from __future__ import annotations

import argparse

from qualification.model_state import externalize_case_model, relocate_case_model
from qualification.numerical.cases import build_case
from shadowspill.memory import device, pinned_host
from shadowspill.pytorch import (
    Runtime,
    plan_step,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="qwen35")
    parser.add_argument("--task", default="task_000017")
    parser.add_argument("--device-budget", type=int, default=30 << 30)
    arguments = parser.parse_args()

    case = build_case(
        arguments.family,
        model_implementation="pytorch",
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
        training = plan_step(
            model,
            objective=case.objective,
            opt=case.optimizer,
            example_inputs=case.microbatches,
            runtime=runtime,
            execution="execution",
            spill="spill",
        )
        run = training._executor._recurrent
        fixed = {item.object_id: item for item in run.lowered.fixed_tensors}
        entrypoint = next(
            item for item in run.entrypoints if item.task_id == arguments.task
        )
        print(
            "entrypoint",
            entrypoint.task_id,
            entrypoint.phase,
            entrypoint.stage_index,
            entrypoint.variant,
        )
        for slot in entrypoint.input_slots:
            item = fixed.get(slot.object_id)
            if item is None:
                continue
            value = item.value
            print(
                "fixed-input",
                slot.leaf_index,
                slot.object_id,
                repr(value),
                value.dtype,
                value.device,
                tuple(value.shape),
                value.requires_grad,
            )
        if entrypoint.artifact is None:
            raise RuntimeError("selected entrypoint has no graph artifact")
        print(entrypoint.artifact.graph_module.graph)
        training.close()
        externalize_case_model(case, runtime=runtime)
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

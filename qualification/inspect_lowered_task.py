"""Inspect one lowered training task and its frontend-owned fixed inputs."""

from __future__ import annotations

import argparse

from qualification.numerical.cases import build_case
from shadowspill.pytorch import plan


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
        training = plan(
            case.model,
            objective=case.objective,
            opt=case.optimizer,
            example_inputs=case.microbatches,
            device_budget=arguments.device_budget,
            host_budget=64 << 30,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Save a case's Program and exact initial values; no execution allocator."""

from pathlib import Path
from typing import Any

import torch

from shadowspill import build_step
from workloads.models.shared.training import LEARNING_RATE


def build_case(
    case: Any, directory: Path, *, dtypes: Any, deterministic: bool = False
) -> None:
    optimizer = case.optimizer(case.model.parameters(), lr=torch.tensor(LEARNING_RATE))
    named = dict(case.model.named_parameters())
    names = {id(value): name for name, value in named.items()}
    # Include nonpersistent buffers (e.g. rotary tables), while retaining the
    # normal state_dict inventory for the independent reference comparison.
    tensors = {
        **dict(case.model.named_buffers(remove_duplicate=False)),
        **dict(case.model.named_parameters(remove_duplicate=False)),
    }
    groups = optimizer.state_dict()["param_groups"]
    for group in groups:
        group["lr"] = LEARNING_RATE
    torch.save(
        {
            "tensors": tensors,
            "model_keys": list(case.model.state_dict()),
            "parameter_names": [
                names[id(p)] for g in optimizer.param_groups for p in g["params"]
            ],
            "param_groups": groups,
            "args": [tuple(values) for values in case.microbatches],
        },
        directory / "inputs.pt",
    )
    program = build_step(
        case.model,
        optimizer=optimizer,
        objective=case.objective,
        example_calls=[(tuple(values), {}) for values in case.microbatches],
        fuse_narrow_gradient_accumulation=not deterministic,
        **dtypes.plan_arguments(),
    )
    program.save(directory / "program.ssp")
    print(
        f"built {len(program.tasks)} tasks; "
        f"{len(program.task_definitions)} unique forms",
        flush=True,
    )

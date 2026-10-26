"""Compute weights, stored gradients and optimizer masters retain distinct dtypes."""

import pytest
import torch
from torch import nn

from shadowspill import build_step
from tests.shadowspill.execution.optimizers.common import run_worker, save_plan


def objective(model, x):
    result = model(x).float().sum()
    return result, {"sum": result.detach()}


@pytest.mark.cuda
@pytest.mark.parametrize(
    "weight_dtype,master_dtype,gradient_dtype",
    [
        (torch.bfloat16, torch.float32, None),
        (torch.bfloat16, torch.float32, torch.float32),
        (torch.float32, torch.float64, torch.float64),
    ],
    ids=["bf16_native_grad", "bf16_fp32_grad", "fp32_fp64_master"],
)
def test_mixed_precision_master_update_and_checkpoint(
    tmp_path, weight_dtype, master_dtype, gradient_dtype
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    with torch.device("meta"):
        template = nn.Linear(8, 8, bias=False, dtype=weight_dtype)
    optimizer = torch.optim.AdamW(template.parameters(), lr=0.001, foreach=False)
    examples = [
        ((torch.full((4, 8), value, dtype=weight_dtype),), {})
        for value in (0.125, 0.25)
    ]
    program = build_step(
        template,
        optimizer=optimizer,
        objective=objective,
        example_args=examples[0][0],
        num_examples=2,
        partition="whole",
        master_dtype=master_dtype,
        grad_dtype=gradient_dtype,
    )
    assert template.weight.dtype == weight_dtype and not optimizer.state
    assert optimizer.param_groups[0]["params"][0] is template.weight
    bindings = {item["name"]: item for item in program.bindings["optimizer_bindings"]}
    assert bindings["weight"]["dtype"] == str(master_dtype)
    assert bindings["compute.weight"]["dtype"] == str(weight_dtype)
    assert bindings["gradient.weight"]["dtype"] == str(gradient_dtype or weight_dtype)
    assert bindings["optimizer.weight.exp_avg"]["dtype"] == str(master_dtype)
    save_plan(program, tmp_path)
    reference = nn.Linear(8, 8, bias=False, dtype=weight_dtype)
    with torch.no_grad():
        reference.weight.copy_(torch.arange(64).reshape(8, 8) / 64)
    master = nn.Parameter(reference.weight.to(master_dtype).detach().clone())
    opt = torch.optim.AdamW([master], lr=0.001, foreach=False)
    initial = {"weight": reference.weight.detach().clone()}
    expected, parameter_metrics = [], []
    for index in range(3):
        gradient = torch.zeros_like(
            reference.weight, dtype=gradient_dtype or weight_dtype
        )
        outputs = []
        for args, kwargs in examples:
            reference.zero_grad(set_to_none=True)
            loss, metrics = objective(reference, *args, **kwargs)
            loss.backward()
            gradient.add_(reference.weight.grad.to(gradient.dtype))
            outputs.append((loss.detach(), metrics))
        master.grad = gradient.to(master_dtype)
        parameter_metrics.append(
            {
                "weight": {
                    "param_norm": torch.linalg.vector_norm(
                        reference.weight.detach(), dtype=torch.float32
                    ),
                    "grad_norm": torch.linalg.vector_norm(
                        gradient.float(), dtype=torch.float32
                    ),
                }
            }
            if index != 1
            else None
        )
        opt.step()
        with torch.no_grad():
            reference.weight.copy_(master)
        state = {
            "weight": reference.weight.detach().clone(),
            "master.weight": master.detach().clone(),
            **{
                f"optimizer.weight.{name}": value.clone()
                for name, value in opt.state[master].items()
            },
        }
        expected.append((outputs, state))
    torch.save(
        dict(
            initial=initial,
            examples=examples,
            expected=expected,
            hyperparams=[{}, {}, {}],
            accumulate=[False] * 3,
            restore=False,
            parameter_metrics=parameter_metrics,
        ),
        tmp_path / "oracle.pt",
    )
    run_worker(tmp_path)

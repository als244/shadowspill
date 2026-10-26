"""A fixed provider kernel can compile lazily without changing task allocation."""

import pytest
import torch
from torch import nn

from shadowspill import build_step
from tests.shadowspill.execution.optimizers.common import run_worker, save_plan


def objective(model, features):
    return model(features).float().square().mean()


@pytest.mark.cuda
def test_mlops_adamw_cold_worker_matches_independent_torch_updates(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    AdamW = pytest.importorskip("mlops.optim").AdamW
    rates = [0.001, 0.0005, 0.00025]
    examples = [((torch.ones(4, 8, dtype=torch.bfloat16),), {})]
    with torch.device("meta"):
        template = nn.Sequential(nn.Linear(8, 8, dtype=torch.bfloat16))
    program = build_step(
        template,
        optimizer=AdamW(
            template.parameters(),
            lr=torch.tensor(rates[0]),
            gradient_dtype=torch.float32,
            opt_state_dtype=torch.float32,
        ),
        objective=objective,
        example_args=examples[0][0],
        grad_dtype=torch.float32,
        master_dtype=torch.float32,
    )
    save_plan(program, tmp_path)

    reference = nn.Sequential(nn.Linear(8, 8, dtype=torch.bfloat16))
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.fill_(0.125)
    initial = {name: tensor.clone() for name, tensor in reference.state_dict().items()}
    masters = [nn.Parameter(p.detach().float().clone()) for p in reference.parameters()]
    optimizer = torch.optim.AdamW(masters, lr=rates[0], foreach=False)
    expected = []
    for rate in rates:
        optimizer.param_groups[0]["lr"] = rate
        reference.zero_grad(set_to_none=True)
        loss = objective(reference, *examples[0][0])
        loss.backward()
        for parameter, master in zip(reference.parameters(), masters, strict=True):
            master.grad = parameter.grad.float()
        optimizer.step()
        state = {
            "optimizer_group.0.lr": torch.tensor(rate),
            "optimizer_group.0.betas.0": torch.tensor(0.9, dtype=torch.float64),
            "optimizer_group.0.betas.1": torch.tensor(0.999, dtype=torch.float64),
            "optimizer_group.0.eps": torch.tensor(1e-8, dtype=torch.float64),
            "optimizer_group.0.weight_decay": torch.tensor(0.01, dtype=torch.float64),
            "optimizer_group.0.rounding_salts.0": torch.tensor(0),
            "optimizer_group.0.rounding_salts.1": torch.tensor(1),
        }
        with torch.no_grad():
            for (name, parameter), master in zip(
                reference.named_parameters(), masters, strict=True
            ):
                parameter.copy_(master)
                state[name] = parameter.clone()
                state[f"master.{name}"] = master.clone()
                for key, value in optimizer.state[master].items():
                    state[f"optimizer.{name}.{key}"] = (
                        value.to(torch.int64) if key == "step" else value.clone()
                    )
        expected.append(([(loss.detach(), None)], state))
    torch.save(
        dict(
            initial=initial,
            examples=examples,
            expected=expected,
            hyperparams=[{"optimizer_group.0.lr": rate} for rate in rates],
            accumulate=[False] * 3,
            restore=False,
            # Fixed BLOCK=1024 and launch geometry, no tuning or device scratch.
            # Cold compilation is permitted; changing the task is not.
            lazy_kernels=["_adamw_torch_kernel"],
        ),
        tmp_path / "oracle.pt",
    )
    run_worker(tmp_path)

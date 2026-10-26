"""Weights, accumulated gradients and moments keep independent precision choices."""

from functools import partial

import pytest
import torch

from shadowspill import build_step
from tests.shadowspill.execution.helpers import run_worker, save_plan
from tests.shadowspill.execution.optimizers.reference import Network, objective, oracle


@pytest.mark.cuda
@pytest.mark.parametrize(
    "master_dtype", [None, torch.float32], ids=["no_master", "fp32_master"]
)
@pytest.mark.parametrize(
    "gradient_dtype", [torch.bfloat16, torch.float32], ids=["bf16_grad", "fp32_grad"]
)
@pytest.mark.parametrize(
    "state_dtype", [torch.bfloat16, torch.float32], ids=["bf16_moments", "fp32_moments"]
)
def test_saved_adam_precision_matrix(
    tmp_path, master_dtype, gradient_dtype, state_dtype
):
    check_case(tmp_path, master_dtype, gradient_dtype, state_dtype, mixed=False)


@pytest.mark.cuda
def test_mixed_parameters_only_create_masters_where_needed(tmp_path):
    check_case(tmp_path, torch.float32, torch.float32, torch.float32, mixed=True)


def check_case(directory, master_dtype, gradient_dtype, state_dtype, *, mixed):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    AdamW = pytest.importorskip("mlops.optim").AdamW
    optimizer_factory = partial(
        AdamW,
        lr=0.003,
        gradient_dtype="parameter",
        opt_state_dtype=state_dtype,
    )
    examples = [
        ((torch.full((rows, 8), value, dtype=torch.bfloat16),), {})
        for rows, value in ((2, 0.125), (3, 0.3125))
    ]
    with torch.device("meta"):
        template = Network(mixed=mixed)
    optimizer = optimizer_factory(template.parameters())
    program = build_step(
        template,
        objective=objective,
        optimizer=optimizer,
        example_calls=examples,
        # Compare the separate BF16 product/add rounding of ordinary autograd.
        # Fused accumulation has its own graph-pair/operator numerical tests.
        fuse_narrow_gradient_accumulation=False,
        master_dtype=master_dtype,
        grad_dtype=gradient_dtype,
    )
    assert not optimizer.state and all(p.is_meta for p in template.parameters())
    bindings = {b["name"]: b for b in program.bindings["optimizer_bindings"]}
    for name, weight in template.named_parameters():
        assert bindings[name]["dtype"] == str(master_dtype or weight.dtype)
        assert bindings[f"gradient.{name}"]["dtype"] == str(gradient_dtype)
        for key in ("exp_avg", "exp_avg_sq"):
            assert bindings[f"optimizer.{name}.{key}"]["dtype"] == str(state_dtype)
        has_master = master_dtype is not None and weight.dtype != master_dtype
        assert (f"compute.{name}" in bindings) == has_master
    # Keep the optimizer partition coverage from the former monolithic API test.
    assert len(program.bindings["optimizer_templates"]) == 2
    save_plan(program, directory)
    torch.save(
        oracle(
            Network(mixed=mixed),
            examples,
            optimizer_factory,
            master_dtype=master_dtype,
            gradient_dtype=gradient_dtype,
        ),
        directory / "oracle.pt",
    )
    run_worker(directory)

"""Compile local shard updates once; collectives are not part of their math."""

import pytest
import torch
from torch import nn

from shadowspill import Parallelism
from shadowspill.build.common import compile_artifacts
from shadowspill.build.optimizer import capture_update
from shadowspill.build.optimizer_parameters import local_optimizer_parameters
from shadowspill.build.parallelism import model_parallelism
from shadowspill.build.training_capture import capture_training_model
from shadowspill.profiling.task import measure_task
from shadowspill.pytorch.executable import restore_executable
from shadowspill.pytorch.optimizer.sandbox import (
    copy_optimizer_to_meta,
    optimizer_parameters,
)


def owned_parameters(model, optimizer, *, master_dtype):
    named = dict(model.named_parameters())
    _, declarations = model_parallelism(model, Parallelism.dp(ranks_per_replica=4))
    copied, names = copy_optimizer_to_meta(
        optimizer, {id(value): name for name, value in named.items()}
    )
    return local_optimizer_parameters(
        copied,
        names,
        named,
        set(named),
        master_dtype,
        {item.name: item for item in declarations},
    )


def test_shards_remain_meta_and_optimizer_capability_is_required():
    model = nn.Linear(7, 5, bias=False, dtype=torch.bfloat16, device="meta")
    original = torch.optim.AdamW(model.parameters())
    optimizer, names, copies = owned_parameters(
        model, original, master_dtype=torch.float32
    )
    (parameter,) = optimizer_parameters(optimizer)
    assert parameter.shape == (9,)
    assert parameter.dtype == torch.float32 and parameter.is_meta
    assert copies["weight"].shape == (9,) and copies["weight"].is_meta
    assert copies["weight"].dtype == torch.bfloat16
    assert names[id(parameter)] == "weight"
    assert original.param_groups[0]["params"][0] is model.weight
    assert model.weight.shape == (5, 7)
    assert not original.state
    same_dtype, _, copies = owned_parameters(model, original, master_dtype=None)
    assert optimizer_parameters(same_dtype)[0].dtype == torch.bfloat16
    assert not copies
    with pytest.raises(TypeError, match="independent elementwise updates"):
        owned_parameters(
            model, torch.optim.RMSprop(model.parameters()), master_dtype=None
        )


def test_preexisting_optimizer_state_needs_executor_restore():
    model = nn.Linear(7, 5, bias=False)
    optimizer = torch.optim.AdamW(model.parameters())
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    with pytest.raises(ValueError, match="restore persistent optimizer tensors"):
        owned_parameters(model, optimizer, master_dtype=None)


@pytest.mark.cuda
@pytest.mark.parametrize("backend", ["torch", "mlops"])
def test_one_compiled_local_update_matches_full_adam_over_four_owned_shards(
    monkeypatch,
    backend,
):
    if not torch.cuda.is_available():
        pytest.skip("requires representative GPU")

    def forbidden(*args, **kwargs):
        raise AssertionError("local optimizer capture must not create process groups")

    monkeypatch.setattr(torch.distributed, "init_process_group", forbidden)
    monkeypatch.setattr(torch.distributed, "new_group", forbidden)
    model = nn.Linear(7, 5, bias=False, dtype=torch.bfloat16, device="meta")
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
        if backend == "torch"
        else pytest.importorskip("mlops.optim").AdamW(
            model.parameters(),
            lr=torch.tensor(0.001),
            gradient_dtype=torch.float32,
            opt_state_dtype=torch.float32,
        )
    )
    _, ownership = model_parallelism(model, Parallelism.dp(ranks_per_replica=4))
    training = capture_training_model(
        model,
        lambda model, x: model(x).float().square().mean(),
        (torch.empty(2, 7, dtype=torch.bfloat16),),
        {},
        num_examples=1,
        example_calls=None,
        fuse_narrow_gradient_accumulation=False,
        partition="whole",
        operator_recompute=None,
        gradient_dtype=torch.float32,
        device_index=0,
    )
    update = capture_update(
        model,
        optimizer,
        training,
        gradient_dtype=torch.float32,
        master_dtype=torch.float32,
        parameter_ownership=ownership,
    )
    artifact = update.captured.update
    assert not any(
        "c10d" in str(node.target) for node in artifact.graph_module.graph.nodes
    )
    assert all(
        tuple(binding.tensor.shape) == (9,)
        for binding in update.captured.bindings
        if binding.tensor.ndim
    )
    compiled = compile_artifacts((artifact,), torch.device("cuda", 0))[
        artifact.compatibility_digest
    ]
    function = restore_executable(compiled.payload, dependencies=compiled.dependencies)

    torch.manual_seed(7)
    expected = nn.Parameter(torch.randn(35, device="cuda", dtype=torch.float32))
    teacher = torch.optim.AdamW((expected,), lr=0.001, foreach=False)
    padded = torch.nn.functional.pad(expected.detach(), (0, 1))
    states = []
    for rank in range(4):
        state = {}
        for binding in update.captured.bindings:
            if binding.role.value == "parameter":
                value = padded[rank * 9 : (rank + 1) * 9].clone()
            elif binding.role.value == "compute_copy":
                value = padded[rank * 9 : (rank + 1) * 9].to(torch.bfloat16)
            elif binding.role.value == "hyperparameter":
                source = update.initialization[binding.name]
                assert source["kind"] in {"constant", "value"}
                value = torch.tensor(
                    source["value"],
                    dtype=binding.tensor.dtype,
                    device=binding.tensor.device,
                )
            else:
                value = torch.zeros(
                    binding.tensor.shape,
                    dtype=binding.tensor.dtype,
                    device=binding.tensor.device,
                )
            state[binding.name] = value
        states.append(state)

    for _ in range(3):
        expected.grad = torch.randn_like(expected)
        gradient = torch.nn.functional.pad(expected.grad, (0, 1))
        teacher.step()
        with torch.no_grad():
            for rank, state in enumerate(states):
                state["gradient.weight"].copy_(gradient[rank * 9 : (rank + 1) * 9])
                function(*(state[binding.name] for binding in update.captured.bindings))
        actual = torch.cat([state["weight"] for state in states])[:35]
        torch.testing.assert_close(actual, expected)
        actual_compute = torch.cat([state["compute.weight"] for state in states])[:35]
        torch.testing.assert_close(actual_compute, expected.to(torch.bfloat16))
        for name in ("exp_avg", "exp_avg_sq"):
            actual = torch.cat([state[f"optimizer.weight.{name}"] for state in states])[
                :35
            ]
            torch.testing.assert_close(actual, teacher.state[expected][name])
        for state in states:
            assert (
                state["optimizer.weight.step"].item()
                == teacher.state[expected]["step"].item()
            )
    measured = measure_task(
        function,
        tuple(states[0][binding.name].clone() for binding in update.captured.bindings),
        compiled.manifest.to_dict(),
        device=torch.device("cuda", 0),
        warmup_seconds=0.005,
        measure_seconds=0.01,
    )
    assert measured.duration_seconds > 0
    assert measured.metadata["memory_kind"] == "measured"
    assert "allocation_contract" in measured.metadata

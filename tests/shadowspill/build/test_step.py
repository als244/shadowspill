import copy
from itertools import product

import pytest
import torch

from shadowspill import Program
from shadowspill.build.step import build_step
from shadowspill.planning.training.bind import bind_training
from shadowspill.step import StepDataOrdering

from .test_training import Network, _check_cpu_planning, objective, replay_training


def initial_optimizer_values(program, model):
    result = {}
    for binding in program.bindings["optimizer_bindings"]:
        if binding["role"] not in {"state", "hyperparameter"}:
            continue
        source = binding["initialization"]
        dtype = getattr(torch, binding["dtype"].removeprefix("torch."))
        device = binding["device"]
        if source["kind"] in {"constant", "value"}:
            result[binding["name"]] = torch.full(
                binding["shape"], source["value"], dtype=dtype, device=device
            )
        elif source["kind"] == "parameter":
            result[binding["name"]] = (
                model.get_parameter(source["name"]).to(dtype).clone()
            )
        else:
            raise AssertionError(source)
    return result


@pytest.mark.cuda
@pytest.mark.parametrize("tied", [False, True])
def test_adam_updates_and_moments_match_pytorch_without_rebuilding(tmp_path, tied):
    if not torch.cuda.is_available():
        pytest.skip("finalizing CUDA task kernels needs a representative GPU")
    with torch.device("meta"):
        model = Network(tied=tied)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
    program = build_step(
        model,
        objective=objective,
        optimizer=optimizer,
        example_args=(torch.empty(4, 8),),
        example_kwargs={"labels": torch.empty(4, 8)},
        num_examples=2,
    )
    assert not optimizer.state
    assert all(value.is_meta for value in model.parameters())
    program.save(tmp_path / "step.ssp")
    program = Program.load(tmp_path / "step.ssp")
    examples = [
        (
            (torch.randn(4, 8, device="cuda"),),
            {"labels": torch.randn(4, 8, device="cuda")},
        )
        for _ in range(2)
    ]
    for variant, ordering, optimizer_ordering in product(
        ("save", "recompute"),
        (StepDataOrdering.depth_first(2), StepDataOrdering(1, 2)),
        ("tail", "stage_interleaved"),
    ):
        actual = Network(tied=tied).cuda()
        expected = copy.deepcopy(actual)
        expected_optimizer = torch.optim.AdamW(
            expected.parameters(), lr=0.001, foreach=False
        )
        bound = bind_training(program, ordering, optimizer_ordering=optimizer_ordering)
        state = initial_optimizer_values(bound, actual)
        for _ in range(2):
            expected_optimizer.zero_grad(set_to_none=True)
            for args, kwargs in examples:
                objective(expected, *args, **kwargs)[0].backward()
            expected_optimizer.step()
            with torch.no_grad():
                replay_training(
                    bound, actual, examples, variant, optimizer_values=state
                )
            torch.testing.assert_close(
                dict(actual.named_parameters()), dict(expected.named_parameters())
            )
            for name, tensor in state.items():
                parameter_name, entry_name = name.removeprefix("optimizer.").rsplit(
                    ".", 1
                )
                parameter = expected.get_parameter(parameter_name)
                torch.testing.assert_close(
                    tensor.cpu(), expected_optimizer.state[parameter][entry_name].cpu()
                )
        phases = [task.phase for task in bound.tasks]
        if optimizer_ordering == "tail":
            assert phases[phases.index("optimizer") :] == ["optimizer"] * phases.count(
                "optimizer"
            )
        assert bound.executables == program.executables
    if not tied:
        _check_cpu_planning(tmp_path, "step.ssp")

"""All builders resolve the same name inventory before touching CUDA."""

import pytest
import torch
from torch import nn

from shadowspill import (
    Parallelism,
    ParameterStorage,
    build_forward,
    build_step,
)
from shadowspill.build.parallelism import model_parallelism


class Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4, device="meta")
        self.head = nn.Linear(4, 8, bias=False, device="meta")
        self.head.weight = self.embedding.weight
        self.register_buffer("positions", torch.empty(4, device="meta"))


def test_model_inventory_deduplicates_ties_and_includes_buffers():
    model = Tied()
    spec = Parallelism(
        parameter_storage={"head.weight": ParameterStorage(optimizer="replicated")}
    )
    actual, parameters = model_parallelism(model, spec)
    assert actual is spec
    assert parameters[0].names == ("embedding.weight", "head.weight")
    assert parameters[1].names == ("positions",)
    assert not parameters[1].trainable
    assert model.head.weight is model.embedding.weight
    assert all(value.is_meta for value in model.parameters())


def test_unsupported_full_step_redistribution_fails_before_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("validation reached GPU initialization")

    monkeypatch.setattr(torch.cuda, "init", forbidden)
    model = Tied()
    with pytest.raises(
        NotImplementedError, match="differing layouts need redistribution"
    ):
        build_step(
            model,
            parallelism=Parallelism(
                ranks_per_replica=8,
                groups={"all_ranks": (tuple(range(8)),)},
                default_parameter_storage=ParameterStorage(
                    group="all_ranks",
                    weights="sharded",
                    gradients="replicated",
                    optimizer="sharded",
                ),
            ),
            objective=lambda model, x: x.sum(),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        )


def test_parameter_typo_fails_before_capture():
    with pytest.raises(ValueError, match="matches no registered tensor"):
        build_forward(
            Tied(),
            parallelism=Parallelism(
                parameter_storage={"hed.weight": ParameterStorage()}
            ),
        )

"""Compare shared Inkling gradients with an independent parameter set per visit."""

from collections import defaultdict
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from shadowspill_workloads.lora import configure_lora
from shadowspill_workloads.models.inkling2_mini.config import PRESETS
from shadowspill_workloads.models.inkling2_mini.implementations.mlops import Model


class IndependentVisits(torch.nn.Module):
    def __init__(self, layer, count):
        super().__init__()
        self.visits = torch.nn.ModuleList(deepcopy(layer) for _ in range(count))
        self.index = 0

    def forward(self, *args):
        layer = self.visits[self.index]
        self.index = (self.index + 1) % len(self.visits)
        return layer(*args)


@pytest.mark.parametrize("passes,block,train_queries", [(2, 2, True), (3, 1, False)])
def test_shared_gradients_match_the_ordered_sum_of_independent_visits(
    passes, block, train_queries
):
    torch.manual_seed(43713)
    config = replace(
        PRESETS["tiny"],
        n_layers=6,
        core_start=2,
        core_end=4,
        local_layer_ids=(0, 1, 3, 5),
        expert_formats=("dense",) * 6,
        block_size=block,
        passes=passes,
    )
    shared = configure_lora(Model(config), rank=4, alpha=4)
    for name, param in shared.named_parameters():
        if "quer" in name:
            param.requires_grad_(train_queries)
        if param.requires_grad:
            with torch.no_grad():
                param.normal_(0, 0.03)
    unrolled = deepcopy(shared)
    for i in range(config.core_start, config.core_end):
        unrolled.layers[i] = IndependentVisits(unrolled.layers[i], passes)
    params = {n: p for n, p in shared.named_parameters() if p.requires_grad}
    visits = defaultdict(list)
    for name, param in unrolled.named_parameters():
        if param.requires_grad:
            # layers.i.visits.j.parameter belongs to physical layers.i.parameter.
            prefix, marker, suffix = name.partition(".visits.")
            physical = prefix + "." + suffix.split(".", 1)[1] if marker else name
            visits[physical].append(param)
    assert params.keys() == visits.keys()
    expected = {}

    def record(tape):
        def hook(module, inputs, output):
            output[0].retain_grad()
            tape.append(output[0])

        return hook

    with ExitStack() as hooks:
        shared_states, unrolled_states = [], []
        for model, tape in ((shared, shared_states), (unrolled, unrolled_states)):
            for layer in model.layers:
                hooks.enter_context(layer.register_forward_hook(record(tape)))
        for microbatch in range(2):
            tokens = torch.tensor([[2 + microbatch, 19, 3, 7, 0, 11, 9, 8, 17]])
            targets = torch.tensor([[19, 3, 7, 2, 11, 9, 8, 17, 13]])
            lengths = torch.tensor([4, 0, 5], dtype=torch.int32)
            unrolled.zero_grad(set_to_none=True)
            actual = shared.loss(tokens, targets, lengths)
            reference = unrolled.loss(tokens, targets, lengths)
            assert torch.equal(actual, reference)
            actual.backward()
            reference.backward()
            for left, right in zip(shared_states, unrolled_states, strict=True):
                assert torch.equal(left.grad, right.grad)
            shared_states.clear()
            unrolled_states.clear()
            for name, physical_param in params.items():
                parts = [p.grad for p in reversed(visits[name]) if p.grad is not None]
                if parts:
                    local = parts[0].clone()
                    for part in parts[1:]:
                        local.add_(part)
                    if name in expected:
                        expected[name].add_(local)
                    else:
                        expected[name] = local
                if name in expected:
                    assert torch.equal(physical_param.grad, expected[name]), name
                else:
                    assert physical_param.grad is None, name

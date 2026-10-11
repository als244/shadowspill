"""The GLM workload supplies its policy through the generic planning contract."""

import pytest
import torch
from torch import nn

pytest.importorskip("mlops")

from shadowspill.pytorch.partition.policy import resolve_partition_assignments
from workloads.mlops.glm53_flash.partition import GLMStages
from workloads.mlops.glm53_flash.quickstart import experiment


class Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    def forward(self, x):
        for expert in self.experts:
            x = expert(x)
        return x


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.before = nn.Linear(4, 4)
        self.mlp = nn.Module()
        self.mlp.backend = Experts()

    def forward(self, x):
        return self.mlp.backend(self.before(x)).sin()


class Example(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer(), Layer()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.sum()


@pytest.mark.parametrize("wrapped", (False, True))
def test_each_expert_is_separate_under_export_and_objective_wrapper(wrapped):
    model = Example()
    if wrapped:
        model = nn.Sequential(model)
    exported = torch.export.export(model, (torch.randn(2, 4),))
    assignments, _ = resolve_partition_assignments(
        exported.graph_module, model, GLMStages()
    )
    groups = {}
    for node, stage in assignments.items():
        groups.setdefault(stage, []).append(node)
    # Each layer: one pre-expert group and three experts. The work after a
    # layer's experts joins the next layer's pre-expert group; the last
    # layer's forms its own. The scalar objective stays in a root-module group.
    assert len(groups) == 10
    expert_stages = []
    for stage, nodes in groups.items():
        experts = set()
        for node in nodes:
            for path, _ in node.meta.get("nn_module_stack", {}).values():
                if ".backend.experts." in path:
                    experts.add(
                        path.split(".backend.experts.")[0]
                        + ".expert."
                        + path.split(".backend.experts.")[1].split(".")[0]
                    )
        assert len(experts) <= 1
        if experts:
            expert_stages.append(stage)
    assert len(expert_stages) == 6

    def paths_of(node):
        return [path for path, _ in node.meta.get("nn_module_stack", {}).values()]

    def stage_of(predicate):
        found = {
            assignments[node]
            for node in exported.graph_module.graph.nodes
            if node in assignments and predicate(node)
        }
        assert len(found) == 1, found
        return found.pop()

    combination = stage_of(
        lambda node: (
            node.target is torch.ops.aten.sin.default
            and paths_of(node)[-1].endswith("layers.0")
        )
    )
    next_attention = stage_of(
        lambda node: any(path.endswith("layers.1.before") for path in paths_of(node))
    )
    assert combination == next_attention
    assert repr(GLMStages()) == "GLMStages()"


def test_recipe_preserves_step_data_across_microbatch_candidates():
    spec = experiment(
        device="cuda:0",
        tiny=True,
        sequence_length=65,
        sequences_per_step=4,
        sequences_per_microbatch=(1, 2, 4),
    )
    assert isinstance(spec["plan_options"]["partition"], GLMStages)
    assert spec["plan_options"]["grad_dtype"] is torch.float32
    expected = spec["candidates"]["4"][0]
    for batches in spec["candidates"].values():
        assert torch.equal(torch.cat([batch[0] for batch in batches]), expected[0])
        assert torch.equal(torch.cat([batch[1] for batch in batches]), expected[1])
        for tokens, _, boundaries, cumulative, chunks in batches:
            assert boundaries[-1] == tokens.numel()
            assert cumulative.tolist() == list(boundaries)
            assert chunks.shape[0] == 2 * (len(boundaries) - 1)
    assert spec["units_per_step"] == 260

    class Loss:
        def loss(self, *args, reduction, chunk_size):
            assert reduction == "sum"
            return args[0].numel() * torch.tensor(2.0)

    for batches in spec["candidates"].values():
        assert sum(spec["objective"](Loss(), *batch) for batch in batches) == 2


@pytest.mark.parametrize(
    "changes",
    (
        {"tiny": False},
        {"checkpoint": "unused", "tiny": True},
        {"sequences_per_microbatch": (3,)},
        {"sequences_per_microbatch": (1, 1)},
        {"lora_rank": 0},
    ),
)
def test_invalid_recipe_fails_before_model_or_gpu_allocation(changes):
    with pytest.raises(ValueError):
        experiment(**{"device": "cuda:0", "tiny": True, **changes})

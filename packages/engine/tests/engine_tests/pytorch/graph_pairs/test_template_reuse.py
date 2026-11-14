"""Reuse exported stage templates without splitting or replaying later examples."""

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import GraphModule

from shadowspill.pytorch.capture.fake import fake_device_inputs, fake_device_model
from shadowspill.pytorch.capture.objective import capture_training
from shadowspill.pytorch.graph_pairs import partition_training_capture


class _MultiLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(3, 8)
        self.second = nn.Linear(8, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(torch.relu(self.first(value)))


def _objective(
    model: nn.Module, value: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return torch.nn.functional.mse_loss(model(value), target)


class _ViewingModel(nn.Module):
    """A view of the microbatch's input crosses the first stage's boundary
    beside the layer's output, so a stage hands on a tensor that shares its
    storage with an input of the step."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(3, 8)
        self.second = nn.Linear(8, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        rows = value.view(-1, 3)
        hidden = self.first(rows)
        return self.second(torch.relu(hidden)) + rows.mean()


class _ThroughFirstLayerPolicy:
    """Everything up to the first layer's product is one stage, the rest
    another: the view of the input leaves the first stage with it."""

    def assign_stages(self, graph_module: GraphModule, module: nn.Module) -> dict:
        del module
        nodes = [
            node
            for node in graph_module.graph.nodes
            if node.op not in {"placeholder", "output", "get_attr"}
        ]
        cut = next(
            index
            for index, node in enumerate(nodes)
            if node.target is torch.ops.aten.linear.default
        )
        return {
            node.name: 10 if index <= cut else 20 for index, node in enumerate(nodes)
        }


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("model_type", "partition"),
    [(_MultiLinearModel, "auto"), (_ViewingModel, _ThroughFirstLayerPolicy())],
    ids=["layers", "input-view-across-boundary"],
)
def test_examples_reuse_split_and_preserve_aliases(
    monkeypatch: pytest.MonkeyPatch,
    model_type: type[nn.Module],
    partition: object,
) -> None:
    import shadowspill.pytorch.partition.api as partition_api
    import shadowspill.pytorch.partition.split as split_owner
    from shadowspill.pytorch.capture.artifacts import TensorGeometry
    from shadowspill.pytorch.capture.objective import rebind_training_objective

    splits = 0
    replays = 0
    original = split_owner.split_module
    observe = partition_api.observe_split_root

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal splits
        splits += 1
        return original(*args, **kwargs)

    def replayed(*args: object, **kwargs: object) -> object:
        nonlocal replays
        replays += 1
        return observe(*args, **kwargs)

    monkeypatch.setattr(split_owner, "split_module", counted)
    monkeypatch.setattr(partition_api, "observe_split_root", replayed)
    real_model = model_type()
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    model = fake_device_model(real_model, mode)

    def microbatch() -> list[torch.Tensor]:
        return fake_device_inputs([torch.randn(4, 3), torch.randn(4, 2)], mode)

    with mode:
        first = capture_training(model, _objective, microbatch())
        captures = (
            first,
            *(rebind_training_objective(first, microbatch()) for _ in range(2)),
        )
        splits = 0
        separately = tuple(
            partition_training_capture(item, partition=partition, accumulating=True)
            for item in captures
        )
        assert splits >= len(captures)
        splits = 0
        template = partition_training_capture(
            captures[0], partition=partition, accumulating=True
        )
        assert splits > 0
        splits = 0
        shared = (
            template,
            *(
                partition_training_capture(
                    item,
                    partition=partition,
                    accumulating=True,
                    template=template.partitioned,
                )
                for item in captures[1:]
            ),
        )
        assert splits == 0
        assert replays == 0

    def sharing(capture: object) -> list[int]:
        """Each stage tensor's storage, numbered by first appearance: the
        pattern of which tensors share storage, whatever the storage is."""

        numbered: dict[int, int] = {}
        pattern = []
        for stage in capture.stages:  # type: ignore[attr-defined]
            for value in (*stage.example.inputs, stage.example.output):
                for leaf in value if isinstance(value, tuple | list) else (value,):
                    if isinstance(leaf, torch.Tensor):
                        key = leaf.untyped_storage()._cdata
                        pattern.append(numbered.setdefault(key, len(numbered)))
        return pattern

    def geometry(value: object) -> object:
        return (
            TensorGeometry.from_tensor(value)
            if isinstance(value, torch.Tensor)
            else value
        )

    assert len(shared[0].stages) > 1
    for alone, reused in zip(separately, shared, strict=True):
        assert sharing(alone) == sharing(reused)
        assert len(alone.stages) == len(reused.stages)
        for one, other, origin in zip(
            alone.stages, reused.stages, template.stages, strict=True
        ):
            assert one.example.stage.graph_module.code == (
                other.example.stage.graph_module.code
            )
            assert one.example.stage.input_sources == other.example.stage.input_sources
            assert one.example.stage.input_provenance == (
                other.example.stage.input_provenance
            )
            assert one.example.stage.mutations == other.example.stage.mutations
            assert [geometry(value) for value in one.example.inputs] == [
                geometry(value) for value in other.example.inputs
            ]
            assert one.graph_pairs.structural_contract == (
                other.graph_pairs.structural_contract
            )
            # Both backward forms, creating and adding into a gradient,
            # whichever microbatch this turns out to be.
            for item in (one, other):
                assert item.graph_pairs.options(accumulates=False)
                assert item.graph_pairs.options(accumulates=True)
            if reused is not template:
                for value, source, original_value in zip(
                    other.example.inputs,
                    other.example.stage.input_sources,
                    origin.example.inputs,
                    strict=True,
                ):
                    if source is not None and source.producer_stage_index is not None:
                        assert value is not original_value

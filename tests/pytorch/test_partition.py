from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import GraphModule

from shadowspill.pytorch.capture.aot import capture_forward
from shadowspill.pytorch.capture.artifacts import capture_forward_stage_artifacts
from shadowspill.pytorch.capture.fake import fake_cuda_inputs, fake_cuda_model
from shadowspill.pytorch.contracts import CaptureError
from shadowspill.pytorch.partition import (
    partition_export,
)


class _NestedRepeatedNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(4)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = torch.relu(block(value))
        return value


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = self.experts[0](value) + self.experts[1](value)
        assert isinstance(result, torch.Tensor)
        return result


class _RepeatedNetworkWithBoundaries(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(8, 8), nn.SiLU())
        self.blocks = nn.ModuleList([_Block() for _ in range(4)])
        self.final_norm = nn.LayerNorm(8)
        self.head = nn.Linear(8, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.embedding(value)
        for block in self.blocks:
            value = torch.relu(block(value))
        return self.head(self.final_norm(value))


class _TwoStagePolicy:
    def assign_stages(
        self,
        graph_module: GraphModule,
        module: nn.Module,
    ) -> dict[str, int]:
        del module
        nodes = tuple(
            node
            for node in graph_module.graph.nodes
            if node.op not in {"placeholder", "output", "get_attr"}
        )
        midpoint = max(1, len(nodes) // 2)
        return {
            node.name: 10 if index < midpoint else 20
            for index, node in enumerate(nodes)
        }


def test_custom_partition_policy_is_normalized_and_shared_by_forward() -> None:
    model = _NestedRepeatedNetwork()
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake_cuda_model(model, mode)
    inputs = fake_cuda_inputs([torch.randn(2, 8)], mode)
    with mode, torch.no_grad():
        exported = capture_forward(replica, inputs)
        partitioned = partition_export(
            exported,
            replica,
            partition=_TwoStagePolicy(),
        )
        artifacts = capture_forward_stage_artifacts(partitioned)
    assert partitioned.repeated_groups == ()
    assert len(partitioned.stages) == 2
    assert len(artifacts) == 2


def test_custom_partition_policy_rejects_incomplete_or_noncontiguous_labels() -> None:
    class Incomplete:
        def assign_stages(
            self,
            graph_module: GraphModule,
            module: nn.Module,
        ) -> dict[str, int]:
            del graph_module, module
            return {}

    class Noncontiguous:
        def assign_stages(
            self,
            graph_module: GraphModule,
            module: nn.Module,
        ) -> dict[str, int]:
            del module
            nodes = tuple(
                node
                for node in graph_module.graph.nodes
                if node.op not in {"placeholder", "output", "get_attr"}
            )
            return {
                node.name: (0 if index % 2 == 0 else 1)
                for index, node in enumerate(nodes)
            }

    model = _NestedRepeatedNetwork()
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake_cuda_model(model, mode)
    inputs = fake_cuda_inputs([torch.randn(2, 8)], mode)
    with mode, torch.no_grad():
        exported = capture_forward(replica, inputs)
        with pytest.raises(CaptureError, match="coverage"):
            partition_export(exported, replica, partition=Incomplete())
        with pytest.raises(CaptureError, match="contiguous"):
            partition_export(exported, replica, partition=Noncontiguous())


def test_auto_partition_uses_outer_repeated_blocks_not_nested_experts() -> None:
    model = _NestedRepeatedNetwork()
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake_cuda_model(model, mode)
    inputs = fake_cuda_inputs([torch.randn(2, 8)], mode)
    with mode, torch.no_grad():
        exported = capture_forward(replica, inputs)
        partitioned = partition_export(exported, replica)
        artifacts = capture_forward_stage_artifacts(partitioned)
    assert partitioned.repeated_groups == ("blocks",)
    assert len(partitioned.stages) == 4
    assert len(artifacts) == 4
    assert all(artifact.operator_targets for artifact in artifacts)
    assert len({artifact.compatibility_digest for artifact in artifacts}) == 1


def test_auto_partition_keeps_first_consumer_and_isolates_epilogue() -> None:
    model = _RepeatedNetworkWithBoundaries()
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake_cuda_model(model, mode)
    inputs = fake_cuda_inputs([torch.randn(2, 8)], mode)
    with mode, torch.no_grad():
        exported = capture_forward(replica, inputs)
        partitioned = partition_export(exported, replica)
        artifacts = capture_forward_stage_artifacts(partitioned)

    assert partitioned.repeated_groups == ("blocks",)
    assert len(partitioned.stages) == 5
    assert len(artifacts) == 5
    block_artifacts = artifacts[1:4]
    assert len({artifact.compatibility_digest for artifact in block_artifacts}) == 1
    assert any("linear" in target for target in artifacts[0].operator_targets)
    assert any("relu" in target for target in artifacts[0].operator_targets)
    assert any("layer_norm" in target for target in artifacts[-1].operator_targets)


def test_whole_partition_is_one_stage() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake_cuda_model(model, mode)
    inputs = fake_cuda_inputs([torch.randn(2, 4)], mode)
    with mode, torch.no_grad():
        partitioned = partition_export(
            capture_forward(replica, inputs), replica, partition="whole"
        )
    assert partitioned.repeated_groups == ()
    assert len(partitioned.stages) == 1


def test_export_buffer_mutation_is_projected_onto_producer_stage() -> None:
    class Stateful(nn.Module):
        running: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("running", torch.zeros(8))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.running.add_(value.sum(0))
            return self.running[2:]

    model = Stateful()
    inputs = (torch.randn(2, 8),)
    capture = capture_forward(model, inputs)
    partitioned = partition_export(capture, model, partition="whole")
    artifacts = capture_forward_stage_artifacts(partitioned)

    assert len(capture.mutations) == 1
    assert len(partitioned.stages[0].stage.mutations) == 1
    mutation = artifacts[0].storage_contract.mutations[0]
    assert mutation.replacement_output_leaf == 0
    assert mutation.producer_target == "aten.add.Tensor"
    assert len(artifacts[0].storage_contract.roots) == 1
    assert artifacts[0].storage_contract.roots[0].kind.value == "fresh"
    assert tuple(
        (view.root_id, view.offset_bytes)
        for view in artifacts[0].storage_contract.output_views
    ) == ((0, 0), (0, 8))

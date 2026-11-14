from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.pytorch.capture.fake import fake_device_model
from shadowspill.pytorch.capture.objective import capture_training_objective
from shadowspill.pytorch.graph_pairs import partition_training_capture
from shadowspill.pytorch.graph_pairs.artifacts import parameter_gradient_leaves
from shadowspill.pytorch.graph_pairs.rebind import rebind_task_graph_pairs
from shadowspill.pytorch.graph_pairs.serialization import CachedAotGraphPair
from shadowspill.pytorch.representations import map_tensor, tensor_components

from .representation_fixtures import ScaledWeight, model


@pytest.fixture(autouse=True)
def clear_capture_caches():
    # These tests deliberately create several subclass variants of the same
    # Export wrapper. Do not consume later tests' Dynamo specialization budget.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def test_fake_model_preserves_component_aliases():
    net = model()
    net.tied = net.weight
    net.register_buffer("window", net.weight.payload[1:])
    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        fake = fake_device_model(net, mode)
        assert isinstance(fake.weight, ScaledWeight)
        assert fake.weight is fake.tied
        assert (
            fake.weight.payload.untyped_storage()._cdata
            == fake.window.untyped_storage()._cdata
        )
        assert fake.window.storage_offset() == 8
        assert fake.weight.device.type == "cuda"
        assert not fake.weight.payload.requires_grad
        assert fake.weight.requires_grad


def test_logical_gradients_survive_physical_inputs_and_cache_restore():
    net = model()
    x = torch.randn(3, 8)
    capture = capture_training_objective(net, lambda m, x: m(x).square().sum(), (x,))
    partitioned = partition_training_capture(capture, partition="whole")
    stage = partitioned.stages[0]
    expected = torch.autograd.grad(net(x).square().sum(), net.weight)[0]
    for variant in stage.graph_pairs.variants:
        pair = variant.pair
        assert pair.forward.argument_count == 3
        leaves = parameter_gradient_leaves(pair)
        assert len(leaves) == 1
        args = tuple(
            component
            for value in stage.example.inputs
            for _, component in tensor_components(value)
        )
        outputs = pair.forward.graph_module(*args)
        saved = outputs[-pair.saved_value_count :]
        actual = pair.backward.graph_module(*saved)[leaves[0]]
        torch.testing.assert_close(actual, expected)
        with FakeTensorMode(allow_non_fake_inputs=True):
            restored = CachedAotGraphPair.capture(pair).restore()
            assert restored.forward.input_components == pair.forward.input_components
            assert restored.gradient_provenance == pair.gradient_provenance
    rebound = rebind_task_graph_pairs(stage.graph_pairs, stage.example)
    assert (
        rebound.reference.forward.input_components
        == stage.graph_pairs.reference.forward.input_components
    )


def test_fused_gradient_slices_have_independent_optimizer_storage():
    class PackedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Parameter(torch.randn(4, 8))
            self.second = torch.nn.Parameter(torch.randn(4, 8))

        def forward(self, x):
            return torch.nn.functional.linear(x, torch.cat((self.first, self.second)))

    net = PackedLinear()
    x = torch.randn(3, 8)
    capture = capture_training_objective(net, lambda m, x: m(x).square().sum(), (x,))
    stage = partition_training_capture(capture, partition="whole").stages[0]
    expected = torch.autograd.grad(net(x).square().sum(), tuple(net.parameters()))
    for variant in stage.graph_pairs.with_gradient_dtype(torch.float32).variants:
        pair = variant.pair
        outputs = pair.forward.graph_module(*stage.example.inputs)
        actual = pair.backward.graph_module(*outputs[-pair.saved_value_count :])
        gradients = tuple(actual[index] for index in parameter_gradient_leaves(pair))
        torch.testing.assert_close(gradients, expected)
        assert (
            gradients[0].untyped_storage()._cdata
            != gradients[1].untyped_storage()._cdata
        )


def test_rebuild_reads_only_physical_leaves():
    source = model().weight
    copied = map_tensor(source, lambda value: value.clone())
    assert isinstance(copied, ScaledWeight)
    assert isinstance(copied, torch.nn.Parameter)
    assert copied is not source
    torch.testing.assert_close(copied.dense(), source.dense())
    assert copied.payload.data_ptr() != source.payload.data_ptr()


def test_detached_components_keep_storage_when_live_handles_are_rebound():
    from shadowspill.pytorch.representations import detached_representation

    source = model().weight
    reference = detached_representation(source)
    expected = reference.dense().clone()
    assert reference.payload.data_ptr() == source.payload.data_ptr()
    assert reference.payload is not source.payload
    source.payload.set_(torch.zeros_like(source.payload))
    source.scale.set_(torch.ones_like(source.scale))
    torch.testing.assert_close(reference.dense(), expected)


def test_optimizer_placeholder_preserves_component_storage_views():
    from shadowspill.pytorch.representations import empty_representation

    storage = torch.empty(65, dtype=torch.uint8)
    source = ScaledWeight(storage[8:40].view(4, 8), storage[48:52].view(torch.float32))
    target = empty_representation(source, torch.empty(0))
    assert (
        target.payload.untyped_storage()._cdata == target.scale.untyped_storage()._cdata
    )
    assert target.payload.storage_offset() == 8
    assert target.scale.storage_offset() == 12
    assert target.payload.untyped_storage().nbytes() == 65
    assert target.payload.data_ptr() != source.payload.data_ptr()


def test_meta_initialization_preserves_representation_and_ties():
    from shadowspill.training._model import initialize_model

    source = model()
    source.tied = source.weight
    source.register_buffer("scale_alias", source.weight.scale)
    with torch.device("meta"):
        target = model()
        target.tied = target.weight
        target.register_buffer("scale_alias", target.weight.scale)
    initialize_model(target, state=source.state_dict())
    assert isinstance(target.weight, ScaledWeight)
    assert target.weight is target.tied
    assert target.weight.scale is target.scale_alias
    torch.testing.assert_close(target.weight.dense(), source.weight.dense())


def test_inference_exposes_physical_parameters():
    from shadowspill.pytorch.capture.export import capture_forward, inference_artifact

    net = model()
    x = torch.randn(3, 8)
    capture = capture_forward(net, (x,))
    artifact = inference_artifact(capture)
    assert artifact.input_components == ((0, ("payload",)), (0, ("scale",)), (1, ()))
    result = artifact.graph_module(net.weight.payload, net.weight.scale, x)
    torch.testing.assert_close(result[0], net(x))


def test_mutable_buffer_is_functional_with_logical_weight_gradients():
    from shadowspill.pytorch.capture.objective import capture_training

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = model()
            self.register_buffer("total", torch.tensor(0.0))

        def forward(self, x):
            self.total.add_(x.sum())
            return self.layer(x) + self.total

    net = Network()
    x = torch.randn(3, 8)
    capture = capture_training(net, lambda m, x: m(x).square().sum(), (x,))
    baseline = net.total.clone()
    expected = (net.layer(x) + baseline + x.sum()).square().sum()
    expected_gradient = torch.autograd.grad(expected, net.layer.weight)[0]
    arguments = tuple(
        component
        for value in capture.exported.flat_inputs
        for _, component in tensor_components(value)
    )
    for pair in (capture.save_pair, capture.recompute_pair):
        # Two weight components precede the ordinary mutable buffer.
        assert pair.forward.storage_contract.mutations[0].input_position == 2
        outputs = pair.forward.graph_module(*arguments)
        torch.testing.assert_close(outputs[0], baseline + x.sum())
        torch.testing.assert_close(outputs[1], expected)
        saved = outputs[-pair.saved_value_count :]
        gradients = pair.backward.graph_module(*saved)
        torch.testing.assert_close(gradients[0], expected_gradient)
        torch.testing.assert_close(net.total, baseline)


def test_optimizer_publishes_all_components_from_dense_master():
    from shadowspill.pytorch.capture.artifacts import GraphArtifact
    from shadowspill.pytorch.optimizer import capture_optimizer

    net = model()
    master = torch.nn.Parameter(net.weight.dense().detach())
    optimizer = torch.optim.SGD([master], lr=0.01, foreach=False)
    capture = capture_optimizer(
        {"weight": master}, optimizer, compute_copies={"weight": net.weight}
    )
    assert isinstance(capture.update, GraphArtifact)
    assert [b.name for b in capture.bindings] == [
        "weight",
        "gradient.weight",
        "compute.weight.payload",
        "compute.weight.scale",
    ]
    values = [
        master.detach().clone(),
        torch.ones_like(master),
        net.weight.payload.clone(),
        net.weight.scale.clone(),
    ]
    expected = master.detach() - 0.01
    with torch.no_grad():
        capture.update.graph_module(*values)
    torch.testing.assert_close(values[0], expected)
    quantized = ScaledWeight(values[2], values[3])
    target = net.weight.detach().clone()
    target.copy_(expected)
    torch.testing.assert_close(quantized.payload, target.payload)
    torch.testing.assert_close(quantized.scale, target.scale)


@pytest.mark.cuda
def test_lowering_counts_components_and_one_logical_gradient():
    from shadowspill import build_step
    from shadowspill.ir import ObjectRole

    net = model()
    program = build_step(
        net,
        objective=lambda m, x: m(x).square().sum(),
        optimizer=torch.optim.SGD(net.parameters(), lr=0.01),
        master_dtype=torch.float32,
        example_args=(torch.ones(3, 8),),
        num_examples=2,
        partition="whole",
    )
    parameters = [o for o in program.objects if o.role == ObjectRole.PARAMETER]
    gradients = [o for o in program.objects if o.role == ObjectRole.GRADIENT]
    assert sorted(o.size_bytes for o in parameters) == [4, 32, 128]
    assert [o.size_bytes for o in gradients] == [128]


@pytest.mark.parametrize("wrapped", [False, True])
def test_repeated_inference_contracts_rebind_occurrence_values(monkeypatch, wrapped):
    from shadowspill.pytorch.capture import artifacts as capture
    from shadowspill.pytorch.capture.export import capture_forward
    from shadowspill.pytorch.partition import partition_export

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                torch.nn.Linear(8, 8, bias=False) for _ in range(4)
            )
            if wrapped:
                for index, block in enumerate(self.blocks):
                    block.weight = torch.nn.Parameter(
                        ScaledWeight(
                            torch.arange(-32, 32, dtype=torch.int8).reshape(8, 8),
                            torch.tensor(0.01 * (index + 1)),
                        )
                    )

        def forward(self, value):
            for block in self.blocks:
                value = block(value)
            return value

    net = Network()
    value = torch.randn(3, 8)
    with torch.no_grad():
        exported = capture_forward(net, (value,))
        split = partition_export(exported, net)
        baseline = tuple(
            capture.capture_inference_artifact(
                graph_module=example.stage.graph_module,
                example_inputs=example.inputs,
                explicit_mutations=example.stage.mutations,
                input_provenance=example.stage.input_provenance,
            )
            for example in split.stages
        )
        calls = []
        original = capture.capture_inference_artifact

        def counted(**kwargs):
            calls.append(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(capture, "capture_inference_artifact", counted)
        reused = capture.capture_forward_stage_artifacts(split)
        assert len(reused) == 4
        assert len(calls) < len(reused)
        assert reused[1].graph_module is reused[2].graph_module
        for actual, expected in zip(reused, baseline, strict=True):
            assert actual.compatibility_digest == expected.compatibility_digest
            assert actual.input_components == expected.input_components
            assert actual.input_provenance == expected.input_provenance
            torch.testing.assert_close(
                actual.graph_module(*actual.example_arguments),
                expected.graph_module(*expected.example_arguments),
            )
        # Each block must retain its own weights and scale, not the first
        # occurrence's examples or provenance.
        assert len({id(item.example_arguments[0]) for item in reused}) == 4
        sources = tuple(item.input_provenance[0].source for item in reused)
        assert len(set(sources)) == 4

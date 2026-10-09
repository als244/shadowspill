from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.pytorch.capture.aot import capture_training_objective
from shadowspill.pytorch.capture.fake import fake_device_model
from shadowspill.pytorch.graph_pairs import partition_training_capture
from shadowspill.pytorch.graph_pairs.artifacts import parameter_gradient_leaves
from shadowspill.pytorch.graph_pairs.rebind import rebind_task_graph_pairs
from shadowspill.pytorch.graph_pairs.serialization import CachedAotGraphPair
from shadowspill.pytorch.representations import map_tensor, tensor_components
from shadowspill.pytorch.state.storage import NamedTensor, _storage_roots

from .representations import ScaledWeight, model


@pytest.fixture(autouse=True)
def clear_capture_caches():
    # These tests deliberately create several subclass variants of the same
    # Export wrapper. Do not consume later tests' Dynamo specialization budget.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def test_storage_inventory_uses_components_and_preserves_aliases():
    net = model()
    net.tied = net.weight
    net.register_buffer("window", net.weight.payload[1:])
    roots = _storage_roots(
        [
            NamedTensor(n, t)
            for n, t in (
                *net.named_parameters(remove_duplicate=False),
                *net.named_buffers(),
            )
        ]
    )
    assert sum(owner.numel() for owner, _ in roots) == 36
    assert len(roots) == 2
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


def test_export_archive_preserves_fake_component_views_and_reloads(tmp_path):
    from torch._export.serde.serialize import _reconstruct_fake_tensor

    from shadowspill.pytorch.capture.aot import capture_forward
    from shadowspill.pytorch.store import _serializable_export

    net = model()
    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        fake = fake_device_model(net, mode)
        capture = capture_forward(fake, (torch.empty(3, 8, device="cuda"),))
    exported = _serializable_export(capture.exported_program)
    path = tmp_path / "export.pt2"
    torch.export.save(exported, path)
    with torch.serialization.safe_globals([ScaledWeight, _reconstruct_fake_tensor]):
        restored = torch.export.load(path)
    weight = next(iter(restored.state_dict.values()))
    assert isinstance(weight, ScaledWeight)
    assert weight.payload.is_meta and weight.scale.is_meta
    assert type(weight.payload) is torch.Tensor


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


def test_step_cache_identity_distinguishes_logical_and_physical_weights():
    from shadowspill.pytorch.planning.identity import step_identity

    def objective(net, data):
        return net(data).sum()

    options = {
        "objective": objective,
        "build_optimizer": lambda parameters: torch.optim.SGD(parameters, lr=0.01),
        "hyperparams": (),
        "example_inputs": ((torch.ones(3, 8),),),
        "partition": "whole",
        "profiling_metadata": None,
        "optimizer_ordering": "interleaved",
        "allocation_probe_seeds": 1,
        "allocation_probe_repetitions": 1,
        "export_bypass_key": "same-model",
        "machine": {},
        "environment": {},
    }
    quantized = model()
    ordinary = torch.nn.Linear(8, 4, bias=False)
    first, second = (step_identity(net, **options) for net in (quantized, ordinary))
    assert first["model"]["parameters"] == second["model"]["parameters"]
    assert first != second
    before = first
    quantized.weight.payload.add_(1)
    assert step_identity(quantized, **options) == before  # Values are not structure.


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
    from shadowspill.pytorch.capture.aot import capture_forward, inference_artifact

    net = model()
    x = torch.randn(3, 8)
    capture = capture_forward(net, (x,))
    artifact = inference_artifact(capture)
    assert artifact.input_components == ((0, ("payload",)), (0, ("scale",)), (1, ()))
    result = artifact.graph_module(net.weight.payload, net.weight.scale, x)
    torch.testing.assert_close(result[0], net(x))


def test_mutable_buffer_is_functional_with_logical_weight_gradients():
    from shadowspill.pytorch.capture.aot import capture_training

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


def test_nested_representation_copy_uses_physical_storage_roots():
    from shadowspill.libraries import resolve_library
    from shadowspill.pytorch.state.model_copy import copy_model_with_runtime_storages
    from shadowspill.pytorch.state.records import PersistentStorage

    net = model()
    net.weight = torch.nn.Parameter(
        ScaledWeight(net.weight.detach(), torch.tensor(2.0))
    )
    net.tied = net.weight
    net.register_buffer("scale_alias", net.weight.payload.scale)
    net.register_buffer("empty", torch.empty(0, 3))
    roots = _storage_roots(
        [
            NamedTensor(name, value)
            for name, value in (
                *net.named_parameters(remove_duplicate=False),
                *net.named_buffers(),
            )
        ]
    )
    assert sum(owner.numel() for owner, _ in roots) == 40
    storages = tuple(
        PersistentStorage(i, i, 0, owner.numel(), 0, owner, views, True)
        for i, (owner, views) in enumerate(roots)
    )
    torch.ops.load_library(str(resolve_library("libshadowspill_pytorch.so")))
    copied, _ = copy_model_with_runtime_storages(net, storages, addressable=False)
    assert copied.weight is copied.tied
    assert copied.weight.payload.scale is copied.scale_alias
    assert copied.empty is not net.empty
    assert copied.empty.shape == net.empty.shape
    assert isinstance(copied.weight.payload, ScaledWeight)
    assert copied.weight.payload.payload.shape == net.weight.payload.payload.shape
    assert all(item.unbacked for item in storages)
    with pytest.raises(RuntimeError, match="non-addressable ShadowSpill pool"):
        copied.weight.payload.payload.data_ptr()
    assert torch.isfinite(net.weight.dense()).all()


def test_optimizer_publishes_all_components_from_dense_master():
    from shadowspill.pytorch.capture.artifacts import GraphArtifact
    from shadowspill.pytorch.optimizer import capture_optimizer

    net = model()
    master = torch.nn.Parameter(net.weight.dense().detach())
    optimizer = torch.optim.SGD([master], lr=0.01, foreach=False)
    capture = capture_optimizer(
        {"weight": master}, optimizer, compute_copies={"weight": net.weight}
    )
    assert isinstance(capture.update, GraphArtifact), capture.opaque_reason
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
    from shadowspill.ir import ObjectRole
    from shadowspill.pytorch.capture.fake import fake_device_inputs
    from shadowspill.pytorch.lowering.training import lower_partitioned_training_program
    from shadowspill.pytorch.optimizer import capture_optimizer
    from tests.shadowspill.pytorch.lowering.test_training_lowering import _measurement

    net = model()
    master = torch.nn.Parameter(net.weight.dense().detach())
    optimizer = capture_optimizer(
        {"weight": master},
        torch.optim.SGD([master], lr=0.01),
        compute_copies={"weight": net.weight},
    )
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    fake = fake_device_model(net, mode)
    with mode:
        captures = tuple(
            partition_training_capture(
                capture_training_objective(
                    fake,
                    lambda m, x: m(x).square().sum(),
                    fake_device_inputs((torch.ones(3, 8),), mode),
                ),
                partition="whole",
                accumulating=True,
            )
            for _ in range(2)
        )
        artifacts = [
            a
            for c in captures
            for s in c.stages
            for v in s.graph_pairs.variants
            for a in (v.pair.forward, v.pair.backward)
        ]
        artifacts += [
            optimizer.update,
            *(task.artifact for task in optimizer.update_tasks),
        ]
        measurements = {a.compatibility_digest: _measurement(a) for a in artifacts}
        lowered = lower_partitioned_training_program(
            fake, captures, measurements, optimizer
        )
    parameters = [o for o in lowered.program.objects if o.role == ObjectRole.PARAMETER]
    gradients = [o for o in lowered.program.objects if o.role == ObjectRole.GRADIENT]
    assert sorted(o.size_bytes for o in parameters) == [4, 32, 128]
    assert [o.size_bytes for o in gradients] == [128]

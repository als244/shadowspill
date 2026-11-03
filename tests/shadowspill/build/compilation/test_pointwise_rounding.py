"""Native scalar rounding survives compilation, caching, and cold code reload."""

import io

import pytest
import torch
from torch._inductor import config, metrics
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.build.compilation import compile_task
from shadowspill.build.compilation.pointwise import native_pointwise_rounding
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.serialized_graph import SerializedFxGraph
from shadowspill.pytorch.capture.storage import capture_task_storage_contract
from shadowspill.pytorch.executable import restore_executable
from shadowspill.task.manifest import validate_value_contract


def _derivative(gradient, output):
    return (torch.ops.aten.sigmoid_backward.default(gradient, output),)


def _scalar_rounding(gradient, output):
    dtype = output.dtype
    complement = (1.0 - output.float()).to(dtype)
    product = (gradient.float() * complement.float()).to(dtype)
    return (product.float() * output.float()).to(dtype)


def _inputs(shape, dtype, device, transposed=False):
    inputs = tuple(torch.randn(shape, dtype=dtype, device=device) for _ in range(2))
    if transposed:
        inputs = tuple(value.t() for value in inputs)
    return inputs[0], inputs[1].sigmoid()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("transposed", [False, True])
def test_fake_cuda_expansion_preserves_values_contract_and_serialization(
    dtype, transposed
):
    with FakeTensorMode(), config.patch(emulate_precision_casts=True):
        inputs = _inputs((7, 11), dtype, "cuda", transposed)
        graph = make_fx(_derivative)(*inputs)
        before = graph.code
        contract = capture_task_storage_contract(graph, inputs)
        changed, numerical = native_pointwise_rounding(graph, emulate=True)
        assert numerical == ("cuda.sigmoid_backward.scalar_rounding.v1",)
        assert graph.code == before and changed is not graph
        assert changed.code != before
        lowered_contract = capture_task_storage_contract(changed, inputs)
        validate_value_contract(contract, lowered_contract)
        assert lowered_contract.output_views == contract.output_views
        assert lowered_contract.mutations == contract.mutations
        assert [
            (root.kind, root.source_input, root.minimum_span_bytes)
            for root in lowered_contract.roots
        ] == [
            (root.kind, root.source_input, root.minimum_span_bytes)
            for root in contract.roots
        ]
        serialized = SerializedFxGraph.capture(changed)
    buffer = io.BytesIO()
    torch.save(serialized, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False).restore()
    actual_inputs = _inputs((7, 11), dtype, "cpu", transposed)
    expected = _scalar_rounding(*actual_inputs)
    assert torch.equal(changed(*actual_inputs)[0], expected)
    assert torch.equal(restored(*actual_inputs)[0], expected)


@pytest.mark.parametrize(
    "device,dtype,emulate",
    [
        ("cpu", torch.bfloat16, True),
        ("cpu", torch.float16, True),
        ("cuda", torch.float32, True),
        ("cuda", torch.bfloat16, False),
    ],
)
def test_native_expansion_leaves_other_arithmetic_unchanged(device, dtype, emulate):
    with FakeTensorMode():
        inputs = _inputs((7, 11), dtype, device)
        graph = make_fx(_derivative)(*inputs)
        changed, numerical = native_pointwise_rounding(graph, emulate=emulate)
    assert changed is graph and numerical == ()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape,transposed", [((1, 4096), False), ((64, 257), True), ((257, 19), False)]
)
def test_native_cuda_rounding_uses_one_packaged_kernel(
    tmp_path, monkeypatch, dtype, shape, transposed
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    torch.manual_seed(728)
    inputs = _inputs(shape, dtype, "cuda", transposed)
    graph = make_fx(_derivative)(*inputs)
    artifact = GraphArtifact.capture(
        kind="backward", graph_module=graph, example_inputs=inputs
    )
    original = artifact.graph_module.code
    with config.patch(emulate_precision_casts=False, compile_threads=1):
        unrounded = compile_task(artifact)
    with config.patch(emulate_precision_casts=True, compile_threads=1):
        before = metrics.generated_kernel_count
        rounded = compile_task(artifact)
        assert metrics.generated_kernel_count - before == 1
        cached = compile_task(artifact)
    assert rounded.numerical_contracts and not unrounded.numerical_contracts
    assert (
        rounded.manifest.compatibility_digest != unrounded.manifest.compatibility_digest
    )
    assert cached.manifest == rounded.manifest
    assert artifact.graph_module.code == original
    expected = _derivative(*inputs)[0]
    for built in (rounded, cached):
        payload = tmp_path / "derivative.pt"
        torch.save(built.payload, payload)
        restored = restore_executable(torch.load(payload, weights_only=False))
        assert torch.equal(restored(*inputs)[0], expected)
    control = restore_executable(unrounded.payload)(*inputs)[0]
    assert not torch.equal(control, expected)

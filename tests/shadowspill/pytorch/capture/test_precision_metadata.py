"""Cache reuse preserves precision semantics as well as graph operations."""

import io
import json

import pytest
import torch
from torch._functorch._aot_autograd.descriptors import (
    DummyAOTOutput,
    SavedForBackwardsAOTOutput,
    SavedForBackwardsNoVcCheckAOTOutput,
)
from torch._guards import TracingContext, tracing
from torch._inductor import config
from torch._inductor.compile_fx import compile_fx_inner
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.objective import capture_training
from shadowspill.pytorch.capture.pointwise import (
    pointwise_precision_barrier,
    saved_outputs,
)
from shadowspill.pytorch.capture.serialized_graph import SerializedFxGraph
from shadowspill.pytorch.capture.torch_deprecations import copy_graph_module
from shadowspill.pytorch.graph_pairs import GraphPairStore, partition_training_capture
from tests.shadowspill.pytorch.graph_pairs.test_graph_pairs import _RepeatedNetwork


def _pointwise(x, y, z):
    return ((x * y + z).sigmoid(),)


def _barriers(graph):
    return tuple(
        node.name for node in graph.graph.nodes if pointwise_precision_barrier(node)
    )


def _compiled_graph(graph, inputs):
    mode = FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True)
    fake_inputs = [mode.from_tensor(value, static_shapes=True) for value in inputs]
    FakeTensorProp(graph, mode=mode).propagate(*fake_inputs)
    with tracing(TracingContext(mode)):
        compiled = compile_fx_inner(graph, fake_inputs)
    with torch.no_grad():
        return compiled(list(inputs))


def test_explicit_precision_changes_graph_compatibility():
    inputs = tuple(
        torch.randn(19, 7, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)
    )
    pairs = []
    for enabled in (False, True):
        with config.patch(emulate_precision_casts=enabled):
            graph = make_fx(_pointwise)(*inputs)
            pairs.append(
                capture_graph_pair(
                    graph,
                    inputs,
                    original_output=_pointwise(*inputs),
                    memory_budget=0.0,
                )
            )
    for direction in ("forward", "backward"):
        plain, rounded = (getattr(pair, direction) for pair in pairs)
        assert plain.graph_module.code == rounded.graph_module.code
        assert not _barriers(plain.graph_module)
        assert _barriers(rounded.graph_module)
        assert plain.compatibility_digest != rounded.compatibility_digest


def test_graph_pair_cache_keys_capture_precision_and_restores_it(tmp_path):
    model = _RepeatedNetwork().bfloat16()
    inputs = (torch.randn(7, 8).bfloat16(), torch.randn(7, 8).bfloat16())
    # One already-exported source graph must be recaptured when the requested
    # tracing precision changes, even when the source carries no markers.
    with config.patch(emulate_precision_casts=False):
        captured = capture_training(
            model, lambda current, x, y: (current(x) - y).square().mean(), inputs
        )
        store = GraphPairStore(tmp_path)
        plain = partition_training_capture(captured, graph_pair_store=store)
    with config.patch(emulate_precision_casts=True):
        rounded = partition_training_capture(captured, graph_pair_store=store)
        assert store.misses == 6 and store.hits == 2
        restored_store = GraphPairStore(tmp_path)
        restored = partition_training_capture(captured, graph_pair_store=restored_store)
        assert restored_store.misses == 0 and restored_store.hits == 4
    for old, before, after in zip(
        plain.stages, rounded.stages, restored.stages, strict=True
    ):
        for old_variant, before_variant, after_variant in zip(
            old.graph_pairs.variants,
            before.graph_pairs.variants,
            after.graph_pairs.variants,
            strict=True,
        ):
            for direction in ("forward", "backward"):
                left = getattr(before_variant.pair, direction)
                right = getattr(after_variant.pair, direction)
                assert _barriers(left.graph_module)
                assert _barriers(left.graph_module) == _barriers(right.graph_module)
                assert left.compatibility_digest == right.compatibility_digest
                assert (
                    left.compatibility_digest
                    != getattr(old_variant.pair, direction).compatibility_digest
                )
    assert {
        json.loads(path.read_text())["emulate_precision_casts"]
        for path in tmp_path.rglob("manifest.json")
    } == {False, True}


def test_disk_restored_rounding_markers_preserve_compiled_bf16_values(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    torch.manual_seed(73)
    inputs = tuple(torch.randn(257, 64, dtype=torch.bfloat16) for _ in range(3))
    with config.patch(
        emulate_precision_casts=True, fx_graph_cache=False, compile_threads=1
    ):
        graph = make_fx(_pointwise)(*inputs)
        buffer = io.BytesIO()
        torch.save(SerializedFxGraph.capture(graph), buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=False).restore()
        assert _barriers(restored) == _barriers(graph)
        actual = _compiled_graph(restored, inputs)[0]
    assert torch.equal(actual, _pointwise(*inputs)[0])


@pytest.mark.parametrize(
    "descriptor", [SavedForBackwardsAOTOutput, SavedForBackwardsNoVcCheckAOTOutput]
)
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_saved_output_precision_survives_restoration_and_compilation(
    tmp_path, monkeypatch, descriptor, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires an accelerator for compilation")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    torch.manual_seed(74)
    inputs = tuple(
        torch.randn(257, 64, dtype=torch.bfloat16, device=device) for _ in range(3)
    )

    def layer(x, y, z):
        value = x * y + z
        return value.square(), value

    with config.patch(
        emulate_precision_casts=False,
        emulate_precision_casts_on_saved_tensors=True,
        fx_graph_cache=False,
        compile_threads=1,
    ):
        graph = make_fx(layer)(*inputs)
        plain = copy_graph_module(graph)
        output = next(node for node in graph.graph.nodes if node.op == "output")
        output.meta["desc"] = [DummyAOTOutput(0), descriptor(2)]
        restored = SerializedFxGraph.capture(graph).restore()
        restored_output = next(n for n in restored.graph.nodes if n.op == "output")
        assert saved_outputs(restored_output) == saved_outputs(output)
        assert GraphArtifact.input_compatibility_digest(
            graph_module=graph, example_inputs=inputs
        ) != GraphArtifact.input_compatibility_digest(
            graph_module=plain, example_inputs=inputs
        )
        expected = _compiled_graph(graph, inputs)
        actual = _compiled_graph(restored, inputs)
        control = _compiled_graph(plain, inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if device == "cuda":
        # The CPU code generator can elide this cast when full emulation is
        # disabled. Test its persistence there and the device arithmetic here.
        assert torch.equal(actual[0], actual[1].square())
        assert not torch.equal(control[0], control[1].square())

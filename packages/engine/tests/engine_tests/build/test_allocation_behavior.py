"""Static wrapper facts cannot silently turn opaque/value-dependent code into proof."""

from types import SimpleNamespace

import pytest
import torch
from torch._inductor.codegen import wrapper
from torch._inductor.utils import LineContext

from shadowspill.build.allocation_behavior import (
    composed_allocation_behavior,
    wrapper_allocation_behavior,
)


@pytest.mark.parametrize(
    "line",
    [
        "",
        "# source location only",
        "del arg0, arg1",
        "alias = arg0",
        "unused = None",
        "arg0 = copy_if_misaligned(arg0)",
        "stream0 = get_raw_stream(0)",
        "assert_alignment(buf0, 16, 'provider.operation')",
        LineContext(None),
        wrapper.MultiOutputLine(None, "buf1", "buf0", [(tuple, 0)]),
        wrapper.UnbackedSymbolDefsLine(None, "buf0", (), None),
    ],
)
def test_non_data_dependent_wrapper_operations_have_no_unknown_behavior(line):
    record = wrapper_allocation_behavior([line])
    assert record["version"] == 1
    assert record["unresolved"] == record["opaque_calls"] == []


@pytest.mark.parametrize(
    "line",
    [
        "scratch = torch.empty((int(arg0.item()),), device='cuda')",
        "value = unknown(arg0)",
        "if arg0.item(): scratch = allocate()",
        "arg0 = copy_if_misaligned(arg0, unexpected=True)",
        "stream = get_raw_stream(arg0.item())",
        "assert_alignment(unknown(), 16, 'provider.operation')",
        "arg0.data = replacement",
        "del arg0.data",
        wrapper.UnbackedSymbolDefsLine(None, "buf0", (), {"symbol": "value"}),
        wrapper.MultiOutputLine(None, "buf1", "buf0", [(object, 0)]),
        object(),
    ],
)
def test_unknown_wrapper_behavior_remains_unresolved(line):
    assert wrapper_allocation_behavior([line])["unresolved"]


def test_opaque_calls_are_requirements_not_a_static_allocation_guarantee():
    node = SimpleNamespace(get_kernel_name=lambda: "provider.kernel")
    record = wrapper_allocation_behavior(
        [wrapper.ExternKernelAllocLine(None, node)],
        geometry_gaps=["allocation extent: dynamic_output"],
    )
    assert record["opaque_calls"] == ["provider.kernel"]
    assert record["unresolved"] == ["allocation extent: dynamic_output"]


@pytest.mark.parametrize(
    "change",
    [None, "float", "mean", "exclude_self", "overload", "scalar", "symbolic_dim"],
)
def test_integer_scatter_rule_does_not_admit_other_reductions_or_dynamic_geometry(
    change,
):
    tensor = SimpleNamespace(
        get_dtype=lambda: torch.float32 if change == "float" else torch.int64,
        get_device=lambda: torch.device("cuda"),
    )
    node = SimpleNamespace(
        python_kernel_name="aten.scatter_.src"
        if change == "overload"
        else "aten.scatter_reduce_.two",
        src_is_tensor=change != "scalar",
        kwargs={
            "reduce": "mean" if change == "mean" else "sum",
            "include_self": change != "exclude_self",
        },
        constant_args=("unknown" if change == "symbolic_dim" else 0,),
        inputs=[tensor] * 3,
    )
    line = wrapper.ScatterFallbackLine(None, node)
    record = wrapper_allocation_behavior([line])
    assert bool(record["unresolved"]) == (change is not None)
    assert wrapper_allocation_behavior([line], geometry_gaps=["dynamic count"])[
        "unresolved"
    ]


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_compiled_router_counts_reuse_fixed_geometry_across_assignment_values(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from torch.fx.experimental.proxy_tensor import make_fx

    from shadowspill.build.compilation import compile_task
    from shadowspill.pytorch.capture.artifacts import GraphArtifact
    from shadowspill.pytorch.executable import restore_executable

    def function(indices, active):
        counts = torch.zeros(4, dtype=torch.int64, device=indices.device)
        return (counts.scatter_add(0, indices, active),)

    indices = torch.zeros(12, dtype=torch.int64, device=device)
    active = torch.ones_like(indices)
    artifact = GraphArtifact.capture(
        kind="forward",
        graph_module=make_fx(function)(indices, active),
        example_inputs=(indices, active),
    )
    compiled = compile_task(artifact)
    assert compiled.memory["allocation_behavior"]["unresolved"] == []
    execute = restore_executable(compiled.payload)
    for divisor in (1, 2, 4):
        indices.copy_(torch.arange(12, device=device) % divisor)
        active.copy_(torch.arange(12, device=device) % 3 != 0)
        (actual,) = execute(indices, active)
        torch.testing.assert_close(actual, function(indices, active)[0])
        assert actual.untyped_storage().nbytes() == 4 * 8


def test_composition_propagates_every_child_requirement_and_missing_evidence():
    first = wrapper_allocation_behavior(
        [
            wrapper.ExternKernelAllocLine(
                None, SimpleNamespace(get_kernel_name=lambda: "a")
            )
        ]
    )
    second = wrapper_allocation_behavior(["tmp = unknown()"])
    composed = composed_allocation_behavior(
        [{"allocation_behavior": record} for record in (first, second, first)]
    )
    assert composed["opaque_calls"] == ["a"]
    assert composed["unresolved"] == second["unresolved"]
    assert composed_allocation_behavior([{}])["unresolved"]

"""Static wrapper facts cannot silently turn opaque/value-dependent code into proof."""

from types import SimpleNamespace

import pytest
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

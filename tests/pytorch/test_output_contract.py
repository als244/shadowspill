from __future__ import annotations

import torch
import torch.nn as nn
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten

from shadowspill.pytorch.output_contract import (
    ExplicitMutation,
    StorageRootKind,
    capture_task_storage_contract,
)

_custom_definitions = torch.library.Library("shadowspill_contract_test", "DEF")
_custom_definitions.define("alias(Tensor(a) value) -> Tensor(a)")
_custom_definitions.define("mutate(Tensor(a!) value) -> Tensor(a!)")
_custom_implementations = torch.library.Library("shadowspill_contract_test", "IMPL")
_custom_implementations.impl(
    "alias", lambda value: value[1:], "CompositeExplicitAutograd"
)
_custom_implementations.impl(
    "mutate", lambda value: value.add_(1), "CompositeExplicitAutograd"
)


def test_repeated_output_node_is_one_semantic_storage() -> None:
    def function(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = torch.sin(value)
        return result, result

    value = torch.randn(4, 8)
    graph = make_fx(function)(value)
    contract = capture_task_storage_contract(graph, (value,))

    assert len(contract.roots) == 1
    assert contract.roots[0].kind is StorageRootKind.FRESH
    assert tuple(view.root_id for view in contract.output_views) == (0, 0)
    assert tuple(view.leaf_index for view in contract.output_views) == (0, 1)


def test_input_view_is_declared_without_execution_storage_observation() -> None:
    def function(value: torch.Tensor) -> torch.Tensor:
        return value[2:10]

    value = torch.randn(16)
    graph = make_fx(function)(value)
    contract = capture_task_storage_contract(graph, (value,))

    assert len(contract.roots) == 1
    assert contract.roots[0].kind is StorageRootKind.INPUT
    assert contract.roots[0].source_input == 0
    assert contract.output_views[0].offset_bytes == 2 * value.element_size()
    assert contract.output_views[0].span_bytes == 8 * value.element_size()


def test_views_of_fresh_intermediate_share_one_compact_bundle() -> None:
    def function(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = torch.sin(value)
        return result[2:8], result[8:14]

    value = torch.randn(16)
    graph = make_fx(function)(value)
    contract = capture_task_storage_contract(graph, (value,))

    assert len(contract.roots) == 1
    assert contract.roots[0].kind is StorageRootKind.FRESH
    assert tuple(view.root_id for view in contract.output_views) == (0, 0)
    assert tuple(view.offset_bytes for view in contract.output_views) == (0, 24)
    assert contract.roots[0].minimum_span_bytes == 48


def test_distinct_outputs_do_not_alias_from_example_storage_coincidence() -> None:
    class Pair(nn.Module):
        def forward(
            self, value: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.sin(value), torch.cos(value)

    value = torch.randn(8)
    graph = make_fx(Pair())(value)
    output = next(node for node in graph.graph.nodes if node.op == "output")
    left, right = output.args[0]
    # The contract follows FX producer identity, not FakeTensor storage IDs.
    right.meta["val"] = left.meta["val"]
    contract = capture_task_storage_contract(graph, (value,))

    assert tuple(root.kind for root in contract.roots) == (
        StorageRootKind.FRESH,
        StorageRootKind.FRESH,
    )
    assert tuple(view.root_id for view in contract.output_views) == (0, 1)


def test_aliased_inputs_use_one_canonical_input_root() -> None:
    def function(
        left: torch.Tensor, right: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return left[1:5], right[3:7]

    base = torch.randn(16)
    left = base[:8]
    right = base[4:12]
    graph = make_fx(function)(left, right)
    contract = capture_task_storage_contract(graph, (left, right))

    assert len(contract.roots) == 1
    assert contract.roots[0].source_input == 0
    assert tuple(view.offset_bytes for view in contract.output_views) == (4, 28)


def test_aliased_input_views_may_have_different_dtypes() -> None:
    def function(
        floating: torch.Tensor, integer: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return floating[1:3], integer[2:4]

    floating = torch.randn(8, dtype=torch.float32)
    integer = floating.view(torch.int32)
    graph = make_fx(function)(floating, integer)
    contract = capture_task_storage_contract(graph, (floating, integer))

    assert len(contract.roots) == 1
    assert tuple(view.dtype for view in contract.output_views) == (
        "torch.float32",
        "torch.int32",
    )


def test_native_multi_result_operation_has_distinct_fresh_roots() -> None:
    def function(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.max(value, dim=1)

    value = torch.randn(4, 8)
    graph = make_fx(function)(value)
    contract = capture_task_storage_contract(graph, (value,))

    assert len(contract.roots) == 2
    assert tuple(root.producer_result for root in contract.roots) == (0, 1)
    assert tuple(view.root_id for view in contract.output_views) == (0, 1)


def test_zero_length_output_has_a_root_without_required_storage() -> None:
    def function(value: torch.Tensor) -> torch.Tensor:
        return torch.sin(value)[:0]

    value = torch.randn(8)
    graph = make_fx(function)(value)
    contract = capture_task_storage_contract(graph, (value,))

    assert contract.roots[0].minimum_span_bytes == 0
    assert contract.output_views[0].span_bytes == 0


def test_scalar_and_none_leaves_have_no_storage_binding() -> None:
    def function(value: torch.Tensor) -> tuple[torch.Tensor, int, None]:
        return torch.sin(value), 7, None

    value = torch.randn(8)
    graph = make_fx(function)(value)
    output = next(node for node in graph.graph.nodes if node.op == "output")
    leaves, _ = tree_flatten(output.args[0])
    contract = capture_task_storage_contract(graph, (value,))

    assert len(leaves) == 3
    assert tuple(view.leaf_index for view in contract.output_views) == (0,)


def test_input_mutation_is_reported_from_operator_schema() -> None:
    def function(value: torch.Tensor) -> torch.Tensor:
        value.add_(1)
        return value

    value = torch.randn(8)
    graph = make_fx(function)(value)
    contract = capture_task_storage_contract(graph, (value,))

    assert contract.roots[0].kind is StorageRootKind.INPUT
    assert len(contract.mutations) == 1
    assert contract.mutations[0].input_position == 0
    assert contract.mutations[0].replacement_output_leaf is None


def test_functional_mutation_keeps_fresh_root_and_names_replacement() -> None:
    def function(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        replacement = value + 1
        return replacement, replacement[2:]

    value = torch.randn(8)
    graph = make_fx(function)(value)
    contract = capture_task_storage_contract(
        graph,
        (value,),
        explicit_mutations=(ExplicitMutation(0, 0, "state"),),
    )

    assert contract.roots[0].kind is StorageRootKind.FRESH
    assert tuple(view.root_id for view in contract.output_views) == (0, 0)
    assert contract.mutations[0].input_position == 0
    assert contract.mutations[0].replacement_output_leaf == 0
    assert contract.mutations[0].argument_name == "state"


def test_contract_digest_is_deterministic() -> None:
    def function(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = torch.sin(value)
        return result[:4], result[4:]

    value = torch.randn(8)
    graph = make_fx(function)(value)
    first = capture_task_storage_contract(graph, (value,))
    second = capture_task_storage_contract(graph, (value,))

    assert first == second
    assert first.compatibility_digest == second.compatibility_digest
    assert first.to_json() == second.to_json()


def test_registered_custom_alias_schema_controls_semantic_root() -> None:
    def function(value: torch.Tensor) -> torch.Tensor:
        return torch.ops.shadowspill_contract_test.alias(value)

    value = torch.randn(8)
    contract = capture_task_storage_contract(make_fx(function)(value), (value,))

    assert contract.roots[0].kind is StorageRootKind.INPUT
    assert contract.output_views[0].offset_bytes == value.element_size()


def test_registered_custom_mutation_schema_controls_semantic_write() -> None:
    def function(value: torch.Tensor) -> torch.Tensor:
        return torch.ops.shadowspill_contract_test.mutate(value)

    value = torch.randn(8)
    contract = capture_task_storage_contract(make_fx(function)(value), (value,))

    assert contract.roots[0].kind is StorageRootKind.INPUT
    assert len(contract.mutations) == 1
    assert contract.mutations[0].producer_target.endswith("mutate.default")


def test_semantic_extraction_has_no_compiler_profiler_or_allocator_dependency() -> None:
    import inspect

    import shadowspill.pytorch.output_contract as module

    source = inspect.getsource(module)
    assert "torch._inductor" not in source
    assert "._telemetry" not in source
    assert ".compiler" not in source
    assert ".profiling" not in source
    assert ".runtime" not in source

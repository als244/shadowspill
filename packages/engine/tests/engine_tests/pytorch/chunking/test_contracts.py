"""Axis contracts reject ambiguous semantics and preserve reusable geometry."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from engine_tests.pytorch.chunking.test_axis_vjp import executable
from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.chunking import AxisChunks, AxisFrom, ChunkAxis, OperatorRule
from shadowspill.pytorch.chunking.transform import chunk_artifact
from shadowspill.pytorch.executable import restore_executable
from shadowspill.pytorch.operators.materialize import MATERIALIZE_GRADIENT


@torch.library.custom_op("row_contract_tests::statistics", mutates_args=())
def statistics(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x.square(), x.sum(-1).flatten()


@statistics.register_fake
def fake_statistics(x):
    return torch.empty_like(x), x.new_empty(x.numel() // x.shape[-1])


def artifact(function, values):
    return GraphArtifact.capture(
        kind="inference", graph_module=make_fx(function)(*values), example_inputs=values
    )


@pytest.mark.parametrize("size", [1, 7, 30])
def test_materialized_gradient_slices_keep_independent_storage_and_reductions(size):
    x = torch.randn(17, 5, dtype=torch.float64)
    dy = torch.randn(17, 9, dtype=torch.float64)

    def backward(x, dy):
        bank = x.T @ dy
        return (
            MATERIALIZE_GRADIENT(x[:, 1:]),
            MATERIALIZE_GRADIENT(bank[0]),
            MATERIALIZE_GRADIENT(bank[1]),
        )

    full = GraphArtifact.capture(
        kind="backward",
        graph_module=make_fx(backward)(x, dy),
        example_inputs=(x, dy),
    )
    chunked = chunk_artifact(
        full, (ChunkAxis(0), ChunkAxis(0)), 17, AxisChunks(size, {0: 0, 1: 0})
    )
    actual = executable(chunked)(x, dy)
    torch.testing.assert_close(actual, backward(x, dy), rtol=1e-12, atol=1e-12)
    storages = {value.untyped_storage()._cdata for value in actual}
    assert len(storages) == len(actual)
    assert x.untyped_storage()._cdata not in storages
    assert chunked.chunking.outputs == (ChunkAxis(0), "sum", "sum")


def test_unknown_row_payload_version_requires_rebuilding_the_program():
    with pytest.raises(ValueError, match="row composition version; rebuild"):
        restore_executable(b"SHADOWSPILL-ROW-COMPOSITION-999\ninvalid-payload")


def test_embedding_table_cannot_be_partitioned_as_ordered_token_rows():
    weight = torch.randn(17, 5)
    indices = torch.tensor([0, 1, 0, 1])
    full = artifact(
        lambda weight, indices: (torch.nn.functional.embedding(indices, weight),),
        (weight, indices),
    )
    with pytest.raises(CaptureError, match="shared table"):
        chunk_artifact(full, (ChunkAxis(0), None), 17, AxisChunks(7, {0: 0}))


@pytest.mark.parametrize("scale_by_frequency", [False, True])
def test_embedding_gradient_counts_cannot_be_restarted_in_each_chunk(
    scale_by_frequency,
):
    gradient = torch.randn(17, 5, dtype=torch.float64)
    indices = torch.tensor([0, 1, 0, 0, 2, 3, 1, 2, 0, 0, 0, 3, 2, 1, 1, 1, 0])

    def backward(gradient, indices):
        return (
            torch.ops.aten.embedding_dense_backward.default(
                gradient, indices, 4, -1, scale_by_frequency
            ),
        )

    full = artifact(backward, (gradient, indices))
    config = AxisChunks(7, {0: 0, 1: 0})
    if scale_by_frequency:
        with pytest.raises(CaptureError, match="complete-axis extents"):
            chunk_artifact(full, (ChunkAxis(0), ChunkAxis(0)), 17, config)
    else:
        chunked = chunk_artifact(full, (ChunkAxis(0), ChunkAxis(0)), 17, config)
        torch.testing.assert_close(
            executable(chunked)(gradient, indices), backward(gradient, indices)
        )


def test_opaque_rule_maps_flattened_batch_head_statistics():
    x = torch.randn(2, 17, 3, 5, dtype=torch.float64)
    full = artifact(statistics, (x,))
    with pytest.raises(CaptureError, match="register an OperatorRule"):
        chunk_artifact(full, (ChunkAxis(1),), 17, AxisChunks(7, {0: 1}))
    rule = OperatorRule(
        "row_contract_tests.statistics.default", "1", (AxisFrom(0), AxisFrom(0, 1))
    )
    chunked = chunk_artifact(
        full, (ChunkAxis(1),), 17, AxisChunks(7, {0: 1}, rules=(rule,))
    )
    torch.testing.assert_close(executable(chunked)(x), statistics(x), rtol=0, atol=0)
    assert chunked.chunking.outputs[1] == ChunkAxis(0, 2, 3)


def test_unused_pure_operator_does_not_require_a_row_contract():
    x = torch.randn(17, 5)

    def region(x):
        statistics(x)
        return (x.sin(),)

    full = artifact(region, (x,))
    assert any(
        "row_contract_tests" in str(n.target) for n in full.graph_module.graph.nodes
    )
    chunked = chunk_artifact(full, (ChunkAxis(0),), 17, AxisChunks(7, {0: 0}))
    torch.testing.assert_close(executable(chunked)(x), (x.sin(),))


def test_unused_random_operator_still_requires_an_effect_contract():
    x = torch.randn(17, 5)

    def region(x):
        torch.rand_like(x)
        return (x.sin(),)

    full = artifact(region, (x,))
    with pytest.raises(CaptureError, match="no row-separability rule"):
        chunk_artifact(full, (ChunkAxis(0),), 17, AxisChunks(7, {0: 0}))


def test_checked_static_metadata_does_not_keep_unused_pure_values_live():
    x = torch.randn(17, 5)

    def region(x):
        value, _ = statistics(x)
        torch.ops.aten._assert_tensor_metadata.default(
            value, None, None, torch.float32, device=torch.device("cpu")
        )
        return (x.sin(),)

    full = artifact(region, (x,))
    chunked = chunk_artifact(full, (ChunkAxis(0),), 17, AxisChunks(7, {0: 0}))
    torch.testing.assert_close(executable(chunked)(x), (x.sin(),))


def test_literal_geometry_factories_only_read_row_input_metadata():
    x = torch.randn(2, 17, 5)

    def region(x):
        shared = x.new_zeros((1, 5)) + x.new_ones((1, 5)) + x.new_full((1, 5), 3)
        return x + shared, shared

    full = artifact(region, (x,))
    chunked = chunk_artifact(full, (ChunkAxis(1),), 17, AxisChunks(7, {0: 1}))
    torch.testing.assert_close(executable(chunked)(x), region(x))


def test_softmax_vjp_cannot_pair_differently_factored_row_coordinates():
    probabilities = torch.randn(34, 5).softmax(-1)
    gradient = torch.randn_like(probabilities)
    full = artifact(
        lambda gradient, probabilities: (
            torch.ops.aten._softmax_backward_data.default(
                gradient, probabilities, -1, torch.float32
            ),
        ),
        (gradient, probabilities),
    )
    with pytest.raises(CaptureError, match="differently aligned chunk domains"):
        chunk_artifact(
            full,
            (ChunkAxis(0, before=2), ChunkAxis(0, after=2)),
            17,
            AxisChunks(7, {0: ChunkAxis(0, before=2), 1: ChunkAxis(0, after=2)}),
        )


@pytest.mark.parametrize("misaligned", [0, 2, 3])
def test_layer_norm_vjp_cannot_pair_differently_factored_row_coordinates(misaligned):
    x = torch.randn(34, 5, dtype=torch.float64)
    weight, bias = torch.randn(2, 5, dtype=torch.float64)
    _, mean, rstd = torch.ops.aten.native_layer_norm.default(x, [5], weight, bias, 1e-5)
    values = (torch.randn_like(x), x, mean, rstd, weight, bias)

    def backward(gradient, x, mean, rstd, weight, bias):
        return torch.ops.aten.native_layer_norm_backward.default(
            gradient, x, [5], mean, rstd, weight, bias, [True, True, True]
        )

    axes = [ChunkAxis(0, before=2)] * 4 + [None, None]
    axes[misaligned] = ChunkAxis(0, after=2)
    full = artifact(backward, values)
    with pytest.raises(CaptureError, match="differently aligned chunk domains"):
        chunk_artifact(
            full,
            tuple(axes),
            17,
            AxisChunks(7, {i: axis for i, axis in enumerate(axes) if axis is not None}),
        )


def test_layer_norm_affine_parameters_cannot_be_mapped_as_token_rows():
    x = torch.randn(17, 17)
    weight, bias = torch.randn(2, 17)
    full = artifact(
        lambda x, weight, bias: torch.ops.aten.native_layer_norm.default(
            x, [17], weight, bias, 1e-5
        ),
        (x, weight, bias),
    )
    with pytest.raises(CaptureError, match="shared weight and bias"):
        chunk_artifact(
            full, (ChunkAxis(0), ChunkAxis(0), None), 17, AxisChunks(7, {0: 0, 1: 0})
        )


def test_opaque_shared_output_must_keep_its_geometry_in_every_child():
    x = torch.randn(2, 17, 3, 5)
    full = artifact(statistics, (x,))
    rule = OperatorRule(
        "row_contract_tests.statistics.default", "1", (None, AxisFrom(0, 1))
    )
    with pytest.raises(
        CaptureError, match="output 0 violates its declared chunk geometry"
    ):
        chunk_artifact(full, (ChunkAxis(1),), 17, AxisChunks(7, {0: 1}, rules=(rule,)))


def test_partial_fresh_output_keeps_its_boundary_offset():
    x = torch.randn(2, 17, 12, dtype=torch.float64)
    full = artifact(lambda x: (x.sin()[..., 6:],), (x,))
    chunked = chunk_artifact(full, (ChunkAxis(1),), 17, AxisChunks(7, {0: 1}))
    (result,) = executable(chunked)(x)
    torch.testing.assert_close(result, x.sin()[..., 6:])
    assert result.storage_offset() == 6
    assert result.stride() == x.stride()


def test_expanded_output_copies_each_physical_element_once():
    x = torch.randn(17, 5, dtype=torch.float64)

    def region(x):
        return (x.sin().unsqueeze(1).expand(-1, 4, -1),)

    full = artifact(region, (x,))
    chunked = chunk_artifact(full, (ChunkAxis(0),), 17, AxisChunks(7, {0: 0}))
    (actual,) = executable(chunked)(x)
    torch.testing.assert_close(actual, region(x)[0], rtol=0, atol=0)
    assert actual.stride() == (5, 0, 1)
    assert actual.untyped_storage().nbytes() == x.numel() * x.element_size()


def test_broadcast_gradient_assembly_preserves_its_shared_storage():
    x = torch.randn(17, 5, dtype=torch.float64)

    def gradient(x):
        return (x.sum(0).expand(4, 5),)

    full = GraphArtifact.capture(
        kind="backward", graph_module=make_fx(gradient)(x), example_inputs=(x,)
    )
    chunked = chunk_artifact(full, (ChunkAxis(0),), 17, AxisChunks(7, {0: 0}))
    assert not chunked.chunking.accumulating_children
    (actual,) = executable(chunked)(x)
    torch.testing.assert_close(actual, gradient(x)[0], rtol=1e-12, atol=1e-12)
    assert actual.stride() == (0, 1)


def test_static_feature_coordinates_remain_shared_between_chunks():
    x = torch.randn(2, 17, 12, dtype=torch.float64)

    def region(x):
        frequencies = 10000 ** (-torch.arange(0, 12, 2).double() / 12)
        return (x[..., ::2] * frequencies,)

    full = artifact(region, (x,))
    chunked = chunk_artifact(full, (ChunkAxis(1),), 17, AxisChunks(7, {0: 1}))
    torch.testing.assert_close(executable(chunked)(x), region(x))


def test_generated_full_sequence_positions_cannot_reset_inside_each_chunk():
    x = torch.randn(17, 5, dtype=torch.float64)
    full = artifact(lambda x: (x + torch.arange(17)[:, None],), (x,))
    with pytest.raises(CaptureError, match="row chunk shape 7 failed"):
        chunk_artifact(full, (ChunkAxis(0),), 17, AxisChunks(7, {0: 0}))


def test_common_child_geometry_reuses_identity_across_sequence_lengths():
    children = []
    parents = []
    for rows in (17, 21):
        x = torch.randn(2, rows, 5)
        full = artifact(lambda x: (x.sin().reshape(-1, 5),), (x,))
        chunked = chunk_artifact(full, (ChunkAxis(1),), rows, AxisChunks(7, {0: 1}))
        children.append(dict(chunked.chunking.children)[7].compatibility_digest)
        parents.append(chunked.compatibility_digest)
    assert children[0] == children[1]
    assert parents[0] != parents[1]


def test_aliased_reductions_are_refused_before_double_accumulation():
    x = torch.randn(17, 5)

    def function(x):
        value = x.sum(0)
        return value, value.view(1, 5)

    full = artifact(function, (x,))
    with pytest.raises(CaptureError, match="one canonical output"):
        chunk_artifact(full, (ChunkAxis(0),), 17, AxisChunks(7, {0: 0}))


@pytest.mark.parametrize(
    "function",
    [
        lambda x: (x.sum(0).softmax(0),),
        lambda x: (x.sum(0).amax(),),
        lambda x: (1 / x.sum(0),),
        lambda x: (x.sum(0).square(),),
    ],
)
def test_nonlinear_combinations_of_chunk_sums_are_refused(function):
    x = torch.randn(17, 5)
    full = artifact(function, (x,))
    with pytest.raises(CaptureError, match="row reduction"):
        chunk_artifact(full, (ChunkAxis(0),), 17, AxisChunks(7, {0: 0}))


def test_shared_addmm_bias_is_not_added_once_per_chunk():
    x, y, bias = torch.randn(17, 5), torch.randn(17, 7), torch.randn(5, 7)
    full = artifact(lambda x, y, b: (torch.addmm(b, x.T, y),), (x, y, bias))
    with pytest.raises(CaptureError, match="shared value to a row reduction"):
        chunk_artifact(
            full, (ChunkAxis(0), ChunkAxis(0), None), 17, AxisChunks(7, {0: 0, 1: 0})
        )


def test_zero_addmm_beta_does_not_duplicate_a_shared_bias():
    x, y, bias = torch.randn(17, 5), torch.randn(17, 7), torch.randn(5, 7)
    full = artifact(lambda x, y, b: (torch.addmm(b, x.T, y, beta=0),), (x, y, bias))
    chunked = chunk_artifact(
        full, (ChunkAxis(0), ChunkAxis(0), None), 17, AxisChunks(7, {0: 0, 1: 0})
    )
    torch.testing.assert_close(executable(chunked)(x, y, bias), (x.T @ y,))


def test_random_operations_require_an_explicit_replay_contract():
    x = torch.randn(17, 5)
    full = artifact(lambda x: (x + torch.rand_like(x),), (x,))
    with pytest.raises(CaptureError, match="no row-separability rule"):
        chunk_artifact(full, (ChunkAxis(0),), 17, AxisChunks(7, {0: 0}))


@torch.library.custom_op("row_contract_tests::sort_rows", mutates_args=())
def sort_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    order = x[:, 0].argsort()
    return x[order], order


@sort_rows.register_fake
def fake_sort_rows(x):
    return torch.empty_like(x), x.new_empty(x.shape[0], dtype=torch.int64)


@torch.library.custom_op("row_contract_tests::restore_rows", mutates_args=())
def restore_rows(x: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    return x[order.argsort()]


@restore_rows.register_fake
def fake_restore_rows(x, order):
    return torch.empty_like(x)


@pytest.mark.parametrize("batch", [1, 2])
def test_chunk_local_permutations_restore_rows_before_assembly(batch):
    from shadowspill.pytorch.chunking import ChunkLocal

    x = torch.randn(batch * 17, 5)

    def region(x):
        sorted_x, order = sort_rows(x)
        # ATen split metadata is a list, even though getitem accepts either
        # container. Preserve its pytree through local-coordinate propagation.
        pieces = sorted_x.sin().split((2, 3), dim=1)
        return (restore_rows(torch.cat(pieces, dim=1), order),)

    rules = (
        OperatorRule(
            "row_contract_tests.sort_rows.default", "1", (ChunkLocal(0), ChunkLocal(0))
        ),
        OperatorRule(
            "row_contract_tests.restore_rows.default", "1", (ChunkAxis(0, batch),)
        ),
    )
    config = AxisChunks(7, {0: ChunkAxis(0, batch)}, rules=rules)
    full = artifact(region, (x,))
    chunked = chunk_artifact(full, (ChunkAxis(0, batch),), 17, config)
    torch.testing.assert_close(executable(chunked)(x), (x.sin(),), rtol=0, atol=0)
    with pytest.raises(CaptureError, match="chunk-local scratch cannot escape"):
        chunk_artifact(artifact(sort_rows, (x,)), (ChunkAxis(0, batch),), 17, config)


def test_chunk_local_coordinates_cannot_mix_with_ordered_row_values():
    from shadowspill.pytorch.chunking import ChunkLocal

    x = torch.randn(17, 5)
    rule = OperatorRule(
        "row_contract_tests.sort_rows.default", "1", (ChunkLocal(0), ChunkLocal(0))
    )
    full = artifact(lambda x: (sort_rows(x)[0] + x,), (x,))
    with pytest.raises(CaptureError, match="explicit row-restoration rule"):
        chunk_artifact(full, (ChunkAxis(0),), 17, AxisChunks(7, {0: 0}, rules=(rule,)))


def test_unrelated_shared_embedding_gradient_is_not_counted_per_chunk():
    x = torch.randn(17, 5)
    gradient = torch.randn(4, 5)
    indices = torch.tensor([0, 2, 2, 1])

    def region(x, gradient, indices):
        shared = torch.ops.aten.embedding_dense_backward.default(
            gradient, indices, 3, -1, False
        )
        return x.sin(), shared

    full = artifact(region, (x, gradient, indices))
    chunked = chunk_artifact(
        full, (ChunkAxis(0), None, None), 17, AxisChunks(7, {0: 0})
    )
    torch.testing.assert_close(
        executable(chunked)(x, gradient, indices), region(x, gradient, indices)
    )


def test_scalar_output_literals_survive_composition():
    x = torch.randn(17, 5)
    full = artifact(lambda x: (x.sin(), True, 3.5, "ok", None), (x,))
    chunked = chunk_artifact(full, (ChunkAxis(0),), 17, AxisChunks(7, {0: 0}))
    values = executable(chunked)(x)
    torch.testing.assert_close(values[0], x.sin())
    assert values[1:] == (True, 3.5, "ok", None)


def test_stacked_summaries_add_but_cannot_mix_in_shared_constants():
    x = torch.randn(17, 5, dtype=torch.float64)
    config = AxisChunks(7, {0: 0})

    def region(x):
        return (torch.stack((x.sum(0), x.square().sum(0))),)

    chunked = chunk_artifact(artifact(region, (x,)), (ChunkAxis(0),), 17, config)
    torch.testing.assert_close(executable(chunked)(x), region(x))
    with pytest.raises(CaptureError, match="different row/reduction domains"):
        chunk_artifact(
            artifact(lambda x: (torch.stack((x.sum(0), torch.ones(5))),), (x,)),
            (ChunkAxis(0),),
            17,
            config,
        )


@pytest.mark.parametrize("left", [True, False])
def test_matrix_broadcasting_preserves_the_output_row_dimension(left):
    from torch.fx import Graph, GraphModule

    x, weight = torch.randn(17, 5), torch.randn(3, 5, 7)
    if not left:
        x, weight = x.T, weight.transpose(1, 2)
    values = (x, weight) if left else (weight, x)
    graph = Graph()
    a, b = graph.placeholder("a"), graph.placeholder("b")
    output = graph.call_function(torch.ops.aten.matmul.default, (a, b))
    graph.output((output,))
    full = GraphArtifact.capture(
        kind="inference", graph_module=GraphModule({}, graph), example_inputs=values
    )
    axes = (ChunkAxis(0), None) if left else (None, ChunkAxis(1))
    config = AxisChunks(7, {0: 0} if left else {1: 1})
    chunked = chunk_artifact(full, axes, 17, config)
    torch.testing.assert_close(executable(chunked)(*values), (values[0] @ values[1],))


def test_row_positions_index_a_shared_table_without_resetting_positions():
    table = torch.randn(256, 5)
    indices = torch.arange(2 * 17).reshape(2, 17) * 3 + 7
    full = artifact(lambda table, positions: (table[positions],), (table, indices))
    chunked = chunk_artifact(full, (None, ChunkAxis(1)), 17, AxisChunks(7, {1: 1}))
    torch.testing.assert_close(executable(chunked)(table, indices), (table[indices],))

"""Explicit representation coordinates preserve payloads and dense optimizer state."""

import io
from dataclasses import replace

import pytest
import torch

from engine_tests.checkpoint.test_conversion import (
    HostRank,
    family,
    save_sources,
    temporary,
)
from engine_tests.checkpoint.test_exports import decode
from engine_tests.checkpoint.test_state import plan
from shadowspill.artifacts.component_layouts import (
    ComponentLayout,
    attach_component_layouts,
)
from shadowspill.artifacts.program import Program
from shadowspill.artifacts.state import StateId
from shadowspill.checkpoint import write_weights
from shadowspill.checkpoint.compatibility import describe_family, require_compatible
from shadowspill.checkpoint.conversion import convert_state
from shadowspill.checkpoint.state import PlanState
from shadowspill.checkpoint.weights_conversion import convert_weights
from shadowspill.training.components import ModelBuild


def states(count, dim, *, mapped=True, axes=(0, 1), encoding="example/bf16@1"):
    result = []
    for state in family(count, dim):
        source = state.plan.program
        source = replace(
            source,
            bindings={
                **source.bindings,
                "registrations": [
                    {**row, "component_path": ("payload",)}
                    for row in source.bindings["registrations"]
                ],
                "optimizer_bindings": [
                    {
                        **row,
                        **(
                            {"component_path": ("payload",)}
                            if row["role"] == "compute_copy"
                            else {}
                        ),
                    }
                    for row in source.bindings["optimizer_bindings"]
                ],
            },
        )
        if mapped:
            source = attach_component_layouts(
                source,
                [
                    ComponentLayout(name, ("payload",), axes, encoding)
                    for name in ("linear.weight", "head.weight")
                ],
            )
        result.append(PlanState(plan(source), {}, logical_rank=state.logical_rank))
    return result


def test_equal_component_geometry_without_a_declaration_is_still_rejected():
    with pytest.raises(ValueError, match="explicit logical map"):
        tuple(states(2, 0, mapped=False)[0].logical_tensors())
    declared = states(2, 0)
    for rank, state in enumerate(declared):
        tensors = {row.tensor.identity: row for row in state.logical_tensors()}
        component = tensors[StateId("parameter", "linear.weight", ("payload",))]
        master = tensors[StateId("master", "linear.weight")]
        assert component.view.shape == master.view.shape == (4, 12)
        assert component.view.origin == master.view.origin == (2 * (1 - rank), 0)
        assert component.tensor.geometry["dtype"] == "torch.bfloat16"
        assert master.tensor.geometry["dtype"] == "torch.float32"
        assert (
            tensors[StateId("parameter", "head.weight", ("payload",))].archive
            == component.archive
        )
        copied = Program.from_dict(state.plan.program.to_dict())
        assert tuple(
            PlanState(plan(copied), {}, logical_rank=rank).logical_tensors()
        ) == tuple(state.logical_tensors())


@pytest.mark.parametrize("source_count,target_count", [(1, 4), (4, 1), (2, 4), (4, 2)])
def test_bounded_checkpoint_redistribution_preserves_payload_masters_and_moments(
    tmp_path, source_count, target_count
):
    sources = save_sources(tmp_path, states(source_count, 0), "master")
    targets = states(target_count, 1)
    with convert_state(
        sources,
        targets,
        tmp_path,
        scratch_bytes=128,
        max_disk_bytes=4096,
        initialize_temporary=temporary,
    ) as converted:
        for rank, state in enumerate(targets):
            expected = HostRank(state, initialize=True)
            actual = HostRank(state)

            def receive(chunk, value, actual=actual):
                start = chunk.offset - chunk.region_offset
                actual.values[chunk.name][start : start + value.numel()].copy_(value)

            converted.ranks[rank].export_tensors(receive, chunk_bytes=16)
            for name, value in expected.values.items():
                if name == "optimizer_group.0.lr":
                    torch.testing.assert_close(
                        actual.values[name], torch.full_like(value, 0.5), rtol=0, atol=0
                    )
                else:
                    torch.testing.assert_close(
                        actual.values[name], value, rtol=0, atol=0
                    )


def test_master_export_keeps_explicit_component_payload_and_weight_only_conversion(
    tmp_path,
):
    sources = []
    for rank, state in enumerate(states(2, 0)):
        worker = HostRank(state, initialize=True)
        stream = io.BytesIO()
        write_weights(worker, stream, chunk_bytes=16)
        manifest, values = decode(stream.getvalue())
        assert set(values) == {"linear.weight.payload", "master.linear.weight"}
        assert not manifest["derived_tensors"]
        path = tmp_path / f"weights-{rank}.ssp"
        path.write_bytes(stream.getvalue())
        sources.append((state, path))
    targets = states(4, 1)
    with convert_weights(
        sources, targets, tmp_path, scratch_bytes=128, max_disk_bytes=4096
    ) as converted:
        for rank, state in enumerate(targets):
            expected = HostRank(state, initialize=True)
            actual = {
                name: torch.empty_like(expected.values[name])
                for name in ("linear.weight.payload", "master.linear.weight")
            }

            def receive(chunk, value, actual=actual):
                assert chunk.name in actual
                start = chunk.offset - chunk.region_offset
                actual[chunk.name][start : start + value.numel()].copy_(value)

            converted.ranks[rank].export_tensors(receive, chunk_bytes=16)
            for name, value in actual.items():
                torch.testing.assert_close(value, expected.values[name], rtol=0, atol=0)
    with (
        pytest.raises(ValueError, match="model interpretation differs"),
        convert_weights(
            sources,
            states(4, 1, encoding="example/other@1"),
            tmp_path,
            scratch_bytes=128,
            max_disk_bytes=4096,
        ),
    ):
        pytest.fail("changed encoding admitted")


def test_changed_encoding_is_not_ordinary_resharding_and_plain_state_is_unchanged():
    with pytest.raises(ValueError, match="interpretation differs"):
        require_compatible(
            describe_family(states(2, 0)),
            describe_family(states(4, 1, encoding="example/other@1")),
        )
    source = family(2, 0)[0].plan.program
    assert attach_component_layouts(source, ()) is source


def test_ownership_cannot_substitute_for_a_validated_component_declaration():
    source = states(2, 0, mapped=False)[0].plan.program
    source = replace(
        source,
        bindings={
            **source.bindings,
            "parameter_ownership": [
                {**row, "component_layout": {"axes": (0, 1), "encoding": "bypass"}}
                for row in source.bindings["parameter_ownership"]
            ],
        },
    )
    with pytest.raises(ValueError, match="ownership cannot declare"):
        tuple(PlanState(plan(source), {}, logical_rank=0).logical_tensors())


def test_permuted_component_axes_keep_physical_strides_and_parameter_placement():
    source = states(2, 0, mapped=False)[0].plan.program
    geometry = source.bindings["model_tensor_geometry"]
    source = replace(
        source,
        bindings={
            **source.bindings,
            "model_tensor_geometry": {
                key: {
                    **row,
                    "shape": tuple(reversed(row["shape"])),
                    "stride": tuple(reversed(row["stride"])),
                }
                for key, row in geometry.items()
            },
            "optimizer_bindings": [
                {
                    **row,
                    **(
                        {
                            "shape": tuple(reversed(row["shape"])),
                            "stride": tuple(reversed(row["stride"])),
                        }
                        if row["role"] == "compute_copy"
                        else {}
                    ),
                }
                for row in source.bindings["optimizer_bindings"]
            ],
        },
    )
    source = attach_component_layouts(
        source,
        [
            ComponentLayout(name, ("payload",), (1, 0), "example/transposed@1")
            for name in ("linear.weight", "head.weight")
        ],
    )
    for rank in range(2):
        values = {
            row.tensor.identity: row
            for row in PlanState(plan(source), {}, logical_rank=rank).logical_tensors()
        }
        component = values[StateId("parameter", "linear.weight", ("payload",))]
        assert component.view.shape == (12, 4)
        assert component.view.local_shape == (12, 2)
        assert component.view.origin == (0, 2 * (1 - rank))
        assert component.view.stride == (1, 12)


@pytest.mark.parametrize(
    "change,error",
    [
        ({"axes": (0, 0)}, "distinct"),
        ({"axes": (True, 1)}, "distinct"),
        ({"component_path": ()}, "component"),
        ({"encoding": ""}, "encoding"),
    ],
)
def test_invalid_coordinate_declarations_fail_without_tensor_reads(change, error):
    with pytest.raises(ValueError, match=error):
        ComponentLayout(
            **{
                **dict(
                    parameter="linear.weight",
                    component_path=("payload",),
                    axes=(0, 1),
                    encoding="example/bf16@1",
                ),
                **change,
            }
        )


@pytest.mark.parametrize(
    "change,error",
    [
        ({"axes": (1, 0)}, "physical shape"),
        ({"axes": (2, 1)}, "parameter rank"),
        ({"parameter": "absent"}, "registered parameter"),
        ({"component_path": ("absent",)}, "registered parameter"),
    ],
)
def test_changed_saved_layout_is_revalidated_on_cpu(change, error):
    source = states(2, 0)[0].plan.program
    layout = ComponentLayout("linear.weight", ("payload",), (0, 1), "example/bf16@1")
    with pytest.raises(ValueError, match=error):
        attach_component_layouts(source, [replace(layout, **change)])
    with pytest.raises(ValueError, match="duplicate"):
        ModelBuild({}, component_layouts=[layout, layout])
    description = ModelBuild({}, component_layouts=[layout])
    assert description.to_dict()["component_layouts"] == [layout.to_dict()]


def scale_states(count, dim, blocks=(1, 128), *, unaligned=False):
    result = []
    for state in states(count, dim, mapped=False):
        source = state.plan.program
        owners = []
        for owner in source.bindings["parameter_ownership"]:
            local = [a * b for a, b in zip(owner["local_shape"], blocks, strict=True)]
            global_shape = [
                a * b for a, b in zip(owner["global_shape"], blocks, strict=True)
            ]
            if unaligned:
                local[dim] -= 1
                global_shape[dim] = local[dim] * count
            owners.append(
                {
                    **owner,
                    "local_shape": local,
                    "global_shape": global_shape,
                    "trainable": False,
                }
            )
        source = replace(
            source,
            bindings={
                **source.bindings,
                "parameter_ownership": owners,
                "optimizer_bindings": [
                    item
                    for item in source.bindings["optimizer_bindings"]
                    if item["name"] == "optimizer_group.0.lr"
                ],
                "persistent_gradients": [],
            },
        )
        source = attach_component_layouts(
            source,
            [
                ComponentLayout(
                    name, ("payload",), (0, 1), "example/block-scales@1", blocks
                )
                for name in ("linear.weight", "head.weight")
            ],
        )
        result.append(PlanState(plan(source), {}, logical_rank=state.logical_rank))
    return result


@pytest.mark.parametrize("source_count,target_count", [(1, 4), (4, 1), (2, 4), (4, 2)])
def test_block_scale_resharding_copies_encoded_values_without_conversion(
    tmp_path, source_count, target_count
):
    sources = []
    for rank, state in enumerate(scale_states(source_count, 0)):
        worker = HostRank(state, initialize=True)
        stream = io.BytesIO()
        write_weights(worker, stream, chunk_bytes=16)
        path = tmp_path / f"block-scales-{rank}.ssp"
        path.write_bytes(stream.getvalue())
        sources.append((state, path))
    targets = scale_states(target_count, 1)
    with convert_weights(
        sources, targets, tmp_path, scratch_bytes=128, max_disk_bytes=4096
    ) as converted:
        for rank, state in enumerate(targets):
            expected = HostRank(state, initialize=True)
            actual = HostRank(state)

            def receive(chunk, value, actual=actual):
                start = chunk.offset - chunk.region_offset
                actual.values[chunk.name][start : start + value.numel()].copy_(value)

            converted.ranks[rank].export_tensors(receive, chunk_bytes=16)
            torch.testing.assert_close(
                actual.values["linear.weight.payload"],
                expected.values["linear.weight.payload"],
                rtol=0,
                atol=0,
            )
            restored = Program.from_dict(state.plan.program.to_dict())
            assert tuple(
                PlanState(plan(restored), {}, logical_rank=rank).logical_tensors()
            ) == tuple(state.logical_tensors())


def test_component_blocks_reject_partial_ownership_and_changed_encodings():
    with pytest.raises(ValueError, match="align to its blocks"):
        tuple(scale_states(2, 1, unaligned=True)[0].logical_tensors())
    with pytest.raises(ValueError, match="interpretation differs"):
        require_compatible(
            describe_family(scale_states(2, 0)),
            describe_family(scale_states(2, 0, blocks=(1, 64))),
        )
    for blocks in ((0, 128), (True, 128), (128,), (1, 1, 1)):
        with pytest.raises(ValueError, match="component blocks"):
            ComponentLayout("w", ("scale",), (0, 1), "example/block@1", blocks)

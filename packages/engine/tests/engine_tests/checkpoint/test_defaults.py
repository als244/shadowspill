"""Conversion uses target declarations without fresh weights or full constants."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from engine_tests.checkpoint.test_conversion import HostRank, family, save_sources
from engine_tests.checkpoint.test_state import plan
from engine_tests.offline.test_state_inventory import training
from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.checkpoint import load_checkpoint, save_checkpoint
from shadowspill.checkpoint.conversion import convert_state
from shadowspill.checkpoint.defaults import TemporaryDefaults
from shadowspill.checkpoint.state import PlanState


def defaults(chunk, initial):
    return TemporaryDefaults(
        [
            SimpleNamespace(
                logical_rank=0, temporary_initializers=lambda: [(chunk, initial)]
            )
        ]
    )


def test_saved_target_defaults_preserve_checkpoint_state_and_clear_gradients():
    source = training()
    source = replace(
        source,
        bindings={
            **source.bindings,
            "gradients": [],
            "optimizer_bindings": [
                {**row, "initialization": {"kind": "constant", "value": 0.25}}
                for row in source.bindings["optimizer_bindings"]
            ],
        },
    )
    state = PlanState(plan(source, kind="forward_backward"), {}, logical_rank=0)
    prepared = TemporaryDefaults([state])
    chunks = {chunk.name: chunk for chunk in state.tensor_chunks()}
    for name, expected in (
        ("gradient.linear.weight", 0),
        ("optimizer_group.0.lr", 0.25),
    ):
        value = torch.full((chunks[name].storage_elements,), -1.0)
        prepared(0, chunks[name], value)
        assert (value == expected).all()
    for name in (
        "linear.weight",
        "master.linear.weight",
        "optimizer.linear.weight.exp_avg",
        "optimizer.linear.weight.step",
    ):
        value = torch.full((1,), -1, dtype=chunks[name].dtype)
        with pytest.raises(ValueError, match="cannot use fresh"):
            prepared(0, chunks[name], value)
        assert value.item() == -1


def test_strided_partial_values_fill_only_borrowed_chunks_without_tensor_copy(
    monkeypatch,
):
    chunk = TensorChunk("constant", (2, 3), torch.int64, 2, 11, (1, 4), 0, 2, 9)
    fill = defaults(chunk, {"kind": "value", "value": [[1, 2, 3], [4, 5, 6]]})
    expected = torch.tensor([0, 0, 2, 5, 0, 0, 3, 6, 0])
    output = torch.empty_like(expected)
    monkeypatch.setattr(
        torch, "as_tensor", lambda *a, **k: pytest.fail("whole constant")
    )
    monkeypatch.setattr(torch, "tensor", lambda *a, **k: pytest.fail("new tensor"))
    for at in range(0, output.numel(), 2):
        fill(0, replace(chunk, offset=chunk.region_offset + at), output[at : at + 2])
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "initial", [None, {"kind": "external"}, {"kind": "parameter", "name": "weight"}]
)
def test_missing_or_weight_dependent_private_initializer_fails_at_preflight(initial):
    with pytest.raises(ValueError, match="no portable default"):
        defaults(TensorChunk("control", (), torch.float32, 0, 1), initial)


@pytest.mark.parametrize("value", [[[1, 2], [3]], [1, 2, 3], [[1, 2], [3, "bad"]]])
def test_inconsistent_value_shape_fails_before_destination_is_opened(value):
    with pytest.raises(ValueError, match=r"ragged|shape differs|portable numbers"):
        defaults(
            TensorChunk("control", (2, 2), torch.float32, 0, 4, (2, 1)),
            {"kind": "value", "value": value},
        )


def test_wrong_request_cannot_write_another_owned_region():
    chunk = TensorChunk("control", (), torch.float32, 0, 1)
    fill = defaults(chunk, {"kind": "constant", "value": 0.5})
    output = torch.full((1,), -1.0)
    with pytest.raises(ValueError, match="owned region"):
        fill(0, replace(chunk, offset=1), output)
    assert output.item() == -1


def test_real_codec_conversion_uses_declared_defaults_and_preserves_counter(tmp_path):
    sources = save_sources(tmp_path, family(2, 0), "master")
    targets = family(4, 1)
    with convert_state(
        sources,
        targets,
        tmp_path,
        scratch_bytes=128,
        max_disk_bytes=4096,
        initialize_temporary=TemporaryDefaults(targets),
    ) as converted:
        for rank in converted.ranks:
            path = tmp_path / f"converted-{rank.logical_rank}.ssp"
            save_checkpoint(rank, path, chunk_bytes=17)
            loaded = HostRank(rank.state)
            load_checkpoint(loaded, path, chunk_bytes=17)
            assert loaded.values["optimizer.linear.weight.step"].item() == 17
            assert loaded.values["optimizer_group.0.lr"].item() == 0

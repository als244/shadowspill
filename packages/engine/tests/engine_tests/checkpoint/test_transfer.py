"""Actual rank archives stream state across layouts before a CPU optimizer update."""

from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from engine_tests.checkpoint.test_mapping import view
from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.checkpoint import save_checkpoint
from shadowspill.checkpoint.reading import read_rank
from shadowspill.checkpoint.transfer import SourceView, TargetView, transfer_tensor


class StoredView:
    def __init__(self, mapping, values, rank):
        self.plan = SimpleNamespace(program=SimpleNamespace(digest="transfer-source"))
        self.plans = {}
        self.logical_rank = rank
        self.chunk = TensorChunk(
            "state",
            mapping.local_shape,
            values.dtype,
            mapping.region_offset,
            mapping.storage_elements,
            mapping.stride,
            mapping.storage_offset,
            mapping.region_offset,
            mapping.region_elements,
        )
        self.values = values

    def tensor_chunks(self):
        yield self.chunk

    def export_tensors(self, callback, *, chunk_bytes):
        for offset in range(
            0, self.values.numel(), chunk_bytes // self.values.element_size()
        ):
            part = self.values[
                offset : offset + chunk_bytes // self.values.element_size()
            ]
            callback(
                replace(self.chunk, offset=self.chunk.region_offset + offset), part
            )


def open_sources(stack, tmp_path, mappings, logical, *, change=None):
    result = []
    for rank, mapping in enumerate(mappings):
        raw = torch.zeros(mapping.storage_elements, dtype=logical.dtype)
        local = logical[
            tuple(
                slice(start, start + size)
                for start, size in zip(mapping.origin, mapping.local_shape, strict=True)
            )
        ]
        raw.as_strided(
            mapping.local_shape, mapping.stride, mapping.storage_offset
        ).copy_(local)
        owned = raw[
            mapping.region_offset : mapping.region_offset + mapping.region_elements
        ].clone()
        if change is not None:
            change(rank, owned)
        state = StoredView(mapping, owned, rank)
        path = tmp_path / f"rank-{rank}.ssp"
        save_checkpoint(state, path, chunk_bytes=32)
        reader = stack.enter_context(read_rank(state, path, chunk_bytes=32))
        result.append(SourceView(reader, reader.chunks["state"], mapping))
    return result


def outputs(mappings, dtype):
    values, targets = [], []
    for mapping in mappings:
        destination = torch.full((mapping.region_elements,), -100, dtype=dtype)

        def write(offset, chunk, destination=destination):
            destination[offset : offset + chunk.numel()].copy_(chunk)

        values.append(destination)
        targets.append(TargetView(mapping, dtype, write))
    return values, targets


@pytest.mark.parametrize("source_count,target_count", [(1, 4), (4, 1), (2, 4), (4, 2)])
def test_archives_transfer_rows_to_columns_for_different_worker_counts(
    tmp_path, source_count, target_count
):
    logical = torch.arange(48, dtype=torch.float32).reshape(4, 12)
    source_maps = [
        view(
            (4, 12),
            local=(4 // source_count, 12),
            origin=(index * 4 // source_count, 0),
        )
        for index in range(source_count)
    ]
    target_maps = [
        view(
            (4, 12),
            local=(4, 12 // target_count),
            origin=(0, index * 12 // target_count),
        )
        for index in range(target_count)
    ]
    values, targets = outputs(target_maps, logical.dtype)
    with ExitStack() as stack:
        sources = open_sources(stack, tmp_path, source_maps, logical)
        transfer_tensor(sources, targets, scratch_bytes=60)
    for mapping, value in zip(target_maps, values, strict=True):
        start = mapping.origin[1]
        torch.testing.assert_close(
            value.reshape(mapping.local_shape),
            logical[:, start : start + mapping.local_shape[1]],
            rtol=0,
            atol=0,
        )


def test_flat_owned_regions_transpose_and_replicas_preserve_every_stored_value(
    tmp_path,
):
    logical = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    source_maps = [view((3, 5), region=region) for region in ((0, 8), (8, 7), (0, 15))]
    target_maps = [
        view((3, 5), stride=(1, 3)),
        view((3, 5), region=(0, 7)),
        view((3, 5), region=(7, 8)),
    ]
    values, targets = outputs(target_maps, logical.dtype)
    with ExitStack() as stack:
        transfer_tensor(
            open_sources(stack, tmp_path, source_maps, logical),
            targets,
            scratch_bytes=36,
        )
    torch.testing.assert_close(values[0].as_strided((3, 5), (1, 3)), logical)
    torch.testing.assert_close(torch.cat(values[1:]), logical.reshape(-1))


def test_replica_conflict_fails_without_publishing_partial_conversion(tmp_path):
    logical = torch.arange(20, dtype=torch.float32)
    maps = [view((20,)), view((20,))]
    values, targets = outputs([maps[0]], logical.dtype)

    def change(rank, values):
        if rank == 1:
            values[-1] += 1

    with ExitStack() as stack:
        sources = open_sources(stack, tmp_path, maps, logical, change=change)
        with pytest.raises(ValueError, match="replicas disagree"):
            transfer_tensor(sources, targets, scratch_bytes=48)
    # Earlier ranges may have been written; the unpublished generation must be
    # discarded. The conflicting range is never written or silently averaged.
    assert values[0][-1] == -100


def test_missing_logical_coverage_fails_before_any_destination_write(tmp_path):
    logical = torch.arange(8, dtype=torch.float32)
    maps = [view((8,), region=(0, 7))]
    values, targets = outputs([view((8,))], logical.dtype)
    with ExitStack() as stack:
        sources = open_sources(stack, tmp_path, maps, logical)
        with pytest.raises(ValueError, match="gap at element 7"):
            transfer_tensor(sources, targets, scratch_bytes=48)
    assert (values[0] == -100).all()


@pytest.mark.parametrize("target_dtype", [torch.bfloat16, torch.float32])
def test_explicit_weight_cast_uses_one_bounded_comparison_and_output_buffer(
    tmp_path, monkeypatch, target_dtype
):
    logical = torch.linspace(-3, 3, 1003, dtype=torch.float64)
    maps = [view((1003,)), view((1003,))]
    values, targets = outputs([maps[0]], target_dtype)
    allocations = []
    with ExitStack() as stack:
        sources = open_sources(stack, tmp_path, maps, logical)
        for name in ("empty", "empty_like"):
            original = getattr(torch, name)

            def allocate(*a, original=original, **k):
                result = original(*a, **k)
                allocations.append(result.nbytes)
                return result

            monkeypatch.setattr(torch, name, allocate)
        with pytest.raises(ValueError, match="explicit transformation"):
            transfer_tensor(sources, targets, scratch_bytes=100)
        transfer_tensor(
            sources, targets, scratch_bytes=100, transformation="scalar_cast"
        )
    assert len(allocations) == 3 and sum(allocations) <= 100
    torch.testing.assert_close(values[0], logical.to(target_dtype), rtol=0, atol=0)


def test_scalar_optimizer_counter_replication_is_not_inferred_from_weight_shape(
    tmp_path,
):
    values, targets = outputs([view(())] * 4, torch.int64)
    with ExitStack() as stack:
        sources = open_sources(
            stack, tmp_path, [view(())] * 2, torch.tensor(173, dtype=torch.int64)
        )
        transfer_tensor(sources, targets, scratch_bytes=24)
    assert all(value.item() == 173 for value in values)


def test_restored_weight_and_moments_support_the_next_adam_update(tmp_path):
    weight = torch.linspace(-1, 2, 48).reshape(6, 8)
    first = torch.linspace(-0.3, 0.2, 48).reshape(6, 8)
    second = torch.linspace(0.01, 0.5, 48).reshape(6, 8)
    gradient = torch.cos(torch.arange(48).float()).reshape(6, 8)
    source_maps = [view((6, 8), local=(3, 8), origin=(row, 0)) for row in (0, 3)]
    target_maps = [
        view((6, 8), local=(6, 2), origin=(0, column)) for column in (0, 2, 4, 6)
    ]
    converted = []
    for index, tensor in enumerate((weight, first, second)):
        directory = tmp_path / str(index)
        directory.mkdir()
        values, targets = outputs(target_maps, tensor.dtype)
        with ExitStack() as stack:
            sources = open_sources(stack, directory, source_maps, tensor)
            transfer_tensor(sources, targets, scratch_bytes=60)
        converted.append(values)

    def update(parameter, moment, variance, grad):
        parameter = torch.nn.Parameter(parameter.clone())
        optimizer = torch.optim.Adam(
            [parameter], lr=0.03, betas=(0.8, 0.95), eps=1e-7, foreach=False
        )
        optimizer.state[parameter] = {
            "step": torch.tensor(7.0),
            "exp_avg": moment.clone(),
            "exp_avg_sq": variance.clone(),
        }
        parameter.grad = grad.clone()
        optimizer.step()
        return parameter.detach(), optimizer.state[parameter]

    expected, expected_state = update(weight, first, second, gradient)
    for index, mapping in enumerate(target_maps):
        column = mapping.origin[1]
        tensors = [values[index].reshape(6, 2) for values in converted]
        actual, state = update(*tensors, gradient[:, column : column + 2])
        torch.testing.assert_close(
            actual, expected[:, column : column + 2], rtol=0, atol=0
        )
        torch.testing.assert_close(
            state["exp_avg"],
            expected_state["exp_avg"][:, column : column + 2],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            state["exp_avg_sq"],
            expected_state["exp_avg_sq"][:, column : column + 2],
            rtol=0,
            atol=0,
        )
        assert state["step"].item() == 8

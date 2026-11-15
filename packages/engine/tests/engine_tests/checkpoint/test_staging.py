"""Unpublished target files preserve ties and bound payload scratch and disk."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from engine_tests.checkpoint.test_mapping import view
from engine_tests.execution.test_checkpoint import MemoryExecutor
from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.checkpoint import load_checkpoint, save_checkpoint
from shadowspill.checkpoint.staging import stage_tensors, staging_bytes


def descriptor(name="weight", count=19, dtype=torch.float32):
    return TensorChunk(name, (count,), dtype, 0, count, (1,))


def test_partial_tied_writes_and_storage_holes_feed_the_existing_encoder(tmp_path):
    chunk = descriptor()
    chunks = {(0, chunk.name): chunk}
    target = tmp_path / "published.ssp"
    with stage_tensors(
        chunks, tmp_path, scratch_bytes=25, max_disk_bytes=95
    ) as staging:
        staging.write((0, "weight"), 2, torch.arange(2, 8, dtype=torch.float32))
        staging.write((0, "weight"), 0, torch.arange(11, dtype=torch.float32))
        staging.write((0, "weight"), 8, torch.arange(8, 14, dtype=torch.float32))
        staging.require_written((0, "weight"), view((14,), storage=19))
        staging.seal()
        # The encoder receives its ordinary descriptor/export interface. The
        # remaining physical holes are deterministic zeros, not logical state.
        source = SimpleNamespace(
            plan=SimpleNamespace(program=SimpleNamespace(digest="staged")),
            plans={},
            logical_rank=0,
            tensor_chunks=lambda: iter((chunk,)),
            export_tensors=lambda callback, *, chunk_bytes: staging.export(
                0, callback, chunk_bytes=chunk_bytes
            ),
        )
        save_checkpoint(source, target, chunk_bytes=12)
        with pytest.raises(ValueError, match="sealed"):
            staging.write((0, "weight"), 0, torch.ones(1))
    assert list(tmp_path.iterdir()) == [target]
    destination = MemoryExecutor()
    destination.values = {"weight": torch.full((19,), -1.0)}
    load_checkpoint(destination, target, chunk_bytes=16)
    expected = torch.zeros(19)
    expected[:14] = torch.arange(14)
    torch.testing.assert_close(destination.values["weight"], expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="closed"):
        staging.export(0, lambda *_: None, chunk_bytes=16)


@pytest.mark.parametrize("second_offset,second_count", [(0, 5), (3, 8), (8, 8)])
def test_conflicting_full_and_partial_alias_writes_invalidate_all_output(
    tmp_path, second_offset, second_count
):
    chunks = {(0, "weight"): descriptor(count=16)}
    with stage_tensors(
        chunks, tmp_path, scratch_bytes=25, max_disk_bytes=80
    ) as staging:
        staging.write((0, "weight"), 3, torch.full((8,), 2.0))
        with pytest.raises(ValueError, match="conflicting values"):
            staging.write(
                (0, "weight"), second_offset, torch.full((second_count,), 3.0)
            )
        with pytest.raises(ValueError, match="invalid"):
            staging.seal()
        with pytest.raises(ValueError, match="invalid"):
            staging.export(
                0, lambda *_: pytest.fail("published failed staging"), chunk_bytes=16
            )
    assert not list(tmp_path.iterdir())


def test_missing_logical_values_fail_but_owned_storage_padding_is_allowed(tmp_path):
    chunk = descriptor(count=20)
    with stage_tensors(
        {(0, "weight"): chunk}, tmp_path, scratch_bytes=30, max_disk_bytes=100
    ) as staging:
        staging.write((0, "weight"), 0, torch.arange(6, dtype=torch.float32))
        staging.require_written((0, "weight"), view((6,), storage=20))
        with pytest.raises(ValueError, match="unwritten logical"):
            staging.require_written((0, "weight"), view((7,), storage=20))
        with pytest.raises(ValueError, match="invalid"):
            staging.seal()


def test_scratch_is_shared_across_ranks_dtypes_and_exports(tmp_path, monkeypatch):
    chunks = {
        (rank, "weight"): descriptor(count=1003, dtype=dtype)
        for rank, dtype in enumerate((torch.float64, torch.bfloat16, torch.int64))
    }
    original = torch.empty
    allocations = []

    def allocate(*args, **kwargs):
        result = original(*args, **kwargs)
        allocations.append(result.nbytes)
        return result

    inputs = {key: torch.arange(1003).to(chunk.dtype) for key, chunk in chunks.items()}
    monkeypatch.setattr(torch, "empty", allocate)
    with stage_tensors(
        chunks, tmp_path, scratch_bytes=99, max_disk_bytes=staging_bytes(chunks)
    ) as staging:
        assert sum(allocations) == 99
        for key, value in inputs.items():
            staging.write(key, 0, value)
            staging.write(key, 0, value)
            staging.require_written(key, view((1003,)))
        staging.seal()
        for key in chunks:

            def check(request, value, key=key):
                assert value.nbytes <= 17
                torch.testing.assert_close(
                    value,
                    inputs[key][request.offset : request.offset + value.numel()],
                    rtol=0,
                    atol=0,
                )

            staging.export(key[0], check, chunk_bytes=17)
        assert len(allocations) == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("scratch,disk", [(4, 95), (25, 94), (True, 95), (25, False)])
def test_budget_rejection_creates_no_directory_or_tensor(
    tmp_path, monkeypatch, scratch, disk
):
    monkeypatch.setattr(
        torch, "empty", lambda *_a, **_k: pytest.fail("allocated before admission")
    )
    with (
        pytest.raises((ValueError, TypeError)),
        stage_tensors(
            {(0, "weight"): descriptor()},
            tmp_path,
            scratch_bytes=scratch,
            max_disk_bytes=disk,
        ),
    ):
        pytest.fail("invalid reservation admitted")
    assert not list(tmp_path.iterdir())


def test_file_write_failure_removes_only_owned_temporary_files(tmp_path, monkeypatch):
    unrelated = tmp_path / "unrelated"
    unrelated.write_text("keep")
    chunks = {(0, "weight"): descriptor()}
    with stage_tensors(
        chunks, tmp_path, scratch_bytes=25, max_disk_bytes=95
    ) as staging:
        original = Path.open

        def fail(path, *args, **kwargs):
            if path.suffix == ".data" and args[0] == "r+b":
                raise OSError("injected disk error")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", fail)
        with pytest.raises(OSError, match="disk error"):
            staging.write((0, "weight"), 0, torch.ones(1))
        with pytest.raises(ValueError, match="invalid"):
            staging.seal()
    assert list(tmp_path.iterdir()) == [unrelated]
    assert unrelated.read_text() == "keep"


def test_original_region_offsets_and_empty_tensors_are_exported_exactly(tmp_path):
    chunk = replace(descriptor(count=15), offset=8, region_offset=8, region_elements=7)
    empty = descriptor(name="empty", count=0)
    chunks = {(2, "weight"): chunk, (2, "empty"): empty}
    with stage_tensors(
        chunks, tmp_path, scratch_bytes=25, max_disk_bytes=35
    ) as staging:
        staging.write((2, "weight"), 0, torch.arange(8, 15, dtype=torch.float32))
        staging.require_written((2, "weight"), view((15,), region=(8, 7)))
        staging.require_written((2, "empty"), view((0,)))
        staging.seal()
        records = []
        staging.export(
            2,
            lambda request, values: records.append((request, values.clone())),
            chunk_bytes=9,
        )
        assert all(
            request.name == "weight" and 8 <= request.offset < 15
            for request, _ in records
        )
        assert all(
            request.region_offset == 8 and request.region_elements == 7
            for request, _ in records
        )
        torch.testing.assert_close(
            torch.cat([values for _, values in records]),
            torch.arange(8, 15, dtype=torch.float32),
        )


def test_staging_export_requires_owner_to_seal_after_logical_validation(tmp_path):
    with (
        stage_tensors(
            {(0, "weight"): descriptor()}, tmp_path, scratch_bytes=25, max_disk_bytes=95
        ) as staging,
        pytest.raises(ValueError, match="sealed before export"),
    ):
        staging.export(0, lambda *_: pytest.fail("premature export"), chunk_bytes=16)

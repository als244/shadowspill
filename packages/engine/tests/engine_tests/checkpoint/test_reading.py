"""Random checkpoint reads reuse complete archive validation and bounded hashing."""

import tarfile
from dataclasses import replace

import pytest
import torch

from engine_tests.execution.test_checkpoint import MemoryExecutor
from shadowspill.checkpoint import save_checkpoint
from shadowspill.checkpoint.reading import read_rank


def saved(tmp_path):
    state = MemoryExecutor()
    path = tmp_path / "source.ssp"
    save_checkpoint(state, path, metadata={"updates": 7}, chunk_bytes=16)
    return state, path


def test_verified_random_reads_match_strided_storage_and_preserve_exact_dtypes(
    tmp_path, monkeypatch
):
    state, path = saved(tmp_path)
    empty = torch.empty
    allocated = []

    def allocate(*args, **kwargs):
        result = empty(*args, **kwargs)
        allocated.append(result.nbytes)
        return result

    monkeypatch.setattr(torch, "empty", allocate)
    with read_rank(state, path, chunk_bytes=17) as source:
        assert allocated == [17]
        assert source.manifest["metadata"]["updates"] == 7
        with pytest.raises(TypeError):
            source.manifest["metadata"]["updates"] = 9
        for name in reversed(source.chunks):
            chunk = source.chunks[name]
            value = state.values[name]
            raw = value.as_strided((chunk.storage_elements,), (1,), 0)
            width = max(1, 17 // value.element_size())
            for start in reversed(range(0, chunk.storage_elements, width)):
                output = torch.empty(
                    min(width, chunk.storage_elements - start), dtype=value.dtype
                )
                source.readinto(replace(chunk, offset=start), output)
                # uint8 comparison includes fp8 and scalar state, with no cast.
                torch.testing.assert_close(
                    output.view(torch.uint8),
                    raw[start : start + output.numel()].view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
    assert max(allocated) <= 17
    assert source.closed
    with pytest.raises(ValueError, match="closed"):
        source.readinto(chunk, output)


def test_corruption_in_a_never_requested_tensor_fails_before_yield(tmp_path):
    state, path = saved(tmp_path)
    with tarfile.open(path) as archive:
        location = archive.getmember("tensors/2").offset_data
    with path.open("r+b") as stream:
        stream.seek(location)
        stream.write(b"\xff")
    with (
        pytest.raises(ValueError, match="checksum differs"),
        read_rank(state, path, chunk_bytes=16),
    ):
        pytest.fail("corrupt source became readable")


@pytest.mark.parametrize("budget", [0, -1, True, 3.0, 7])
def test_invalid_or_sub_element_budget_is_rejected(tmp_path, budget):
    state, path = saved(tmp_path)
    with (
        pytest.raises(ValueError, match=r"positive integer|one element"),
        read_rank(state, path, chunk_bytes=budget),
    ):
        pytest.fail("invalid budget accepted")


@pytest.mark.parametrize(
    "change", ["before", "after", "dtype", "geometry", "unknown", "budget"]
)
def test_invalid_subregion_is_rejected_before_mutating_destination(tmp_path, change):
    state, path = saved(tmp_path)
    with read_rank(state, path, chunk_bytes=16) as source:
        chunk = source.chunks["bf16"]
        output = torch.full((2,), -3, dtype=chunk.dtype)
        if change == "before":
            chunk = replace(chunk, offset=-1)
        elif change == "after":
            chunk = replace(chunk, offset=chunk.storage_elements - 1)
        elif change == "dtype":
            output = output.float()
        elif change == "geometry":
            chunk = replace(chunk, shape=(chunk.storage_elements,))
        elif change == "unknown":
            chunk = replace(chunk, name="absent")
        else:
            output = torch.full((9,), -3, dtype=chunk.dtype)
        with pytest.raises(ValueError):
            source.readinto(chunk, output)
        assert (output == -3).all()


def test_changed_open_file_invalidates_conversion_before_context_success(tmp_path):
    state, path = saved(tmp_path)
    with (
        pytest.raises(ValueError, match="changed during conversion"),
        read_rank(state, path, chunk_bytes=16) as source,
        path.open("r+b") as stream,
    ):
        stream.seek(512)
        stream.write(b"\xff")
    assert source.closed


def test_reader_closes_after_owner_failure_and_does_not_mask_it(tmp_path):
    state, path = saved(tmp_path)
    with (
        pytest.raises(RuntimeError, match="publication failed"),
        read_rank(state, path, chunk_bytes=16) as source,
    ):
        raise RuntimeError("publication failed")
    assert source.closed

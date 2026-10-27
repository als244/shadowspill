"""Bounded backend I/O keeps layouts, padding and byte representations intact."""

import pytest
import torch

from shadowspill.execution.tensor_spans import spans
from tests.shadowspill.execution._spill import MemorySpill


def test_chunks_preserve_neighbors_and_abort_unfinished_write():
    owner = torch.arange(33, dtype=torch.int16)
    spill = MemorySpill({"a": owner})
    for _, chunk in spill.chunks(
        "a",
        dtype=owner.dtype,
        start=3,
        length=19,
        chunk_bytes=10,
        read=False,
        write=True,
    ):
        chunk.fill_(7)
    expected = torch.arange(33, dtype=owner.dtype)
    expected[3:22] = 7
    torch.testing.assert_close(owner, expected)
    assert max(size for _, _, size, _ in spill.transfers) <= 10
    with (
        pytest.raises(RuntimeError, match="loader failed"),
        spill.buffer("a", 0, 8, read=False, write=True) as chunk,
    ):
        chunk.zero_()
        raise RuntimeError("loader failed")
    torch.testing.assert_close(owner, expected)
    with (
        pytest.raises(ValueError, match="outside"),
        spill.buffer("a", owner.nbytes - 1, 2),
    ):
        pytest.fail("range accepted")
    with pytest.raises(ValueError, match="one tensor element"):
        list(spill.chunks("a", dtype=owner.dtype, chunk_bytes=1))
    for start, length in ((-1, 1), (0, -1), (owner.numel() + 1, 0)):
        with pytest.raises(ValueError, match="outside"):
            list(spill.chunks("a", dtype=owner.dtype, start=start, length=length))


@pytest.mark.parametrize(
    "shape,stride,offset",
    [
        ((), (), 0),
        ((2, 3), (3, 1), 1),
        ((3, 2), (1, 3), 0),
        ((2, 2), (7, 2), 1),
        ((2, 0), (3, 1), 0),
        ((2, 3), (0, 1), 0),
    ],
)
def test_spans_match_arbitrary_pytorch_view(shape, stride, offset):
    storage = torch.arange(64)
    reference = storage.as_strided(shape, stride, offset).reshape(-1)
    result = torch.empty_like(reference)
    for logical, start, count in spans(shape, stride, offset):
        result[logical : logical + count].copy_(storage[start : start + count])
    torch.testing.assert_close(result, reference)


@pytest.mark.parametrize("case", ["capacity", "backend"])
def test_invalid_spill_rejected_before_runtime_setup(tmp_path, monkeypatch, case):
    from types import SimpleNamespace

    from shadowspill.execution import setup
    from shadowspill.ssd import SSDPool

    plan = SimpleNamespace(
        simulation_only_reasons=(),
        program=SimpleNamespace(
            devices=(0,), setup={"torch": torch.__version__}, bindings={}
        ),
        memory_budgets=SimpleNamespace(spill_bytes=64 << 20),
        communication_info=SimpleNamespace(metadata={}),
    )
    monkeypatch.setattr(setup, "require_compiled_device", lambda *args: None)
    monkeypatch.setattr(setup, "Runtime", lambda **kw: pytest.fail("opened a pool"))
    capacity, error, message = 64 << 20, ValueError, "capacity"
    if case == "capacity":
        capacity *= 2
    elif case == "backend":
        plan.communication_info.metadata["spill_kind"] = "remote"
        message = "different spill backend"
    with pytest.raises(error, match=message):
        setup.open_worker(plan, 0, 0, SSDPool(capacity, directory=tmp_path))

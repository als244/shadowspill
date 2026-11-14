import pytest
import torch

from shadowspill.profiling.observation.native import observe_allocations


@pytest.mark.cuda
def test_native_allocator_reports_unrounded_requests_and_frees():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    torch.cuda.init()
    before = torch.cuda.memory.get_allocator_backend()
    with observe_allocations(0) as events:
        tensor = torch.empty(37, device="cuda", dtype=torch.uint8)
        pointer = tensor.data_ptr()
        del tensor
    relevant = [
        event
        for event in events
        if event["address"] == pointer
        and event["action"] in {"allocate", "free_requested", "free_completed"}
    ]
    assert [event["action"] for event in relevant] == [
        "allocate",
        "free_requested",
        "free_completed",
    ]
    assert relevant[0]["requested_bytes"] == 37
    assert relevant[0]["pool_first"] == relevant[0]["pool_second"] == 0
    assert torch.cuda.memory.get_allocator_backend() == before == "native"


@pytest.mark.cuda
def test_overflow_is_explicit_and_observer_can_restart():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    torch.cuda.init()
    with (
        pytest.raises(RuntimeError, match="overflowed"),
        observe_allocations(0, capacity=1),
    ):
        value = torch.empty(7, device="cuda")
        del value
    with observe_allocations(0) as events:
        value = torch.empty(9, device="cuda")
        del value
    assert any(event["action"] == "allocate" for event in events)

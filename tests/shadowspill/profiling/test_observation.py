import pytest
import torch

from shadowspill.profiling.observation.observe import observe_task
from shadowspill.task.allocations import TaskAllocationOperation


@pytest.mark.cuda
def test_views_mutations_and_temporary_lifetimes():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    value = torch.zeros(17, device="cuda")

    def task(x):
        temporary = torch.empty(37, device="cuda", dtype=torch.uint8)
        temporary.fill_(3)
        del temporary
        result = torch.empty(19, device="cuda")
        result.fill_(2)
        x.add_(1)
        return x[3:], result[2:7], result[19:]

    observed = observe_task(task, (value,), device=torch.device("cuda", 0))
    assert observed.persistent_bytes == ()
    assert observed.workspace_requested_bytes == 37
    assert observed.workspace_bytes == 37
    assert len(observed.input_bindings) == 1
    binding = observed.input_bindings[0]
    assert (binding.output_leaf_index, binding.input_position) == (0, 0)
    assert binding.output_offset_bytes == 12
    allocations = [
        event
        for event in observed.trace
        if event.operation is TaskAllocationOperation.ALLOCATE
    ]
    assert [event.requested_bytes for event in allocations] == [37, 76]
    assert allocations[1].output_leaf_indices == (1, 2)
    assert allocations[1].output_view_offsets == (8, 76)
    assert all(event.reuses_ordinal is None for event in observed.trace)
    torch.testing.assert_close(value, torch.ones_like(value))


@pytest.mark.cuda
def test_retained_anonymous_memory_is_explicit():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    retained = []

    def task():
        retained.append(torch.empty(37, device="cuda", dtype=torch.uint8))
        return torch.empty(5, device="cuda")

    observed = observe_task(task, (), device=torch.device("cuda", 0))
    assert observed.persistent_bytes == (37,)
    retained.clear()


@pytest.mark.cuda
def test_joined_private_stream_work_retains_correct_lifetimes():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    side = torch.cuda.Stream()
    current = torch.cuda.current_stream()

    def task():
        temporary = torch.empty(37, device="cuda", dtype=torch.uint8)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            temporary.fill_(5)
        current.wait_stream(side)
        result = temporary.clone()
        del temporary
        return result

    observed = observe_task(task, (), device=torch.device("cuda", 0))
    assert observed.persistent_bytes == ()
    assert observed.workspace_requested_bytes == 37
    assert observed.workspace_bytes == 37

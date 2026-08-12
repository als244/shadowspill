from __future__ import annotations

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.pytorch.compiled_layout import reconcile_compiled_task_layout
from shadowspill.pytorch.contracts import CaptureError
from shadowspill.pytorch.output_contract import capture_task_storage_contract
from shadowspill.pytorch.profiling import (
    TaskAllocationEvent,
    TaskAllocationOperation,
    TaskMeasurement,
)


def _measurement(
    *events: TaskAllocationEvent,
    workspace: int = 0,
) -> TaskMeasurement:
    return TaskMeasurement(
        runtime_ns=100,
        workspace_requested_bytes=workspace,
        workspace_charged_bytes=workspace,
        workspace_extent_bytes=() if workspace == 0 else (workspace,),
        samples_ns=(100,),
        provenance="unit-test",
        allocation_trace=events,
    )


def test_repeated_output_reconciles_to_one_physical_allocation() -> None:
    def function(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = torch.sin(value)
        return result, result

    value = torch.randn(8)
    contract = capture_task_storage_contract(make_fx(function)(value), (value,))
    layout = reconcile_compiled_task_layout(
        contract,
        _measurement(
            TaskAllocationEvent(
                0,
                TaskAllocationOperation.ALLOCATE,
                32,
                32,
                (0, 1),
                (0, 0),
            )
        ),
    )

    assert len(layout.roots) == 1
    assert layout.roots[0].requested_bytes == 32
    assert tuple(view.allocation_ordinal for view in layout.output_views) == (0, 0)
    assert layout.to_json() == layout.to_json()


def test_input_passthrough_requires_no_output_allocation() -> None:
    def function(value: torch.Tensor) -> torch.Tensor:
        return value[2:6]

    value = torch.randn(8)
    contract = capture_task_storage_contract(make_fx(function)(value), (value,))
    layout = reconcile_compiled_task_layout(contract, _measurement())

    assert layout.roots[0].allocation_ordinal is None
    assert layout.output_views[0].offset_bytes == 8


def test_distinct_fresh_roots_reconcile_to_split_allocations() -> None:
    def function(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.sin(value), torch.cos(value)

    value = torch.randn(6528, dtype=torch.float32)
    contract = capture_task_storage_contract(make_fx(function)(value), (value,))
    layout = reconcile_compiled_task_layout(
        contract,
        _measurement(
            TaskAllocationEvent(
                0,
                TaskAllocationOperation.ALLOCATE,
                26_112,
                26_112,
                (0,),
                (0,),
            ),
            TaskAllocationEvent(
                1,
                TaskAllocationOperation.ALLOCATE,
                26_112,
                26_112,
                (1,),
                (0,),
            ),
        ),
    )

    assert tuple(root.requested_bytes for root in layout.roots) == (26_112, 26_112)
    assert tuple(root.allocation_ordinal for root in layout.roots) == (0, 1)


def test_compact_compiled_view_may_be_smaller_than_unseen_intermediate() -> None:
    def function(value: torch.Tensor) -> torch.Tensor:
        intermediate = torch.sin(value)
        return intermediate[9984:]

    value = torch.randn(19_968, dtype=torch.float32)
    contract = capture_task_storage_contract(make_fx(function)(value), (value,))
    layout = reconcile_compiled_task_layout(
        contract,
        _measurement(
            TaskAllocationEvent(
                0,
                TaskAllocationOperation.ALLOCATE,
                39_936,
                39_936,
                (0,),
                (0,),
            )
        ),
    )

    assert contract.roots[0].minimum_span_bytes == 39_936
    assert layout.roots[0].requested_bytes == 39_936
    assert layout.output_views[0].offset_bytes == 0


def test_physical_observation_cannot_merge_semantic_roots() -> None:
    def function(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.sin(value), torch.cos(value)

    value = torch.randn(8)
    contract = capture_task_storage_contract(make_fx(function)(value), (value,))

    with pytest.raises(CaptureError, match="distinct fresh roots"):
        reconcile_compiled_task_layout(
            contract,
            _measurement(
                TaskAllocationEvent(
                    0,
                    TaskAllocationOperation.ALLOCATE,
                    64,
                    64,
                    (0, 1),
                    (0, 32),
                )
            ),
        )


def test_physical_observation_cannot_split_one_semantic_root() -> None:
    def function(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        intermediate = torch.sin(value)
        return intermediate[:4], intermediate[4:]

    value = torch.randn(8)
    contract = capture_task_storage_contract(make_fx(function)(value), (value,))

    with pytest.raises(CaptureError, match="spans compiled allocations"):
        reconcile_compiled_task_layout(
            contract,
            _measurement(
                TaskAllocationEvent(
                    0,
                    TaskAllocationOperation.ALLOCATE,
                    16,
                    16,
                    (0,),
                    (0,),
                ),
                TaskAllocationEvent(
                    1,
                    TaskAllocationOperation.ALLOCATE,
                    16,
                    16,
                    (1,),
                    (0,),
                ),
            ),
        )

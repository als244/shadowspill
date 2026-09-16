"""Small readers and checks shared by the geometry it validates."""

from __future__ import annotations

from typing import Any

from torch._inductor.graph import GraphLowering

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.storage import (
    OutputView,
    StorageRoot,
)


def _static_int(graph: GraphLowering, value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            sizevars: Any = graph.sizevars
            hinted = sizevars.size_hint(value)
        except BaseException as exc:
            raise CaptureError(f"Inductor {field} is not fixed-shape: {value}") from exc
        return int(hinted)


def _span_bytes(shape: tuple[int, ...], stride: tuple[int, ...], item_size: int) -> int:
    if not shape or any(extent == 0 for extent in shape):
        return 0 if shape and any(extent == 0 for extent in shape) else item_size
    last_element = sum(
        (extent - 1) * step for extent, step in zip(shape, stride, strict=True)
    )
    return (1 + last_element) * item_size


def _copy_root(root: StorageRoot, root_id: int) -> StorageRoot:
    return StorageRoot(
        root_id,
        root.kind,
        root.source_input,
        root.producer_node,
        root.producer_target,
        root.producer_result,
        root.minimum_span_bytes,
    )


def _copy_view(view: OutputView, leaf_index: int, root_id: int) -> OutputView:
    return OutputView(
        leaf_index,
        root_id,
        view.offset_bytes,
        view.span_bytes,
        view.shape,
        view.stride,
        view.dtype,
        view.layout,
    )

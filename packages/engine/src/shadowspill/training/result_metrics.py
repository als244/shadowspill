"""Declared scalar reductions over the actual logical input contributors."""

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import BinaryIO, cast

from shadowspill_client._records import number
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.tensors import TensorPayload

from .inputs.frame_reader import FrameReader
from .inputs.values import numeric_values
from .outputs import OutputLayout
from .worker.output_frame import WorkerOutputFrame


def _scalar(reader: FrameReader, path: tuple[str, ...]) -> float | int:
    if reader.frame.shapes[path] != ():
        raise ValueError("declared scalar result has a non-scalar frame shape")
    raw = b"".join(bytes(chunk) for chunk in reader.chunks(path))
    tensor = TensorPayload(reader.frame.dtypes[path], (), raw)
    value = next(numeric_values(tensor)).item()
    if type(value) is bool:
        value = int(value)
    number(value, "published scalar")
    return cast(float | int, value)


def result_metrics(
    layout: OutputLayout,
    frames: Mapping[int, WorkerOutputFrame],
    open_rank: Callable[[int], AbstractContextManager[BinaryIO]],
    memory: BufferBudget,
) -> tuple[float | int | None, dict[str, float | int]]:
    """Sum reported loss contributions; apply each named metric's own reduction.

    Only InputPlacement's contributors are read. Replicated worker outputs are
    counted once. The loss is separate from named metrics, so custom names such
    as ``loss`` never overwrite the objective value. No implicit mean is added.
    """
    metadata = 4096 + sum(
        512 + 8 * len(name.encode()) for name in layout.outputs.metrics
    )
    with memory.reserve(metadata, timeout=0):
        metrics: dict[str, float | int] = {}
        loss: float | int | None = None
        for buffer in range(len(layout.schedule.buffers)):
            for rank in layout.metric_workers(buffer):
                frame = frames[rank]
                index = frame.buffers.index(buffer)
                with (
                    open_rank(rank) as stream,
                    FrameReader(stream, frame.frame, memory) as reader,
                ):
                    path = ("inputs", str(index), "loss")
                    if path in frame.frame.shapes:
                        value = _scalar(reader, path)
                        loss = value if loss is None else loss + value
                    for name, reduction in layout.outputs.metrics.items():
                        value = _scalar(reader, ("inputs", str(index), "values", name))
                        if name not in metrics:
                            metrics[name] = value
                        elif reduction == "sum":
                            metrics[name] += value
                        elif reduction == "min":
                            metrics[name] = min(metrics[name], value)
                        elif reduction == "max":
                            metrics[name] = max(metrics[name], value)
                        elif metrics[name] != value:
                            raise ValueError(
                                "declared identical scalar metrics disagree"
                            )
        if loss is not None:
            number(loss, "published loss")
        for value in metrics.values():
            number(value, "published metric")
        return loss, metrics

"""Scalar reporting respects input ownership and each declared reduction."""

import io
from types import SimpleNamespace

import numpy as np
import pytest

from engine_tests.training.test_packing import schema, shape
from shadowspill.parallelism import GroupRef, InputPlacement
from shadowspill.training.inputs.frames import InputFrame
from shadowspill.training.inputs.shapes import PackingSchedule
from shadowspill.training.outputs import OutputLayout
from shadowspill.training.result_metrics import result_metrics
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.definitions import OutputSet


def configured(*, ownership="replicated", broken=None):
    schedule = PackingSchedule("two", [[shape(2), shape(2)]])
    placement = InputPlacement(
        GroupRef("pair", ((1, 0),)),
        **({"loss": "partitioned"} if ownership == "partitioned" else {}),
    )
    layout = OutputLayout(
        schema(),
        OutputSet(
            metrics={
                "sum": "sum",
                "min": "min",
                "max": "max",
                "identical": "identical",
                "loss": "sum",
                "valid": "sum",
                "large_integer": "sum",
            }
        ),
        schedule,
        placement,
    )
    frames, files = {}, {}
    for rank in (0, 1):
        fields = {}
        for buffer in (0, 1):
            value = (buffer + 1) * (10 if rank == 0 else 1)
            fields["inputs", str(buffer), "loss"] = np.asarray(value * 0.5, np.float32)
            metrics = {
                "sum": value,
                "min": value,
                "max": value,
                "identical": 7,
                "loss": 100,
                "valid": True,
                "large_integer": 2**60 + 1,
            }
            if buffer and broken is not None:
                metrics["identical"] = broken
            fields.update(
                {
                    ("inputs", str(buffer), "values", name): np.asarray(x)
                    for name, x in metrics.items()
                }
            )
        frame = InputFrame(
            {path: x.shape for path, x in fields.items()},
            {path: x.dtype.name for path, x in fields.items()},
        )
        stream = io.BytesIO()
        frame.write(stream, fields, BufferBudget(4 << 20), "metrics")
        frames[rank] = SimpleNamespace(frame=frame, buffers=(0, 1))
        files[rank] = stream.getvalue()
    return layout, frames, lambda rank: io.BytesIO(files[rank])


@pytest.mark.parametrize("ownership", ["replicated", "partitioned"])
def test_scalar_metrics_count_only_declared_logical_contributors(ownership):
    layout, frames, source = configured(ownership=ownership)
    memory = BufferBudget(4 << 20, 10)
    loss, metrics = result_metrics(layout, frames, source, memory)
    both = ownership == "partitioned"
    assert loss == (16.5 if both else 1.5)
    assert metrics == {
        "sum": 33 if both else 3,
        "min": 1,
        "max": 20 if both else 2,
        "identical": 7,
        "loss": 400 if both else 200,
        "valid": 4 if both else 2,
        "large_integer": (4 if both else 2) * (2**60 + 1),
    }
    assert memory.used == (0, 0)


@pytest.mark.parametrize("broken", [8, float("nan"), float("inf")])
def test_inconsistent_or_nonfinite_metrics_fail_without_retaining_reader_scratch(
    broken,
):
    layout, frames, source = configured(broken=broken)
    memory = BufferBudget(4 << 20, 10)
    with pytest.raises(ValueError):
        result_metrics(layout, frames, source, memory)
    assert memory.used == (0, 0)

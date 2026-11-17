"""Logical token counts for optional client displays, before packing or copies."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..datum import Datum
from ..display import BatchStatistics


def _vector(value: Any, name: str) -> NDArray[Any]:
    device = getattr(value, "device", None)
    if device is not None and str(device) != "cpu":
        raise ValueError("text statistics require CPU inputs")
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def text_batch_statistics(batch: Sequence[Datum]) -> BatchStatistics:
    """Count model input positions and unmasked targets in an aligned text batch.

    Input-only forward calls have no supervised statistics. Masks count valid
    targets independently of their objective weights. No input arrays are copied
    into the returned statistics, and no packing, upload or service call occurs.
    """
    lengths, supervised = [], []
    has_targets = False
    for datum in batch:
        tokens = _vector(datum.inputs["tokens"], "tokens")
        lengths.append(len(tokens))
        if "target_tokens" not in datum.loss_inputs:
            if "loss_mask" in datum.loss_inputs:
                raise ValueError("a text loss mask requires aligned target tokens")
            supervised.append(0)
            continue
        has_targets = True
        targets = _vector(datum.loss_inputs["target_tokens"], "target_tokens")
        if len(targets) != len(tokens):
            raise ValueError("target tokens must align with model input positions")
        if "loss_mask" not in datum.loss_inputs:
            supervised.append(len(targets))
            continue
        mask = _vector(datum.loss_inputs["loss_mask"], "loss_mask")
        if mask.shape != targets.shape or not np.isin(mask, (False, True)).all():
            raise ValueError("loss mask must align with targets and contain booleans")
        supervised.append(int(np.count_nonzero(mask)))
    return BatchStatistics.from_lengths(
        lengths, supervised=supervised if has_targets else None
    )

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
    """Count the model positions and trained targets of a text batch.

    An example's length is the size of its ``inputs["tokens"]``. Its
    supervised length is the number of true entries in
    ``loss_inputs["loss_mask"]``, or of ``loss_inputs["target_tokens"]``
    without a mask, whatever the weights; an example without targets has
    none. No array is retained, and nothing is packed or sent to the service.

    Parameters
    ----------
    batch : sequence of Datum
        Examples with one-dimensional CPU fields, as `text_datum` makes them.

    Returns
    -------
    BatchStatistics
        The lengths, with ``supervised`` None when no example has targets.

    Raises
    ------
    KeyError
        An example has no ``inputs["tokens"]``.
    ValueError
        The batch is empty; a field is not on the CPU or not one-dimensional;
        a mask comes without targets; the targets or mask differ in length
        from the tokens; or a mask holds a value other than true or false.
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

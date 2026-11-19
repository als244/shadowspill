"""CPU text conveniences; arbitrary Datum fields are never shifted implicitly."""

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..datum import Datum
from .preparation import build_candidates
from .statistics import text_batch_statistics


def text_datum(tokens: Any, *, loss_mask: Any = None, weights: Any = None) -> Datum:
    """Make a next-token example from one token sequence.

    For ``L`` tokens, the example has ``L - 1`` model positions and targets:
    ``inputs["tokens"]`` is ``tokens[:-1]`` and
    ``loss_inputs["target_tokens"]`` is ``tokens[1:]``, both int64 of shape
    ``(L - 1,)``. ``loss_mask`` and ``weights`` align with the original
    tokens and drop their first entry the same way. Fields are views where no
    conversion is needed: an int64 NumPy ``tokens`` array is shared, not
    copied. Tokenization and chat formatting stay with the caller.

    Parameters
    ----------
    tokens : array_like of int
        Token IDs on the CPU: one-dimensional, at least two, each nonnegative
        and within the int64 range.
    loss_mask : array_like of bool, optional
        Whether each original token is a trained target, with the shape of
        ``tokens``. Stored as ``loss_inputs["loss_mask"]`` (bool). By default
        no mask is stored and every target is trained.
    weights : float or array_like of float, optional
        Weight of each original token's loss: a scalar for all tokens, or one
        value per token. Stored as ``loss_inputs["weights"]`` (float32), with no
        normalization; `text_batch` makes weights for a batch mean. By default
        no weights are stored.

    Returns
    -------
    Datum
        The shifted example.

    Raises
    ------
    ValueError
        ``tokens`` is not on the CPU, is not a one-dimensional integer
        sequence of at least two IDs, or holds an ID outside the nonnegative
        int64 range; ``loss_mask`` differs in shape from ``tokens`` or holds
        a value other than true or false; or ``weights`` is not finite or is
        neither a scalar nor shaped like ``tokens``.
    """
    device = getattr(tokens, "device", None)
    if device is not None and str(device) != "cpu":
        raise ValueError("text_datum requires CPU tokens")
    values = np.asarray(tokens)
    if values.ndim != 1 or values.size < 2 or values.dtype.kind not in "iu":
        raise ValueError(
            "tokens must be a one-dimensional integer sequence of length >= 2"
        )
    if (values < 0).any() or (values > np.iinfo(np.int64).max).any():
        raise ValueError("token IDs must fit nonnegative int64 values")
    values = values.astype(np.int64, copy=False)
    loss_inputs = {"target_tokens": values[1:]}
    if loss_mask is not None:
        mask = np.asarray(loss_mask)
        if mask.shape != values.shape or not np.isin(mask, (False, True)).all():
            raise ValueError(
                "loss_mask must align with original tokens and contain booleans"
            )
        loss_inputs["loss_mask"] = mask[1:].astype(np.bool_, copy=False)
    if weights is not None:
        weight = np.asarray(weights, dtype=np.float32)
        if not np.isfinite(weight).all():
            raise ValueError("weights must be finite")
        if weight.ndim == 0:
            weight = np.broadcast_to(weight, values.shape)
        if weight.shape != values.shape:
            raise ValueError("weights must be a scalar or align with original tokens")
        loss_inputs["weights"] = weight[1:]
    return Datum(inputs={"tokens": values[:-1]}, loss_inputs=loss_inputs)


def text_batch(
    sequences: Iterable[Any], *, loss_masks: Iterable[Any] | None = None
) -> list[Datum]:
    """Make next-token examples whose weights average the loss over the batch.

    Each sequence becomes a `text_datum` example, and every target gets
    ``loss_inputs["weights"]`` of ``1 / N`` (float32), where ``N`` counts the
    trained targets of the whole batch. An objective that sums weighted
    target losses, such as ``"cross_entropy"``, then gives the mean over every
    trained target, whatever the sequence lengths. Packing, microbatches and
    worker placement keep these weights; to split one update over several
    calls, build the whole update's batch once and submit slices of the list.

    Parameters
    ----------
    sequences : iterable of array_like of int
        Token sequences, each as ``tokens`` in `text_datum`.
    loss_masks : iterable of array_like of bool, optional
        One mask per sequence, each as ``loss_mask`` in `text_datum`. By
        default every target is trained.

    Returns
    -------
    list of Datum
        One example per sequence, in order.

    Raises
    ------
    ValueError
        A sequence or mask is invalid as in `text_datum`, the number of masks
        differs from the number of sequences, or the batch has no trained
        target, as when it is empty.
    """
    batch = (
        [text_datum(tokens) for tokens in sequences]
        if loss_masks is None
        else [
            text_datum(tokens, loss_mask=mask)
            for tokens, mask in zip(sequences, loss_masks, strict=True)
        ]
    )
    count = sum(
        datum.loss_inputs["target_tokens"].size
        if "loss_mask" not in datum.loss_inputs
        else int(np.count_nonzero(datum.loss_inputs["loss_mask"]))
        for datum in batch
    )
    if not count:
        raise ValueError("text_batch requires at least one unmasked target token")
    weight = np.asarray(1 / count, dtype=np.float32)
    return [
        Datum(
            inputs=datum.inputs,
            loss_inputs={
                **datum.loss_inputs,
                "weights": np.broadcast_to(
                    weight, datum.loss_inputs["target_tokens"].shape
                ),
            },
        )
        for datum in batch
    ]


__all__ = ["build_candidates", "text_batch", "text_batch_statistics", "text_datum"]

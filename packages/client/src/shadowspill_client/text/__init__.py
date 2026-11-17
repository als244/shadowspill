"""CPU text conveniences; arbitrary Datum fields are never shifted implicitly."""

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..datum import Datum
from .statistics import text_batch_statistics


def text_datum(tokens: Any, *, loss_mask: Any = None, weights: Any = None) -> Datum:
    """Shift a complete token sequence and original-token-aligned mask/weights.

    The result has L-1 model positions and targets. Explicit Datum construction
    instead takes already aligned fields. Tokenization and chat formatting stay
    with the application. Views are caller-owned until the submission snapshot.
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
    """Next-token examples weighted by the whole batch's supervised token count.

    Masks align with original tokens, just as in ``text_datum``. Masked targets
    and the first token of each sequence do not enter the denominator. Packing,
    microbatching and worker placement may subsequently divide this batch without
    changing its weights. For split accumulation, construct the whole update's
    batch once, then submit slices of the resulting list.

    Use ``text_datum`` for explicit weights or unnormalized objectives. This
    helper does not change the objective's additive reduction or normalize other
    objective terms.
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


__all__ = ["text_batch", "text_batch_statistics", "text_datum"]

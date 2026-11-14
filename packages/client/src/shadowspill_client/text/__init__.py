"""CPU text conveniences; arbitrary Datum fields are never shifted implicitly."""

from typing import Any

import numpy as np

from ..datum import Datum


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


__all__ = ["text_datum"]

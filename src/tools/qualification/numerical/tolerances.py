"""What a planned state must match, and how a failure is named.

The bounds are the same for both comparisons the gate makes: against the
compiled reference, and between a run and its own checkpoint replay.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# Room for the whole training state of every cell -- parameters, gradients
# and both optimizer moments -- plus the activations that spill, without
# asking a shared machine to reserve more than the measurement needs. The
# gate proves the real bound itself: it fails unless the measured spill peak
# fits the pool.
SPILL_BUDGET = 32 << 30
LOSS_RELATIVE_TOLERANCE = 0.01
LOSS_ABSOLUTE_TOLERANCE = 2e-5
MINIMUM_COSINE = 0.999
MAXIMUM_RELATIVE_L2 = 0.025
# An optimizer moment is an accumulator of small values, and the same step
# run under two plans is the same arithmetic in two reduction orders. That
# alone moves a second-moment estimate past the bound above while every
# weight agrees, so the moments get twice the room; the weights keep the
# bound, being what training produces.
MAXIMUM_RELATIVE_L2_OPTIMIZER = 0.05
MINIMUM_SIGN_AGREEMENT = 0.99


def state_half(key: str) -> str:
    """Which half of the training state a comparison key names."""
    parts = str(key).split("/")
    return parts[1] if len(parts) > 1 and parts[0] == "state" else "other"


def meets_tensor_tolerance(metric: Any, *, key: str = "") -> bool:
    bound = (
        MAXIMUM_RELATIVE_L2_OPTIMIZER
        if state_half(key) == "optimizer"
        else MAXIMUM_RELATIVE_L2
    )
    return bool(
        metric.cosine >= MINIMUM_COSINE
        and metric.relative_l2 <= bound
        and metric.sign_agreement >= MINIMUM_SIGN_AGREEMENT
    )


def failures_by_state(keys: Sequence[str]) -> dict[str, int]:
    """Count failing tensors by which half of the training state they are in.

    Weights disagreeing and optimizer moments disagreeing are different
    findings: the first says the step computed something else, the second is
    usually an accumulator whose small values are ill-conditioned for a
    relative comparison. An aggregate count cannot be read either way, so the
    split is reported even when one side is zero.
    """
    counts = {"model": 0, "optimizer": 0}
    for key in keys:
        half = state_half(key)
        counts[half] = counts.get(half, 0) + 1
    return counts


def state_split(keys: Sequence[str]) -> str:
    """Render the split for a message, always naming both halves."""
    counts = failures_by_state(keys)
    named = [f"model {counts['model']}", f"optimizer {counts['optimizer']}"]
    named.extend(
        f"{half} {count}"
        for half, count in counts.items()
        if half not in ("model", "optimizer")
    )
    return ", ".join(named)

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
# A weight that is still nothing but its optimizer steps -- every element of
# the reference within `steps` learning rates of zero, as a bias started at
# zero is -- has no scale of its own for a relative bound to measure against.
# Two runs that round differently can disagree on the sign of such an
# element's gradient in one step, which moves it by at most two learning
# rates; so such a weight is held to that absolute bound instead.
STEP_QUANTUM_DISAGREEMENTS = 2


def state_half(key: str) -> str:
    """Which half of the training state a comparison key names."""
    parts = str(key).split("/")
    return parts[1] if len(parts) > 1 and parts[0] == "state" else "other"


def meets_tensor_tolerance(
    metric: Any,
    *,
    key: str = "",
    step_size: float | None = None,
    steps: int = 0,
) -> bool:
    """Whether one tensor agrees with its reference.

    The relative bounds decide, except for a weight whose reference is still
    nothing but its optimizer steps: given the run's ``step_size`` and
    ``steps``, one whose every element lies within ``steps`` learning rates of
    zero passes when no element is further from the reference than
    :data:`STEP_QUANTUM_DISAGREEMENTS` learning rates.
    """

    half = state_half(key)
    bound = (
        MAXIMUM_RELATIVE_L2_OPTIMIZER if half == "optimizer" else MAXIMUM_RELATIVE_L2
    )
    if (
        metric.cosine >= MINIMUM_COSINE
        and metric.relative_l2 <= bound
        and metric.sign_agreement >= MINIMUM_SIGN_AGREEMENT
    ):
        return True
    return bool(
        step_size is not None
        and steps > 0
        and half == "model"
        and metric.reference_maximum_absolute <= steps * step_size
        and metric.maximum_absolute_error <= STEP_QUANTUM_DISAGREEMENTS * step_size
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

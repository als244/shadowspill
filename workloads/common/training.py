"""Training settings shared by every qualification workload.

The rate lives here rather than in each case so that the two arms of a
comparison cannot drift apart: the eager reference and the planned step read
the same number. It is applied per step, as ``hyperparams={"lr":
LEARNING_RATE}``, rather than baked into the optimizer -- a value read from a
plain number is fixed when the step is captured.
"""

from __future__ import annotations

import torch

#: The rate every workload trains at.
LEARNING_RATE = 3.0e-4


def optimizer_state_init(
    name: str, tensor: torch.Tensor, parameter: torch.nn.Parameter
) -> None:
    """Give one declared optimizer-state entry its starting value.

    Moments and step counters start at zero, which is what the AdamW family
    means by "no history yet". A master copy of the parameter is the exception,
    and the reason the caller has to say: it starts at the parameter, and
    zeroing it would erase the weights rather than the history.
    """

    if name == "master_parameter":
        tensor.copy_(parameter)
    else:
        tensor.zero_()

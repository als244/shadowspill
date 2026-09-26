"""Training settings shared by every qualification workload.

The rate lives here rather than in each case so that the two arms of a
comparison cannot drift apart: the eager reference and the planned step read
the same number. It is applied per step, as ``hyperparams={"lr":
LEARNING_RATE}``, rather than baked into the optimizer -- a value read from a
plain number is fixed when the step is captured.
"""

from __future__ import annotations

#: The rate every workload trains at.
LEARNING_RATE = 3.0e-4

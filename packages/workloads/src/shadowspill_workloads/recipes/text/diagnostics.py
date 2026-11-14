"""Optional host-side diagnostics for text training observations."""

import math
from statistics import fmean


def fixed_batch_learning_report(losses, *, minimum_relative_drop):
    """Compare the first token-mean loss with the last three on identical data."""
    if not 0 <= minimum_relative_drop < 1:
        raise ValueError("minimum_relative_drop must lie in [0, 1)")
    if len(losses) < 4:
        raise ValueError("fixed-batch learning check requires at least four updates")
    if not all(math.isfinite(loss) and loss >= 0 for loss in losses) or losses[0] <= 0:
        raise ValueError("fixed-batch losses must be finite and initially positive")
    final_mean = fmean(losses[-3:])
    relative_drop = (losses[0] - final_mean) / losses[0]
    return {
        "global_mean_losses": list(losses),
        "initial_loss": losses[0],
        "final_three_mean_loss": final_mean,
        "relative_drop": relative_drop,
        "minimum_relative_drop": minimum_relative_drop,
        "passed": relative_drop >= minimum_relative_drop,
    }

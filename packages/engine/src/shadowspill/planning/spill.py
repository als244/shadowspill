"""Size a worker's spill allocation independently of its planning ceiling."""

from shadowspill.memory import PinnedHostPool, SpillPool


def spill_configuration(
    required_bytes: int, budget_bytes: int, supplied: SpillPool | None
) -> SpillPool:
    """Keep a small margin without changing the saved placement or schedule."""
    if not 0 <= required_bytes <= budget_bytes or budget_bytes <= 0:
        raise ValueError("spill requirement must fit a positive Plan spill budget")
    if supplied is None:
        # A runtime still needs a nonempty spill pool when the Plan never spills.
        capacity = min(budget_bytes, max(1, (required_bytes * 21 + 19) // 20))
        supplied = PinnedHostPool(capacity)
    if not isinstance(supplied, SpillPool):
        raise TypeError("spill must be a SpillPool configuration")
    if not required_bytes <= supplied.capacity <= budget_bytes:
        raise ValueError(
            f"spill capacity must cover the worker's {required_bytes} required "
            f"bytes without exceeding the Plan's {budget_bytes} spill budget"
        )
    return supplied

"""Independent paired progress estimates, without fitting a training timeline."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class PairObservation:
    task_start_ns: float
    task_end_ns: float
    transfers: tuple[tuple[float, float], ...]


def paired_progress(
    solo_tasks: Sequence[PairObservation],
    solo_transfers: Sequence[PairObservation],
    together: Sequence[PairObservation],
) -> tuple[float, float] | None:
    """Use actual overlap and interior transfers; reject unidentifiable trials.

    Exclude the first transfer (rank arrival) and post-task tails. Require three
    supported trials and bounded median absolute deviation. A missing estimate
    is not a guessed tiny rate, nor evidence of zero interference.
    """
    task_rates, transfer_rates = [], []
    for task, copies, pair in zip(solo_tasks, solo_transfers, together, strict=True):
        solo = task.task_end_ns - task.task_start_ns
        elapsed = pair.task_end_ns - pair.task_start_ns
        overlap = sum(
            max(0.0, min(end, pair.task_end_ns) - max(start, pair.task_start_ns))
            for start, end in pair.transfers
        )
        if solo <= 0 or overlap <= 0 or overlap > elapsed + 1000:
            continue
        task_rate = 1 - (elapsed - solo) / overlap
        reference = [end - start for start, end in copies.transfers[1:]]
        interior = [
            end - start
            for start, end in pair.transfers[1:]
            if start >= pair.task_start_ns and end <= pair.task_end_ns
        ]
        if not reference or not interior or min(reference + interior) <= 0:
            continue
        transfer_rate = median(reference) / median(interior)
        if not 0 < task_rate <= 1 or not 0 < transfer_rate <= 1:
            continue
        task_rates.append(task_rate)
        transfer_rates.append(transfer_rate)
    if len(task_rates) < 3:
        return None
    for values in (task_rates, transfer_rates):
        center = median(values)
        if median(abs(value - center) for value in values) > 0.25 * center:
            return None
    return median(task_rates), median(transfer_rates)


def probe_work_counts(runtime_ns: int, transfer_ns: int) -> tuple[int, int]:
    """Bound work while obtaining several transfers fully inside a task burst."""
    tasks = max(1, min(8, math.ceil(4 * transfer_ns / max(1, runtime_ns))))
    copies = max(3, min(128, math.ceil(2 * tasks * runtime_ns / max(1, transfer_ns))))
    return tasks, copies

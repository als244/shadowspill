"""Select explanatory cuts and pace exact replay states."""

import math

import numpy as np


def paced_steps(first, total, frames, *, hold_frames=1):
    """Exact frame budget, with a quadratic ramp and repeated states held."""
    if total <= first:
        return []
    slots = max(1, frames // hold_frames)
    steps = []
    for i in range(1, slots + 1):
        target = first + max(1, math.ceil((total - first) * (i / slots) ** 2))
        duration = round(i * frames / slots) - round((i - 1) * frames / slots)
        if steps and steps[-1][0] == target:
            steps[-1] = (target, steps[-1][1] + duration)
        else:
            steps.append((target, duration))
    return steps


def explanation_events(data, rid, amount, *, minimum_gap=6):
    """Choose visually distinct early cuts, then a release/eviction contrast."""
    cuts = data.trace.reductions[rid].cuts
    total = len(cuts)
    if amount <= 0 or not total:
        return []

    def reason(index):
        comparisons = data.trace.comparisons(rid, index)
        return comparisons[0]["criterion"] if comparisons else None

    def span_length(row):
        first, last = map(int, row[5:7].view("<i8"))
        return max(1, last - first + 1)

    events = [0]
    if amount > 1 and total > 1:
        start = min(max(1, minimum_gap), total - 1)
        end = min(total, max(start + 1, total // 10))
        first = cuts[0]
        first_alias = int(first[4])
        first_size = max(1, int(data.trace.alias_sizes[first_alias]))
        first_span = span_length(first)
        first_reason = reason(0)

        def contrast(index):
            row = cuts[index]
            alias = int(row[4])
            size = max(1, int(data.trace.alias_sizes[alias]))
            span = span_length(row)
            # A changed tie-breaker alone can select a visually identical
            # sibling weight. Prefer differences the viewer can actually see.
            visual = abs(math.log2(size / first_size)) + abs(
                math.log2(span / first_span)
            )
            visual += 3 * (
                data.aliases[alias]["role"] != data.aliases[first_alias]["role"]
            )
            visual += 3 * (int(row[8]) != int(first[8]))
            substantive = reason(index) not in (
                None,
                first_reason,
                "alias",
                "start",
                "cut_id",
            )
            return visual, substantive, -index

        early = max(range(start, end), key=contrast)
        events.append(early)
    if amount > 2 and events[-1] < total - 1:
        start = min(max(events[-1] + max(1, minimum_gap), total // 3), total - 1)
        later = next(
            (i for i in range(start, total) if int(cuts[i, 8]) != int(cuts[0, 8])),
            max(start, total // 3),
        )
        events.append(later)
    if amount > len(events):
        events.extend(
            i
            for i in np.linspace(0, total - 1, min(amount, total), dtype=int)
            if i not in events
        )
    return sorted(int(event) for event in events[:amount])

"""Draw the labels panel without changing replay state."""

from __future__ import annotations

from ....data import GIB, byte_label
from ...style import GREEN, MUTED, WHITE


def draw_labels(self, view, rid, count, event, phase, batch_start):
    d = self.data
    row = view.row
    total = len(d.trace.reductions[rid].cuts)
    context = d.collection_context or {}
    if context or self.title:
        workload = self.title or context.get("workload")
        parts = [workload] if workload else []
        metadata = context.get("metadata", {})
        if metadata.get("trainable", {}).get("mode") == "full":
            parts.append("full training")
        sequence = metadata.get("sequence_length")
        if sequence:
            parts.append(f"{sequence:,} tokens / sequence")
        if sequence and metadata.get("sequences_per_step"):
            parts.append(f"{sequence * metadata['sequences_per_step']:,} tokens / step")
        budget = context.get("physical_budget_bytes")
        if budget is not None:
            parts.append(f"{byte_label(budget)} budget")
        if context.get("spill_budget_bytes") is not None:
            parts.append(f"{byte_label(context['spill_budget_bytes'])} spill")
        self.label(0.035, 0.914, "  ·  ".join(parts), fontsize=11, color=WHITE)
    resolution = d.candidate.get("resolution_recompute_fraction")
    resolution_label = (
        f"recompute resolution {resolution}"
        if resolution is not None
        else f"recompute {d.candidate.get('recompute_share', 0):.0%} of groups"
    )
    policy = [d.candidate["candidate_id"], resolution_label]
    if context.get("ordering"):
        policy.append(f"ordering {context['ordering']}")
    if context.get("geometry"):
        policy.append(f"{context['geometry']} seq / microbatch")
    if context.get("microbatches"):
        policy.append(f"{context['microbatches']} microbatches")
    policy.extend(
        [
            f"{d.compute_seconds:.3f} s of profiled compute",
        ]
    )
    self.label(
        0.035,
        0.879 if context or self.title else 0.914,
        "  ·  ".join(policy),
        fontsize=10.5,
        color=MUTED,
    )
    reduction_context = d.reduction_context(rid)
    object_capacity = reduction_context["object_capacity_bytes"]
    capacity_label = (
        f"{object_capacity / GIB:,.3f} GiB"
        if object_capacity is not None
        else "not recorded"
    )
    self.label(
        0.035,
        0.024,
        f"Reduction {rid + 1}/{len(d.trace.reductions)}  ·  "
        f"object capacity: {capacity_label}",
        fontsize=8.5,
        color=MUTED,
    )
    current_cut = event + 1 if row is not None else count
    counter = f"Cut {current_cut:,}/{total:,}"
    self.label(
        0.975, 0.956, counter, fontsize=17, ha="right", weight="bold", color=WHITE
    )
    if phase == "seed":
        caption = (
            "Start from seed"
            if rid == 0
            else "Restart from seed with revised pressure / capacity"
        )
    elif phase == "final":
        caption = (
            "All boundaries fit"
            if d.trace.reductions[rid].status == 0
            else "Reduction stopped: no legal cut for an over-budget boundary"
        )
    elif batch_start is not None and count - batch_start > 1:
        caption = (
            f"Accelerated  ·  cuts {batch_start + 1:,}-{count:,}  ·  "
            f"{count - batch_start:,} decisions this frame"
        )
    elif phase == "selected":
        caption = f"Choose cut {event + 1:,}  ·  state before applying it"
    else:
        caption = f"Apply cut {count:,}  ·  update residency and pressure"
    self.label(
        0.255,
        0.817,
        caption,
        fontsize=9,
        color=GREEN if phase == "final" else MUTED,
    )

    return reduction_context

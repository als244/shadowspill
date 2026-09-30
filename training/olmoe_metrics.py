"""Aggregate OLMoE's returned microbatch summaries, on the CPU after a step.

Loss terms are weighted by trained targets. Routing counts include every row
processed by the router, including packed padding; their entropy describes
aggregate expert usage, not the mean entropy of individual token predictions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from training.observations import MetricSummary, MetricTable


def reduce_metrics(microbatches: Sequence[dict[str, torch.Tensor]]) -> MetricSummary:
    if not microbatches:
        return MetricSummary()
    if any(value.device.type != "cpu" for mb in microbatches for value in mb.values()):
        raise ValueError("metric reduction expects CPU summaries after the step")
    totals = {key: sum(mb[key] for mb in microbatches) for key in microbatches[0]}
    trained = totals["trained_tokens"].item()
    if trained <= 0:
        raise ValueError("loss metrics require at least one trained target")
    scalars = {
        "loss/cross_entropy": totals["ce_sum"].item() / trained,
        "loss/auxiliary": totals["auxiliary_sum"].item() / trained,
        "loss/weighted_auxiliary": totals["weighted_auxiliary_sum"].item() / trained,
    }
    scalars["loss/total"] = (
        scalars["loss/cross_entropy"] + scalars["loss/weighted_auxiliary"]
    )
    rows = []
    for layer, (counts, probabilities, auxiliary) in enumerate(
        zip(
            totals["expert_counts"],
            totals["probability_sum"],
            totals["layer_auxiliary_sum"],
            strict=True,
        )
    ):
        counts = counts.to(torch.float64)
        assignments = counts.sum().item()
        if assignments <= 0:
            raise ValueError("routing metrics require at least one assignment")
        shares = counts / assignments
        entropy = -(shares * shares.clamp_min(1e-300).log()).sum().item()
        prefix = f"routing/layer_{layer:02d}"
        scalars.update(
            {
                f"{prefix}/load_entropy": entropy,
                f"{prefix}/normalized_load_entropy": entropy / math.log(counts.numel())
                if counts.numel() > 1
                else 0.0,
                f"{prefix}/effective_experts": math.exp(entropy),
                f"{prefix}/max_to_mean_load": counts.max().item()
                / (assignments / counts.numel()),
                f"{prefix}/unused_experts": counts.eq(0).sum().item(),
                f"{prefix}/assignments": assignments,
                f"{prefix}/auxiliary": auxiliary.item() / trained,
            }
        )
        rows.extend(
            (layer, expert, int(count), float(share), float(probability))
            for expert, (count, share, probability) in enumerate(
                zip(counts, shares, probabilities, strict=True)
            )
        )
    table = MetricTable(
        ("layer", "expert", "assignments", "assignment_share", "probability_sum"),
        tuple(rows),
    )
    return MetricSummary(scalars, {"routing/expert_counts": table})

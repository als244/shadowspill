"""Compose the recipe's independent SAVE and RECOMPUTE choices."""

from shadowspill.pytorch import GraphPairPolicy, RetentionPolicy, RetentionRule


def graph_pair_policy(name, *, retained_targets, local_targets=()):
    """Keep operator selection with callers and share only policy composition.

    Local scratch is regenerated inside chunks even in SAVE. Routing choices
    keep their provider-declared retention defaults unless explicitly selected.
    """
    if name not in {"cost-input", "cost-attention", "attention-input"}:
        raise ValueError(f"unknown graph-pair policy: {name}")
    retained = tuple(RetentionRule(target, "retain") for target in retained_targets)
    local = tuple(RetentionRule(target, "recompute") for target in local_targets)
    return GraphPairPolicy(
        save=RetentionPolicy(default="recompute", rules=(*retained, *local))
        if name == "attention-input"
        else RetentionPolicy(rules=local),
        recompute=RetentionPolicy(default="recompute", rules=retained)
        if name == "cost-attention"
        else RetentionPolicy(default="recompute"),
    )

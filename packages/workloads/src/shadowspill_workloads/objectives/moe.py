"""Router balancing from sufficient statistics supplied by any softmax MoE."""


def router_balance(
    counts, probability_sum, rows, *, top_k, layer_reduction, normalize_top_k
):
    counts, probability_sum = counts.float(), probability_sum.float()
    denominator = rows.float().clamp_min(1)
    if layer_reduction == "combined":
        denominator = denominator * counts.shape[0]
        counts, probability_sum = counts.sum(0), probability_sum.sum(0)
    if normalize_top_k:
        counts = counts / top_k
    per_layer = counts.shape[-1] * (
        counts * probability_sum / denominator.square()
    ).sum(-1)
    return per_layer.mean() if layer_reduction == "mean" else per_layer.sum()

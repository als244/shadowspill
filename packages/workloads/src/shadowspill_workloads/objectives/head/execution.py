"""Bind token formulas to an ordinary or a one-pass bounded output head."""

from mlops.objectives import clipped_policy, cross_entropy


def logprobs(evaluated):
    """Prediction-only calls need no gradient seeds."""
    if "logprobs" in evaluated:
        return evaluated["logprobs"].float()
    from shadowspill_workloads.models.shared.token_head import target_logprobs

    return target_logprobs(
        evaluated["head_hidden"], evaluated["head"], evaluated["head_targets"]
    )


def loss(evaluated, loss_inputs, loss_params, *, policy):
    """Reduce the objective inside each head chunk before discarding logits."""
    inputs = [
        loss_inputs["response_mask" if policy else "loss_mask"],
        loss_inputs["weights"],
    ]
    controls = []
    name = "clipped_policy" if policy else "cross_entropy"
    if policy:
        inputs.extend((loss_inputs["old_logprobs"], loss_inputs["advantages"]))
        controls.append(loss_params["clip_ratio"])
        if "reference_logprobs" in loss_inputs:
            inputs.append(loss_inputs["reference_logprobs"])
            controls.append(loss_params["reference_beta"])
    if "head" in evaluated and hasattr(evaluated["head"], "objective"):
        return evaluated["head"].objective(
            evaluated["head_hidden"],
            evaluated["head_targets"],
            objective=name,
            inputs=inputs,
            controls=controls,
        )
    # Packed decoder registration checks its bounded head at construction.
    # Standalone logprob components and ordinary reference heads can supply
    # differentiable predictions to this same token formula.
    current = logprobs(evaluated)
    if policy:
        loss = clipped_policy(
            current,
            *inputs[:4],
            controls[0],
            reference_logprobs=inputs[4] if len(inputs) == 5 else None,
            reference_beta=controls[1] if len(controls) == 2 else None,
        )
    else:
        loss = cross_entropy(current, *inputs)
    return loss, current

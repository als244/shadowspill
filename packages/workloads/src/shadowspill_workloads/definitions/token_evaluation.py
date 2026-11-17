"""One token-model evaluation shared by prediction and router objectives."""

from shadowspill.training.definitions import ObjectiveEvaluation


def compute(model, inputs, loss_inputs, requests):
    import torch

    targets = None
    if requests.get("logprobs"):
        targets = loss_inputs["target_tokens"]
        if not requests.get("unmasked_targets"):
            valid = torch.zeros_like(targets, dtype=torch.bool)
            for request in requests:
                if request.startswith("target_mask."):
                    valid = valid | loss_inputs[request.removeprefix("target_mask.")]
            targets = torch.where(valid, targets, 0)
    if hasattr(model, "token_features"):
        options = {}
        if requests.get("router_statistics"):
            options = {
                "router_statistics": True,
                "router_mask": None
                if requests["include_padding"]
                else loss_inputs["router_mask"],
            }
        return {
            **model.token_features(**inputs, **options),
            "head_targets": targets,
        }
    if requests.get("router_statistics"):
        mask = None if requests["include_padding"] else loss_inputs["router_mask"]
        return model.token_outputs(targets=targets, router_mask=mask, **inputs)
    return {"logprobs": model.target_logprobs(targets=targets, **inputs).float()}


def evaluation(requests):
    return ObjectiveEvaluation("shadowspill/token_outputs@1", compute, requests)


def token_evaluation(config, key, *, policy):
    requests = {"logprobs": True}
    if key.operation == "forward" and key.outputs == "logprobs":
        requests["unmasked_targets"] = True
    else:
        requests["target_mask." + ("response_mask" if policy else "loss_mask")] = True
    return evaluation(requests)

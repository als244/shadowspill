"""Coordinates of native FP8 banks and BF16 compute payloads."""

from shadowspill.training.definitions import ComponentLayout


def layers(architecture):
    return range(
        architecture["first_k_dense_replace"], architecture["num_hidden_layers"]
    )


def layouts(config):
    ep = config.implementation == "mlops_ep"
    banks = ("experts.gate_up_weight", "experts.down_weight") if ep else ("w13", "w2")
    fields = []
    for layer in layers(config.architecture):
        root = f"layers.{layer}.mlp."
        for bank in banks:
            name = root + bank
            if config.architecture["expert_storage"] == "fp8_block":
                fields.extend(
                    (
                        ComponentLayout(
                            name, ("_data",), (0, 1, 2), "mlops/block-fp8@1"
                        ),
                        ComponentLayout(
                            name,
                            ("_scale",),
                            (0, 1, 2),
                            "mlops/block-fp8@1",
                            blocks=(1, 128, 128),
                        ),
                    )
                )
            elif ep and config.grad_dtype == "float32":
                fields.append(
                    ComponentLayout(name, ("_data",), (0, 1, 2), "mlops/bf16-compute@1")
                )
        if ep and config.grad_dtype == "float32":
            fields.extend(
                ComponentLayout(
                    root + "experts.lora_" + factor,
                    ("_data",),
                    (0, 1, 2),
                    "mlops/bf16-compute@1",
                )
                for factor in ("gate_up_a", "gate_up_b", "down_a", "down_b")
            )
    return tuple(fields)

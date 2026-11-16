"""Declare Quack layers and their shared buffer without opening a communicator."""

import torch


def configuration(settings):
    """Materialize the provider's configuration only in a model/resource worker."""
    from mlops.expert_parallel import QuackMoEConfig

    values = dict(settings)
    for name in ("router_dtype", "router_weight_grad_dtype", "weight_grad_dtype"):
        values[name] = {"float32": torch.float32, "bfloat16": torch.bfloat16}[
            values[name]
        ]
    return QuackMoEConfig(**values)


def initialize(*, name, device, groups, options):
    """Translate portable workload options into the existing provider factory.

    The provider owns allocation, registration and cleanup, including failures.
    This adapter never opens an additional group or owns a second buffer.
    """
    from mlops.expert_parallel.quack.config import config_signature
    from mlops.expert_parallel.quack.resources import initialize as initialize_provider

    signature = config_signature(configuration(options["config"]))
    layers = [{"name": layer, "config": signature} for layer in options["layers"]]
    return initialize_provider(
        name=name, device=device, groups=groups, options={"layers": layers}
    )


def model(entry, config, options, precision):
    from mlops.expert_parallel import QuackMoE

    precision.apply()
    names = iter(f"blocks.{index}.moe.experts" for index in range(config.n_layers))
    return entry.model_class("mlops")(
        config,
        parameter_device="meta",
        dtype=torch.bfloat16,
        expert_factory=lambda: QuackMoE.from_resource(options, next(names)),
    )

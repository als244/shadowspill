"""Construct the existing MLOps providers, with matched training contracts."""

import torch


def tensor(shape, *, seed, std=0.02, dtype=torch.bfloat16):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.empty(shape, device="cuda", dtype=dtype).normal_(
        std=std, generator=generator
    )


class SonicLayer(torch.nn.Module):
    def __init__(self, shape, precision, training, seed):
        super().__init__()
        from mlops.expert_parallel.parameters import BF16ComputeWeight
        from mlops.expert_parallel.quack.parameters.fp8 import QuackFP8Weight

        self.shape, self.precision, self.training_mode = shape, precision, training
        self.router_weight = torch.nn.Parameter(
            tensor((shape.experts, shape.width), seed=seed, dtype=torch.float32),
            requires_grad=training == "full",
        )
        for index, (name, dimensions) in enumerate(
            (
                ("gate_up_weight", (shape.experts, 2 * shape.hidden, shape.width)),
                ("down_weight", (shape.experts, shape.width, shape.hidden)),
            )
        ):
            values = tensor(dimensions, seed=seed + index + 1)
            weight = (
                QuackFP8Weight.quantize(
                    values, dtype=torch.float32, interleaved=index == 0
                )
                if precision == "fp8_current"
                else BF16ComputeWeight(values, dtype=torch.float32)
            )
            self.register_parameter(
                name, torch.nn.Parameter(weight, requires_grad=training == "full")
            )
        if training == "lora":
            for index, (name, dimensions) in enumerate(
                (
                    ("lora_gate_up_a", (shape.experts, shape.lora_rank, shape.width)),
                    (
                        "lora_gate_up_b",
                        (shape.experts, 2 * shape.hidden, shape.lora_rank),
                    ),
                    ("lora_down_a", (shape.experts, shape.lora_rank, shape.hidden)),
                    ("lora_down_b", (shape.experts, shape.width, shape.lora_rank)),
                )
            ):
                # Nonzero B factors exercise the trained adapter's input path.
                value = tensor(dimensions, seed=seed + 10 + index, std=0.015)
                self.register_parameter(
                    name, torch.nn.Parameter(BF16ComputeWeight(value))
                )
        self.residual = None

    def forward(self, x):
        import mlops
        from mlops.dispatch.context import use_implementation, weight_gradients_at
        from mlops.expert_parallel.quack.parameters.fp8 import QuackFP8Weight
        from mlops.lora.experts import expert_lora

        def compute(weight):
            return (
                weight
                if isinstance(weight, QuackFP8Weight)
                else weight.to(torch.bfloat16)
            )

        if self.residual is None:
            self.residual = torch.zeros_like(x, requires_grad=False)
        options = dict(
            top_k=self.shape.top_k,
            routing_mode="topk_then_softmax",
            compute_precision=self.precision,
        )
        arguments = (
            x,
            self.residual,
            self.router_weight,
            compute(self.gate_up_weight),
            compute(self.down_weight),
        )
        with weight_gradients_at(torch.float32):
            if self.training_mode == "lora":
                return expert_lora(
                    *arguments,
                    compute(self.lora_gate_up_a),
                    compute(self.lora_gate_up_b),
                    compute(self.lora_down_a),
                    compute(self.lora_down_b),
                    scale=1.0,
                    implementation="sonicmoe",
                    **options,
                )[0]
            with use_implementation("moe", "sonicmoe.moe"):
                return mlops.moe(*arguments, **options)[0]

    def close(self):
        pass


def build(shape, *, backend, precision, training, tokens, world_size, seed):
    if backend == "sonicmoe":
        if world_size != 1:
            raise ValueError("the SonicMoE compute baseline uses one GPU")
        return SonicLayer(shape, precision, training, seed), None
    from mlops.expert_parallel import (
        LoRAConfig,
        QuackMoE,
        QuackMoEConfig,
        QuackMoELoRA,
    )
    from mlops.expert_parallel.buffers import create_buffer

    config = QuackMoEConfig(
        ep_size=world_size,
        num_experts=shape.experts,
        top_k=shape.top_k,
        model_dim=shape.width,
        expert_hidden_dim=shape.hidden,
        compute_precision=precision,
        activation_transport="bf16",
        weight_grad_dtype=torch.float32,
        router_dtype=torch.float32,
        router_weight_grad_dtype=torch.float32,
        num_comm_sms=32,
        profile_ranges=False,
    )
    buffer = create_buffer(config, tokens, torch.distributed.group.WORLD)
    options = dict(buffer=buffer, device=torch.device("cuda", 0))
    if training == "lora":
        layer = QuackMoELoRA(
            config,
            lora=LoRAConfig(rank=shape.lora_rank, alpha=shape.lora_rank),
            **options,
        )
        with torch.no_grad():
            for index, parameter in enumerate(layer.lora_parameters()):
                parameter.copy_(
                    tensor(tuple(parameter.shape), seed=seed + 10 + index, std=0.015)
                )
    else:
        layer = QuackMoE(config, **options)
    # Same router and input distribution on both implementations; real top-k.
    with torch.no_grad():
        layer.router_weight.copy_(
            tensor((shape.experts, shape.width), seed=seed, dtype=torch.float32)
        )
    return layer, buffer

"""Local replicated SonicMoE router and experts."""

import torch

from ..api import ModuleImplementation
from ..tensors import expert_tensor, tensor


class SonicLayer(torch.nn.Module):
    def __init__(self, shape, precision, training, seed, router_weight_dtype="bf16"):
        super().__init__()
        from mlops.expert_parallel.parameters import BF16ComputeWeight
        from mlops.expert_parallel.quack.parameters.fp8 import QuackFP8Weight

        self.shape, self.precision, self.training_mode = shape, precision, training
        router_values = tensor(
            (shape.experts, shape.width),
            seed=seed,
            dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[router_weight_dtype],
        )
        self.router_weight = torch.nn.Parameter(
            BF16ComputeWeight(router_values)
            if router_values.dtype == torch.bfloat16 and training == "full"
            else router_values,
            requires_grad=training == "full",
        )
        for index, (name, dimensions) in enumerate(
            (
                ("gate_up_weight", (shape.experts, 2 * shape.hidden, shape.width)),
                ("down_weight", (shape.experts, shape.width, shape.hidden)),
            )
        ):
            values = expert_tensor(dimensions, seed=seed, slot=index)
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
                value = expert_tensor(dimensions, seed=seed, slot=2 + index, std=0.015)
                self.register_parameter(
                    name, torch.nn.Parameter(BF16ComputeWeight(value))
                )
        self.residual = None

    def forward(self, x):
        import mlops
        from mlops.dispatch.context import use_implementation, weight_gradients_at
        from mlops.expert_parallel.parameters import BF16ComputeWeight
        from mlops.expert_parallel.quack.parameters.fp8 import QuackFP8Weight
        from mlops.lora.experts import expert_lora

        def compute(weight):
            return (
                weight
                if isinstance(weight, (QuackFP8Weight, BF16ComputeWeight))
                else weight.to(torch.bfloat16)
            )

        if self.residual is None or self.residual.shape != x.shape:
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
                    scale=self.shape.lora_alpha / self.shape.lora_rank,
                    implementation="sonicmoe",
                    **options,
                )[0]
            with use_implementation("moe", "sonicmoe.moe"):
                return mlops.moe(*arguments, **options)[0]

    def close(self):
        pass


def build(
    shape, *, precision, training, tokens, world_size, seed, router_weight_dtype="bf16"
):
    layer = SonicLayer(shape, precision, training, seed, router_weight_dtype)
    storage = {}
    for name, parameter in layer.named_parameters():
        flatten = getattr(parameter, "__tensor_flatten__", None)
        components = (
            [(part, getattr(parameter, part)) for part in flatten()[0]]
            if flatten is not None
            else [("weight", parameter)]
        )
        storage[name] = {
            "logical_dtype": str(parameter.dtype),
            "trainable": parameter.requires_grad,
            "components": {
                part: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "bytes": value.numel() * value.element_size(),
                }
                for part, value in components
            },
        }
    return ModuleImplementation(
        layer,
        metadata={
            "parameter_storage": storage,
            "expert_backend": "SonicMoE",
            "dispatcher": None,
            "fp8_scheme": "E4M3 per-row, independently quantized orientations",
            "gradient_dtype": "float32",
            "router_weight_dtype": router_weight_dtype,
            "router_logit_dtype": "float32",
            "router_gradient_compute_dtype": "float32",
            "adapter_compute_dtype": "bfloat16",
        },
    )

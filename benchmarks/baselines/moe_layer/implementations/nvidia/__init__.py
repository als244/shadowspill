"""Megatron-Core routing/dispatch and Transformer Engine grouped expert GEMMs.

The three public implementation folders select only the dispatcher. LoRA uses
per-expert TE grouped A/B projections, outside FP8 autocast, with FP32 main_grad
buffers. No ShadowSpill or MLOps runtime participates in this implementation.
"""

from functools import partial

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8CurrentScaling, Format

from ..api import ModuleImplementation
from ..tensors import expert_tensor, tensor


class LoRAProjection(torch.nn.Module):
    """Add independent BF16 TE grouped adapters to a frozen grouped projection."""

    def __init__(self, base, shape):
        super().__init__()
        self.base = base
        self.scale = shape.lora_alpha / shape.lora_rank
        options = dict(
            num_gemms=base.num_gemms,
            params_dtype=torch.bfloat16,
            device="cuda",
            bias=False,
            fuse_wgrad_accumulation=True,
            init_method=partial(torch.nn.init.normal_, std=0.015),
        )
        self.lora_a = te.GroupedLinear(
            in_features=base.in_features, out_features=shape.lora_rank, **options
        )
        self.lora_b = te.GroupedLinear(
            in_features=shape.lora_rank, out_features=base.out_features, **options
        )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("base"), name)

    def forward(self, inputs, splits):
        output, bias = self.base(inputs, splits)
        with te.fp8_autocast(enabled=False):
            hidden = self.lora_a(inputs, splits)
            delta = self.lora_b(hidden * self.scale, splits)
        return output + delta, bias


class NvidiaLayer(torch.nn.Module):
    def __init__(self, layer, recipe, fp8, dispatcher, tokens):
        super().__init__()
        self.layer = layer
        self.recipe = recipe
        self.fp8 = fp8
        self.dispatcher, self.tokens = dispatcher, tokens

    def prepare_case(self, *, tokens):
        if self.dispatcher == "ncclep" and self.tokens != tokens:
            # Native NCCL EP fixes its dispatch capacity on first use. Release
            # that process-wide context outside timing; the same model weights
            # remain, and the next warmup bootstraps the new token capacity.
            from megatron.core.transformer.moe.token_dispatcher import (
                nccl_ep_release_context,
            )

            nccl_ep_release_context()
        self.tokens = tokens

    def forward(self, inputs):
        with te.fp8_autocast(enabled=self.fp8, fp8_recipe=self.recipe):
            output, bias = self.layer(inputs.unsqueeze(1))
        if bias is not None:
            raise AssertionError("bias-free layer returned a bias")
        return output.squeeze(1)


def build(
    shape,
    *,
    precision,
    training,
    tokens,
    world_size,
    seed,
    dispatcher,
    router_weight_dtype="bf16",
):
    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig
    from transformer_engine.pytorch.tensor import QuantizedTensor

    if world_size < 2 or not torch.distributed.is_initialized():
        raise ValueError("NVIDIA dispatcher benchmarks require a distributed EP group")
    parallel_state.initialize_model_parallel(
        expert_model_parallel_size=world_size,
        expert_tensor_parallel_size=1,
        create_gloo_process_groups=False,
    )
    model_parallel_cuda_manual_seed(seed)
    fp8 = precision == "fp8_current"
    recipe = Float8CurrentScaling(fp8_format=Format.E4M3)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=shape.width,
        num_attention_heads=1,
        ffn_hidden_size=shape.hidden,
        moe_ffn_hidden_size=shape.hidden,
        num_moe_experts=shape.experts,
        moe_router_topk=shape.top_k,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend=dispatcher,
        moe_permute_fusion=True,
        moe_router_pre_softmax=False,
        moe_router_score_function="softmax",
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0,
        moe_router_dtype="fp32",
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        params_dtype=torch.bfloat16,
        bf16=True,
        gradient_accumulation_fusion=True,
        expert_model_parallel_size=world_size,
        expert_tensor_parallel_size=1,
        hidden_dropout=0,
        attention_dropout=0,
        moe_use_grouped_tensor=False,
        fp8="e4m3" if fp8 else None,
        fp8_recipe="tensorwise",
        fp8_param=fp8,
        init_method=partial(torch.nn.init.normal_, std=0.02),
        output_layer_init_method=partial(torch.nn.init.normal_, std=0.02),
    )
    builder = get_mlp_module_spec(
        use_te=True, num_experts=shape.experts, moe_grouped_gemm=True
    )
    with te.quantized_model_init(
        enabled=fp8, recipe=recipe, preserve_high_precision_init_val=False
    ):
        layer = builder(config=config, layer_number=1)
    first_expert = torch.distributed.get_rank() * (shape.experts // world_size)
    with torch.no_grad():
        for slot, projection in enumerate(
            (layer.experts.linear_fc1, layer.experts.linear_fc2)
        ):
            for local_id in range(projection.num_gemms):
                parameter = getattr(projection, f"weight{local_id}")
                value = expert_tensor(
                    (1, *parameter.shape),
                    seed=seed,
                    slot=slot,
                    first_expert=first_expert + local_id,
                )[0]
                parameter.copy_(value)
    # Native Megatron accepts matching BF16 operands with FP32 output logits.
    # Only the explicit FP32 weight option needs an FP32 activation copy.
    router_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[router_weight_dtype]
    layer.router.to(device="cuda", dtype=router_dtype)
    if router_dtype == torch.float32:
        layer.router.register_forward_pre_hook(
            lambda module, args: (args[0].float(), *args[1:])
        )
    with torch.no_grad():
        layer.router.weight.copy_(
            tensor(
                (shape.experts, shape.width),
                seed=seed,
                dtype=router_dtype,
            )
        )
    if training == "lora":
        layer.requires_grad_(False)
        layer.experts.linear_fc1 = LoRAProjection(layer.experts.linear_fc1, shape)
        layer.experts.linear_fc2 = LoRAProjection(layer.experts.linear_fc2, shape)
        with torch.no_grad():
            for projection_id, projection in enumerate(
                (layer.experts.linear_fc1, layer.experts.linear_fc2)
            ):
                for factor_id, factor in enumerate(
                    (projection.lora_a, projection.lora_b)
                ):
                    slot = 2 + 2 * projection_id + factor_id
                    for local_id in range(factor.num_gemms):
                        parameter = getattr(factor, f"weight{local_id}")
                        parameter.copy_(
                            expert_tensor(
                                (1, *parameter.shape),
                                seed=seed,
                                slot=slot,
                                first_expert=first_expert + local_id,
                                std=0.015,
                            )[0]
                        )
    if training == "full" and router_dtype == torch.bfloat16:
        # Native Megatron returns the router gradient in its parameter dtype.
        # Accumulate that native result into FP32, as its DDP wrapper would;
        # expert GEMMs separately write directly into their FP32 main_grad.
        parameter = layer.router.weight
        parameter.main_grad = torch.zeros_like(parameter, dtype=torch.float32)

        def accumulate_router_gradient(parameter):
            parameter.main_grad.add_(parameter.grad)
            parameter.grad = None

        parameter.register_post_accumulate_grad_hook(accumulate_router_gradient)
    for name, parameter in layer.named_parameters():
        if parameter.requires_grad and name.startswith("experts."):
            parameter.main_grad = torch.zeros(
                parameter.shape, device="cuda", dtype=torch.float32
            )
            parameter.grad_added_to_main_grad = False
            parameter.zero_out_wgrad = True
        if hasattr(parameter, "clear_high_precision_init_val"):
            parameter.clear_high_precision_init_val()
    expert_parameters = [
        (name, parameter)
        for name, parameter in layer.experts.named_parameters()
        if "lora_" not in name
    ]
    quantized = sum(
        parameter.numel()
        for _, parameter in expert_parameters
        if isinstance(parameter, QuantizedTensor)
    )
    expert_elements = sum(parameter.numel() for _, parameter in expert_parameters)
    if fp8 and quantized != expert_elements:
        raise AssertionError(
            f"FP8 base storage incomplete: {quantized}/{expert_elements} elements"
        )
    cleanup = (parallel_state.destroy_model_parallel,)
    if dispatcher == "ncclep":
        from megatron.core.transformer.moe.token_dispatcher import (
            nccl_ep_release_context,
        )

        cleanup += (nccl_ep_release_context,)
    return ModuleImplementation(
        NvidiaLayer(layer, recipe, fp8, dispatcher, tokens),
        cleanup=cleanup,
        metadata={
            "expert_backend": "Megatron-Core TEGroupedMLP",
            "dispatcher": dispatcher,
            "expert_parallel_size": world_size,
            "fp8_scheme": "TE E4M3 per-tensor current scaling",
            "expert_parameter_elements_per_rank": expert_elements,
            "quantized_expert_elements_per_rank": quantized,
            "frozen_expert_fp8_storage": fp8 and training == "lora",
            "gradient_dtype": "float32",
            "router_weight_dtype": router_weight_dtype,
            "router_logit_dtype": "float32",
            "router_gradient_compute_dtype": str(router_dtype),
            "adapter_compute_dtype": "bfloat16",
            "adapter_storage_dtype": "bfloat16",
            "adapter_backend": "TE per-expert GroupedLinear with fused FP32 main_grad",
            "activation_transport": "bfloat16",
            "recompute": False,
            "parameter_seed_schedule": (
                "seed + 10000 + 16 * global_expert_id + projection_slot"
            ),
        },
    )

"""Option records and permitted values for megatron_core."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Option:
    """One of Megatron's options, as a case sets it: its name in Megatron,
    the Bridge configuration that holds it (``model``, the provider and its
    ``TransformerConfig``; ``optimizer``; ``ddp``; ``dist``; ``mixed``, the
    mixed-precision configuration Bridge copies onto the others; ``comm``,
    its communication-overlap configuration; ``environment``, a variable the
    launcher sets in the trainer's environment), joined by ``+`` where Bridge
    keeps two in step, the type of its value, what it does, and its allowed
    values. ``form`` says how a string is read: ``names``, a comma-separated
    list of the allowed values; ``integers``, a comma-separated list of
    integers; ``dtype``, a dtype's name."""

    name: str
    section: str
    kind: type
    help: str
    choices: tuple = ()
    form: str = "value"


RECOMPUTE_MODULES = (
    "core_attn",
    "moe_act",
    "layernorm",
    "mla_up_proj",
    "mlp",
    "moe",
    "shared_experts",
    "gdn_norm_out",
    # Of model families no catalog model is: Gated DeltaProduct layers'
    # projections, which recompute nothing in another model, and hybrid
    # models' hyper-connections, shortcut blocks and residual streams, which
    # Megatron refuses without them.
    "gdp_in_proj",
    "gdp_qkv",
    "mhc",
    "shortcut_pre_mlp_layernorm",
    "residual_stream",
)
OFFLOAD_MODULES = (
    "attn_norm",
    "qkv_linear",
    "core_attn",
    "attn_proj",
    "mlp_norm",
    "expert_fc1",
    "moe_act",
    "fused_group_mlp",
    # Gated DeltaProduct layers' and hybrid shortcut blocks'; Megatron
    # refuses the shortcut blocks' without them.
    "gdp_qkv",
    "shortcut_post_norm",
)
#: What selective recomputation recomputes when a case names no modules:
#: Megatron's own default, core attention, with the norms, and in a mixture
#: of experts the experts' activation too.
DEFAULT_RECOMPUTE_MODULES = ("core_attn", "layernorm")
MIXTURE_RECOMPUTE_MODULES = ("moe_act",)
SHARDING = ("no_shard", "optim", "optim_grads", "optim_grads_params")
DTYPES = ("bfloat16", "float16", "float32")

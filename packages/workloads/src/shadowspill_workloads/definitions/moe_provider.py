"""Portable preparation and allocation contracts for supplied BF16 MoE models.

Sonic's routing maps have T*K rows and E+1 offsets. Up/down activations and
cotangents use those complete assignment arrays; parameter products allocate
the declared expert/factor shapes. Empty experts change device work, not storage.
LoRA rank padding depends only on the factor shape. Quack's varlen GEMMs use
split_k=1 and at most a fixed one-element dynamic-scheduler counter; they never
size outputs from the contents of the routing offsets.

The available factory captures selected kernels/launch choices and restores
them in cold workers. Dispatch remains provider-owned. This contract covers
the ordinary BF16, unquantized model/factor paths declared by these definitions;
it does not cover FP8 containers, EP, or destination-writing backward variants.
"""

from shadowspill.preparation import OperatorPreparation

from .moe_allocations import CONTRACTS

PROVIDERS = ("quack", "sonicmoe", "cutlass")
CONTRACT_ID = "shadowspill/sonicmoe-bf16-allocation@1"
SONIC_CONTRACTS = {
    f"torch.ops.mlops.{name}.default": CONTRACT_ID
    for name in (
        "sonicmoe_routing_metadata",
        "moe_sonicmoe_fwd",
        "moe_sonicmoe_bwd",
        "moe_sonicmoe_prepare_fwd",
        "moe_sonicmoe_prepare_bwd",
        "moe_sonicmoe_finish_fwd",
        "moe_sonicmoe_finish_bwd",
        "lora_sonicmoe_up_fwd",
        "lora_sonicmoe_up_bwd",
        "lora_sonicmoe_down_fwd",
        "lora_sonicmoe_down_bwd",
    )
}


def preparations(dtype):
    if dtype != "bfloat16":
        return ()
    return (OperatorPreparation(factory="mlops.preparation.available:factory"),)


def allocations(dtype):
    if dtype != "bfloat16":
        return CONTRACTS
    return {
        **CONTRACTS,
        **SONIC_CONTRACTS,
        # Binding loads all saved objects before execution. Activation only
        # checks their lifetime/context and selects the bound calls/choices;
        # it neither reads tensor values nor allocates device storage. The
        # unsupported-device binding is an empty context with the same guarantee.
        "mlops.preparation.available:factory": (
            "shadowspill/sonicmoe-prepared-activation@1"
        ),
    }

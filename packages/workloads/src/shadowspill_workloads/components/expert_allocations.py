"""Fixed-geometry allocation contracts for the supplied BF16 Quack EP path.

The component fixes BF16 computation/transport, one chunk, owned gradients,
shared expert banks and untuned, static GEMM policies. With S input rows, K
selected experts, E experts, G ranks and q=E/G local experts, MoonEP reserves
P=S*K+2*q*(128-1) dispatch rows. Routing values select slots inside those buffers;
they do not resize them. Communication buffers, VMM-aligned publication/reduction
banks and their cleanup belong to the separately admitted Resource lifetime.

Forward allocates fixed MoonEP plan tensors (int32 backing storage rounded to
multiples of four), E+q cumulative counts, 2*q+1 local offsets, P-row saved inputs,
probabilities and gate/up activations, and S-row combined outputs. Backward uses
the same saved plan and fixed P-row derivatives. Each home gradient is allocated
at the bank's VMM-aligned shape, then returned as a contiguous q-expert tensor;
dtype conversion and padding removal depend only on that fixed geometry.

Quack's GEMMs receive full P-row tensors and fixed expert-offset array extents.
They use split_k=1 and static scheduling. Empty experts alter device work, never
storage. The small aligned offset clones likewise depend on q and fixed layout.
Router selection allocates S*K assignments, E counts and E+1 offsets; router
projection/backward use fixed dense GEMM shapes and requested gradient dtypes.

These exact calls are source-pinned with MLOps, MoonEP, Quack, Sonic and Cutlass.
They establish value invariance, not byte counts: profiling, shared resource
admission and cooperating-rank execution checks remain required. No FP8, LoRA,
fused shared-expert or output-only EP operator is covered by these declarations.
"""

CONTRACT_ID = "shadowspill/quack-ep-bf16-allocation@1"

CONTRACTS = {
    **{
        f"torch.ops.mlops_ep_quack_v1.{name}.default": CONTRACT_ID
        for name in ("forward", "backward", "bf16_forward", "bf16_backward")
    },
    **{
        f"torch.ops.mlops_ep_quack_router.{name}.default": CONTRACT_ID
        for name in ("forward", "backward")
    },
    **{
        f"torch.ops.mlops_router.{name}.default": CONTRACT_ID
        for name in ("logits", "gradients")
    },
}

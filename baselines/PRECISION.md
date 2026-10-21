# Precision across the engines

Every engine takes one precision contract, `baselines/common/precision.py`,
with the names ShadowSpill's quickstart uses: the dtype the model computes
at, the dtype of the masters, the dtype gradients are summed at, the dtype
of the optimizer state, the two roundings, and whether a matrix multiply
adds its weight gradient into a narrower running gradient itself. Each
engine states what it honors as `Capabilities` and in its rules. A request
it cannot honor is refused before launch, with the reason, and is never
substituted. An experiment runs all its engines at one contract, named
once in its spec ([README.md](README.md#experiments)). This page gives what
each value means and what each engine does with it.

## The contract

| Value | Flag | Meaning | Default |
|---|---|---|---|
| Model dtype | `--model-dtype {bfloat16,float16,float32}` | What the weights the forward and backward read, and the activations, are kept and computed at | `bfloat16` |
| Master dtype | `--master-dtype {float32,bfloat16,float16,none}` | What a trained weight's master copy is kept and stepped at; `none` steps the model's weights themselves | `float32` |
| Gradient dtype | `--grad-dtype {float32,bfloat16,float16,none}` | What gradients are summed at over a step, across its microbatches and across ranks; `none` is the model dtype | the model dtype (FSDP2: `float32`) |
| Optimizer state dtype | `--opt-state-dtype {float32,bfloat16,float16,parameter,none}` | What AdamW keeps its two moments at; `parameter` is the dtype of what it steps, `none` the optimizer's own default | `float32` |
| Parameter rounding | `--parameter-rounding {nearest,stochastic,none}` | How the optimizer rounds a weight it steps at a dtype narrower than its update; `none` is the engine's own, rounding to nearest everywhere. ShadowSpill alone takes `nearest` by name; the others refuse it and round to nearest under `none`, and FSDP2 also takes `stochastic` | `none` |
| Optimizer state rounding | `--opt-state-rounding {nearest,stochastic,none}` | How the optimizer rounds a moment it stores narrower than float32; `none` is the engine's own, rounding to nearest everywhere. ShadowSpill alone takes either rounding by name; the others offer only `none` | `none` |
| Fused narrow-gradient accumulation | `--[no-]fuse-narrow-gradient-accumulation` | With gradients summed at a dtype narrower than a multiply's float32 accumulator (bfloat16, float16), a later microbatch's weight gradient that a multiply computes is added into the running gradient by the multiply as it writes it, instead of being written at the narrow dtype and added after; it changes nothing for float32 gradients (below) | on (FSDP2, DeepSpeed, AutoModel: off) |

The stepped dtype is the master dtype when masters are kept, the model
dtype otherwise. The default is bfloat16 compute and gradients with float32
masters and moments, a multiply adding each later microbatch's weight
gradient into the bfloat16 running gradient itself. FSDP2 cannot sum
gradients at the model dtype under float32 masters (below), so its own
default gradient dtype is float32; FSDP2, DeepSpeed and AutoModel, under
whose autograd every weight gradient is written before it is added, cannot
fuse that accumulation, so their own default adds after.

## Gradients: creation, accumulation and reduction

A weight's gradient is made by each microbatch's backward, summed over the
step's microbatches on each rank (gradient accumulation), and combined over
the data-parallel ranks (the reduction). The shared objective normalizes
each microbatch's summed loss by the rank's tokens per update, and the
reduction averages the ranks' gradients (a mean in FSDP2, DeepSpeed and the
reference; ShadowSpill divides each rank's share by the world size and
sums), which then produces the global mean exactly once; Megatron-Core and
AutoModel train with their frameworks' own losses. A dtype applies at each
of the three places, and an engine that reduces after every microbatch
interleaves the last two.

**Creation.** Under PyTorch's autograd, eager or compiled, a weight's
gradient has the weight's own dtype: PyTorch casts whatever an operation
returns to it. So FSDP2, DeepSpeed and the reference create every
microbatch's gradients at the dtype each weight computes at, whatever the
gradient dtype: the model dtype, or float32 for a parameter a model pins
there, except under DeepSpeed's bf16 and fp16 modes, which cast every
parameter, the pinned ones included, to the model dtype (the reference
beside DeepSpeed casts them too). ShadowSpill runs captured graphs that
write gradients into the step's own buffers instead: mlops's operations
that sum a weight's gradient over rows (norms, embeddings, heads, experts,
LoRA factors) hand it over at the gradient dtype, with no rounding to the
model dtype first.

A weight whose gradient a matrix multiply computes, every linear layer's,
can have the multiply write it where the step keeps it. A multiply of
bfloat16 operands sums its products at float32; the first microbatch's
gradient is then either written at the gradient dtype by the multiply
itself or written at bfloat16 and converted, and a later microbatch's is
either added into the running gradient by the multiply as it writes it
(`C = A @ B + C`, the product added at float32 and rounded once to the
running gradient's dtype, though the BLAS kernel decides whether it rounds
the product first) or written at bfloat16 and added after (a second
rounding when the running gradient is bfloat16). Fused narrow-gradient
accumulation is the second choice for gradients narrower than float32;
float32 gradients an engine fuses wherever it fuses at all:

| Engine | float32 gradients of bfloat16 weights | bfloat16 gradients |
|---|---|---|
| FSDP2, DeepSpeed, AutoModel, the reference (autograd) | Each microbatch's product at bfloat16, then summed at float32 | The product at bfloat16, then a separate addition; fused accumulation is refused |
| ShadowSpill | The multiply writes the first microbatch's gradient at float32 and adds each later one into the float32 running gradient as it writes it | By default the multiply adds into the running gradient as it writes it; with `--no-fuse-narrow-gradient-accumulation`, the product at bfloat16, then a separate addition |
| Megatron-Core, `gradient_accumulation_fusion` on (Bridge's default wherever Transformer Engine imports) | As ShadowSpill: Transformer Engine's weight-gradient multiply writes into the float32 accumulation buffer (`grad_reduce_in_fp32`) | The multiply adds into the bfloat16 buffer as it writes it: `--fuse-narrow-gradient-accumulation` |
| Megatron-Core, `gradient_accumulation_fusion` off | As autograd: the product at bfloat16 in the weight's own gradient, added into the float32 buffer | The product at bfloat16, then a separate addition |

So with float32 gradients ShadowSpill, and Megatron-Core with
`gradient_accumulation_fusion` on, never round a weight gradient to
bfloat16 where FSDP2 and DeepSpeed round each microbatch's;
`--fuse-narrow-gradient-accumulation` decides only the bfloat16 case.
Megatron-Core's engine sets the fusion from the contract for bfloat16
gradients, and for float32 gradients from the case's
`--[no-]gradient-accumulation-fusion`, or else leaves Bridge's default: on
wherever Transformer Engine imports, Megatron-Core's own output layer then
fusing through APEX's extension, which the environment installs. Its
qualification matrix runs float32 gradients at that default and with the
fusion off.

**Accumulation and reduction, per engine:**

| Engine | Summed over microbatches | Reduced across ranks | When it reduces |
|---|---|---|---|
| FSDP2 | Into the shard's gradient, at the shard's dtype (the stepped dtype) with `--gradient-sync per-microbatch`; with `last-microbatch`, each microbatch's unsharded gradient is cast to `reduce_dtype` and summed there | A reduce-scatter at `MixedPrecisionPolicy.reduce_dtype`, the gradient dtype; the parameters a model pins at float32 are averaged by hand | After every microbatch (`per-microbatch`), or once after the last (`last-microbatch`) |
| DeepSpeed ZeRO-1 | In each weight's gradient at the model dtype; at float32 when `--grad-dtype float32` asks for it (`data_types.grad_accum_dtype`; under a bfloat16 model, DeepSpeed's BF16 optimizer) | At the model dtype; at float32 with `--grad-dtype float32` (`communication_data_type`) | Once, at the update's last microbatch |
| DeepSpeed ZeRO-2, ZeRO-3 | In the gradient partitions, at the model dtype | A reduce-scatter at the model dtype, overlapped with backward in buckets (`--overlap-comm`, `--reduce-bucket-size`) | After every microbatch |
| ShadowSpill | Into the step's running gradient at the gradient dtype; by default a multiply adds its product into a narrower running gradient as it writes it, and with `--no-fuse-narrow-gradient-accumulation` writes it at that dtype to be added after | A sum at the gradient dtype: an all-reduce, or a reduce-scatter onto the rank that owns each slice of the optimizer state | Once, after the last microbatch |
| Megatron-Core | Into DDP's accumulation buffer, at float32 with `grad_reduce_in_fp32` (float32 gradients) or at the model dtype; under Megatron-FSDP, its main-gradient buffer at the gradient dtype | An all-reduce, or a reduce-scatter with the distributed optimizer, at the buffer's dtype | Once, after the last microbatch, or in buckets during the last backward with `overlap_grad_reduce` |
| AutoModel | Into each weight's gradient at the weights' storage dtype (on one rank, the only one); under FSDP2's default `defer_fsdp_grad_sync`, the unsharded gradient across microbatches; under Megatron-FSDP, its main-gradient buffer, at float32 with `grad_reduce_in_fp32` | FSDP2: a reduce-scatter at `mp_policy.reduce_dtype`, the contract's gradient dtype; Megatron-FSDP: at its buffer's dtype | Once, after the last microbatch (`defer_fsdp_grad_sync`), or after every microbatch with `--no-defer-fsdp-grad-sync` |
| Reference | At the gradient dtype | An all-reduce at the gradient dtype | After every microbatch |

So the gradient dtype each engine accepts follows from where it sums.
FSDP2 sums into the shard it steps, so the gradient dtype is the stepped
dtype: float32 under float32 masters, the model dtype without masters.
DeepSpeed's ZeRO-2 and 3 sum in the model dtype, so they take the model
dtype; ZeRO-1 also takes float32. Megatron-Core's DDP accumulates into a
buffer at the model dtype or at float32 (`grad_reduce_in_fp32`), so it
takes those two; Megatron-FSDP keeps its main gradients at the gradient
dtype, so it takes bfloat16 or float32 whatever the model dtype. AutoModel
sums at the weights' storage dtype, so float32 gradients of bfloat16
weights take FSDP2's float32 storage or Megatron-FSDP's float32 main
gradients, on two or more ranks. ShadowSpill takes any.

## Masters, model and optimizer state

| Engine | Model dtype | Masters | Optimizer state | Roundings |
|---|---|---|---|---|
| FSDP2 | `bfloat16` or `float32`; `float16` is refused (it would need loss scaling, which the engine does not do) | The float32 shards are the masters, or `none`, where shards at the model dtype are stepped in place; the parameters a model pins at float32 stay at their dtype beside the shards, in a group of their own under torch's AdamW (a fused step takes one kind) and in the shards' list under torchao's optimizers | AdamW's moments at the stepped dtype (`parameter` and `none` say the same); torchao's 8-bit and 4-bit AdamW keep their state in low-bit blocks (`none`) | `--parameter-rounding stochastic` steps bfloat16 weights in place with torchao's AdamW; no state rounding, no fused narrow-gradient accumulation |
| DeepSpeed | `bfloat16`, `float16` under dynamic loss scaling, or `float32`; every parameter is cast to the model dtype, the pinned ones included, and the row records which | float32, kept by the ZeRO optimizer, or `none`: a float32 model steps its weights themselves, and a bfloat16 model steps a bfloat16 partition of them with bfloat16 gradients (DeepSpeed's `bf16_master_weights_and_grads`) | float32 moments (DeepSpeed's fused AdamW, its CPU AdamW when offloaded); bfloat16 beside bfloat16 values (`bf16_optimizer_states`) | None named: no rounding choices, no fused narrow-gradient accumulation |
| ShadowSpill | Any of the three; `float16` with no loss scaling, as its quickstart takes it | Any dtype, or `none` | Any dtype, or `parameter` | Rounding to nearest unless a case names `stochastic`, ShadowSpill's own training default for bfloat16 moments; fused narrow-gradient accumulation |
| AutoModel | `bfloat16` or `float32`; `float16` is refused (it would need loss scaling); on one rank DDP and Megatron-FSDP cast the model to bfloat16, so a float32 model takes FSDP2 there | float32 or float16 held by Transformer Engine's FusedAdam over bfloat16 storage, on any number of ranks; float32 storage computed at bfloat16 by FSDP2, or Megatron-FSDP's float32 main weights, on two or more ranks; `none` steps the stored weights. A float32 model is its own float32 master and sums its gradients at float32, so float16 masters or bfloat16 gradients of one are refused; FSDP2's float32 gradients of bfloat16 weights come with its float32 storage as their masters, so float16 masters with float32 gradients are refused where FSDP2 shards | torch's AdamW at the stepped dtype; float32, float16 or bfloat16 through FusedAdam (`exp_avg_dtype`, `exp_avg_sq_dtype`). Under Megatron-FSDP, Adam keeps its moments at float32 beside the float32 main weights and at the weights' dtype without them, so moments at another dtype (bfloat16 beside float32 masters, float32 without a master) are refused there | Rounding to nearest; no fused narrow-gradient accumulation |
| Megatron-Core | `bfloat16` or `float32`, under DDP and Megatron-FSDP alike; `float16` is refused (it would need loss scaling) | float32 by default (Megatron's float32 groups, or its distributed optimizer's shards; under DDP a float32 model's stepped float32 parameters are themselves the masters); float16 through its precision-aware optimizer, Transformer Engine's fused Adam holding them; `none` only under Megatron-FSDP, whose compute weights are then the main weights (`megatron_fsdp_main_params_dtype` none) | float32 by default; float16 or bfloat16 through the precision-aware optimizer (`exp_avg_dtype`, `exp_avg_sq_dtype`), which needs the distributed optimizer; with no master, float32 (Transformer Engine's fused Adam) | Rounding to nearest (Transformer Engine's fused Adam); fused narrow-gradient accumulation through `gradient_accumulation_fusion` |

Under DeepCompile, DeepSpeed takes a bfloat16 model with float32 masters
and gradients at the model dtype, or a float32 model stepping its weights
(`deepspeed/README.md` gives the reasons for each refusal).

## What the tiny qualification compares

The qualification matrix of each engine steps a copy of the same model
beside it (`baselines/common/reference.py`), compiled whole with
`torch.compile(fullgraph=True)` as the numerical gate compiles its
reference: unsharded, at the model dtype, with float32 masters or the
stepped dtype, gradients summed at the gradient dtype, PyTorch's AdamW.
Every update compares the loss, and each trained weight's gradient, both
moments and master, where the engine exposes them: FSDP2 all four,
DeepSpeed all four (its moments and masters under DeepCompile, since
DeepSpeed's gradient accessors refuse after a backward its graphs ran),
ShadowSpill its moments and masters (the step consumes the gradients inside
the pool).

The numbers are the repository's numerical gate's
(`qualification/numerical/tolerances.py`), held in `reference.py`, since no
engine but ShadowSpill's imports ShadowSpill, and pinned to the gate's by a
unit test: the loss within 1% and 2e-5; a master with a cosine of at least
0.999, a relative L2 of at most 2.5% and at least 99% of its signs in
agreement; a moment or a gradient, an accumulator of small values summed in
another order, a cosine of at least 0.999 and a relative L2 of at most 5%, the
gate's bounds for optimizer state. Four rules that apply them are this check's
own. It sign-tests weights alone, where the gate sign-tests optimizer state
too, since a moment's or a gradient's near-zero elements change sign with any
rounding. A weight that started at zero, as a LoRA factor's second matrix does,
is still nothing but its optimizer's steps and has no scale of its own: as in
the gate, it agrees when no element is more than two of its steps from the
reference, since an element whose gradient is rounding noise may step the other
way, or not at all where its sum rounds to exactly zero. This check counts in
that weight's own steps, the largest step the reference took in it in the
update and never less than one learning rate, since Adam's first step is one
learning rate only at float32 and up to about 1.1% more with moments rounded to
bfloat16; it allows that beyond one unit in the last place of the dtype the
weight is stepped at, at the element's magnitude, since a weight stepped at
bfloat16 is held rounded on both sides (under float32 masters the allowance is
negligible); and it tells such a weight by its start, where the gate tells it
by every element lying within its steps, in learning rates, of zero: AdamW's
bias-corrected step exceeds one learning rate a little whenever a gradient
grows, which carries a zero-started factor past the gate's line. A routed
expert's LoRA factor sums its gradient over only the tokens the router sends
that expert, and near-ties in the routing choose differently between two
roundings, so this check holds its gradients and moments to bounds of its own,
a cosine of at least 0.995 and a relative L2 of at most 10%; its master keeps
the weight bounds.

The matrices leave out Qwen 3.5 MoE: at the tiny size its MoE gradients
differ between any two runs that round differently, eager and compiled
PyTorch alike, by more than the bounds allow, most likely its router's
top-k choosing differently on near-ties; Qwen 3.5's dense model, with the
same pinned parameters, takes its cases. Each matrix also trains LoRA at
bfloat16 weights stepped in place, the contract round 1's LoRA experiments
run, on the llama3, qwen35 and qwen3moe families.

The Megatron-Core and AutoModel engines train NVIDIA's implementation of
the architecture, so their matrices step no reference copy: each case's
first update checks the dtypes it holds its values at against the
mechanism the contract maps to, and a case that holds another fails.
Megatron-Core's check reads the weights, the accumulated gradients, the
masters and the moments; AutoModel's reads the weights, the gradients, the
masters FusedAdam holds and the moments, and under Megatron-FSDP its main
weights, main gradients and moments. Without a reference to agree with,
the AutoModel matrix keeps Qwen 3.5 MoE.

The copy creates gradients as PyTorch's autograd does, at the model dtype,
each microbatch's added after it is written, so the ShadowSpill
qualification matrix creates and sums its gradients the same way by
default, at the model dtype (`grad_dtype` none) with fused narrow-gradient
accumulation off: both then sum the same rounded values, where float32
gradients from the captured graphs would be finer than the copy on every
element whose microbatches cancel. Its fused case (bfloat16 gradients with
`--fuse-narrow-gradient-accumulation`) is the comparison's single
departure: there the multiply rounds each later microbatch's sum once,
where the copy rounds the product and then the sum.

# Precision across the harnesses

Every harness takes one precision contract, `baselines/common/precision.py`,
with the names ShadowSpill's quickstart uses: the dtype the model computes
at, the dtype of the masters, the dtype gradients are summed at, the dtype
of the optimizer state, the two roundings, and whether a gradient
accumulation rounds once. Each harness states what it honors as
`Capabilities` and in its rules. A request it cannot honor is refused before
launch, with the reason, and is never substituted. An experiment runs all
its harnesses at one contract, named once in its spec
([README.md](README.md#experiments)). This page gives what each value means
and what each harness does with it.

## The contract

| Value | Flag | Meaning | Default |
|---|---|---|---|
| Model dtype | `--model-dtype {bfloat16,float16,float32}` | What the weights the forward and backward read, and the activations, are kept and computed at | `bfloat16` |
| Master dtype | `--master-dtype {float32,bfloat16,float16,none}` | What a trained weight's master copy is kept and stepped at; `none` steps the model's weights themselves | `float32` |
| Gradient dtype | `--grad-dtype {float32,bfloat16,float16,none}` | What gradients are summed at over a step, across its microbatches and across ranks; `none` is the model dtype | the model dtype (FSDP2: `float32`) |
| Optimizer state dtype | `--opt-state-dtype {float32,bfloat16,float16,parameter,none}` | What AdamW keeps its two moments at; `parameter` is the dtype of what it steps, `none` the optimizer's own default | `float32` |
| Parameter rounding | `--parameter-rounding {nearest,stochastic,none}` | How the optimizer rounds a weight it steps at a dtype narrower than its update; `none` is the harness's own, rounding to nearest everywhere | `none` |
| Optimizer state rounding | `--opt-state-rounding {nearest,stochastic,none}` | How the optimizer rounds a moment it stores narrower than float32 | `none` |
| Round accumulation once | `--[no-]round-accumulation-once` | A multiply adds its product into a running gradient kept narrower than it computes, rounding once instead of rounding the product and then the sum | off |

The stepped dtype is the master dtype when masters are kept, the model
dtype otherwise. The default is bfloat16 compute and gradients with float32
masters and moments. FSDP2 cannot sum gradients at the model dtype under
float32 masters (below), so its own default gradient dtype is float32.

## Gradients: creation, accumulation and reduction

A weight's gradient is made by each microbatch's backward, summed over the
step's microbatches on each rank (gradient accumulation), and summed over
the data-parallel ranks (the reduction). The objective makes each
microbatch's loss its share of the step's mean, so these sums are the
step's gradient; a dtype applies at each of the three places, and a harness
that reduces after every microbatch interleaves the last two.

**Creation.** Under PyTorch's autograd, eager or compiled, a weight's
gradient has the weight's own dtype: PyTorch casts whatever an operation
returns to it. So FSDP2, DeepSpeed and the reference create every
microbatch's gradients at the dtype each weight computes at, the model dtype
(or float32 for a parameter a model pins there), whatever the gradient
dtype. ShadowSpill runs captured graphs that write gradients into the step's
own buffers instead: mlops's operations that sum a weight's gradient over
rows (norms, embeddings, heads, experts, LoRA factors) hand it over at the
gradient dtype, with no rounding to the model dtype first.

**Accumulation and reduction, per harness:**

| Harness | Summed over microbatches | Reduced across ranks | When it reduces |
|---|---|---|---|
| FSDP2 | Into the shard's gradient, at the shard's dtype (the stepped dtype) with `--gradient-sync per-microbatch`; with `last-microbatch`, each microbatch's unsharded gradient is cast to `reduce_dtype` and summed there | A reduce-scatter at `MixedPrecisionPolicy.reduce_dtype`, the gradient dtype; the parameters a model pins at float32 are averaged by hand | After every microbatch (`per-microbatch`), or once after the last (`last-microbatch`) |
| DeepSpeed ZeRO-1 | In each weight's gradient at the model dtype; at float32 when `--grad-dtype float32` asks for it (`data_types.grad_accum_dtype`; under a bfloat16 model, DeepSpeed's BF16 optimizer) | At the model dtype; at float32 with `--grad-dtype float32` (`communication_data_type`) | Once, at the update's last microbatch |
| DeepSpeed ZeRO-2, ZeRO-3 | In the gradient partitions, at the model dtype | A reduce-scatter at the model dtype, overlapped with backward in buckets (`--overlap-comm`, `--reduce-bucket-size`) | After every microbatch |
| ShadowSpill | Into the step's running gradient at the gradient dtype; with `--round-accumulation-once` a multiply adds its product into a narrower running gradient as it writes it | A sum at the gradient dtype: an all-reduce, or a reduce-scatter onto the rank that owns each slice of the optimizer state | Once, after the last microbatch |
| Reference | At the gradient dtype | An all-reduce at the gradient dtype | After every microbatch |

So the gradient dtype each harness accepts follows from where it sums.
FSDP2 sums into the shard it steps, so the gradient dtype is the stepped
dtype: float32 under float32 masters, the model dtype without masters.
DeepSpeed's ZeRO-2 and 3 sum in the model dtype, so they take the model
dtype; ZeRO-1 also takes float32. ShadowSpill takes any.

## Masters, model and optimizer state

| Harness | Model dtype | Masters | Optimizer state | Roundings |
|---|---|---|---|---|
| FSDP2 | `bfloat16` or `float32`; `float16` is refused (it would need loss scaling, which the harness does not do) | The float32 shards are the masters, or `none`, where shards at the model dtype are stepped in place; the parameters a model pins at float32 stay at their dtype beside the shards and step in their own group | AdamW's moments at the stepped dtype (`parameter` and `none` say the same); torchao's 8-bit and 4-bit AdamW keep their state in low-bit blocks (`none`) | `--parameter-rounding stochastic` steps bfloat16 weights in place with torchao's AdamW; no state rounding, no rounding once |
| DeepSpeed | `bfloat16`, `float16` under dynamic loss scaling, or `float32`; every parameter is cast to the model dtype, the pinned ones included, and the row records which | float32, kept by the ZeRO optimizer, or `none`: a float32 model steps its weights themselves, and a bfloat16 model steps a bfloat16 partition of them with bfloat16 gradients (DeepSpeed's `bf16_master_weights_and_grads`) | float32 moments (DeepSpeed's fused AdamW, its CPU AdamW when offloaded); bfloat16 beside bfloat16 values (`bf16_optimizer_states`) | None named: no rounding choices, no rounding once |
| ShadowSpill | Any of the three; `float16` with no loss scaling, as its quickstart takes it | Any dtype, or `none` | Any dtype, or `parameter` | Rounding to nearest unless a case names `stochastic`, ShadowSpill's own training default for bfloat16 moments; rounding an accumulation once |

Under DeepCompile, DeepSpeed takes a bfloat16 model with float32 masters
and gradients at the model dtype, or a float32 model stepping its weights
(`deepspeed/README.md` gives the reasons for each refusal).

## What the tiny qualification compares

The qualification matrix of each harness steps a copy of the same model
beside it (`baselines/common/reference.py`), compiled whole with
`torch.compile(fullgraph=True)` as the numerical gate compiles its
reference: unsharded, at the model dtype, with float32 masters or the
stepped dtype, gradients summed at the gradient dtype, PyTorch's AdamW.
Every update compares the loss, and each trained weight's gradient, both
moments and master, where the harness exposes them: FSDP2 all four,
DeepSpeed all four (its moments and masters under DeepCompile, since
DeepSpeed's gradient accessors refuse after a backward its graphs ran),
ShadowSpill its moments and masters (the step consumes the gradients inside
the pool). The bounds are the repository's numerical gate's
(`qualification/numerical/tolerances.py`): the loss within 1% and 2e-5; a
master with a cosine of at least 0.999, a relative L2 of at most 2.5% and at
least 99% of its signs in agreement; a moment or a gradient, an accumulator
of small values summed in another order, a cosine of at least 0.999 and a
relative L2 of at most 5% with no sign test, since its near-zero elements
change sign with any rounding. A weight that started at zero, as a LoRA
factor's second matrix does, is still nothing but its optimizer's steps and
has no scale of its own: it agrees when no element is more than two learning
rates from the reference, since an element whose gradient is rounding noise
may step the other way, or not at all where its sum rounds to exactly zero.
The gate tells such a weight by every element lying within its steps, in
learning rates, of zero; AdamW's bias-corrected step exceeds one learning
rate a little whenever a gradient grows, so the check here tells it by its
start instead. The matrices leave out Qwen 3.5 MoE: at the tiny size its MoE
gradients differ by 4 to 13% between any two runs that round differently,
eager and compiled PyTorch alike, more than the bounds allow and most likely
its router's top-k choosing differently on near-ties; Qwen 3.5's dense
model, with the same pinned parameters, takes its cases.

The copy creates gradients as PyTorch's autograd does, at the model dtype, so
the ShadowSpill qualification creates its gradients there too: both then
sum the same rounded values, where float32 gradients from the captured
graphs would be finer than the copy on every element whose microbatches
cancel.

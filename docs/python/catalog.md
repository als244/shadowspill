# Models, optimizers and objectives

Choose an installed definition by name. Built-ins and custom definitions use the
same `Model`, `Optimizer` and `Objective` settings.

## Model architectures

These architectures are supplied by `shadowspill-workloads`:

| Name | Architecture |
| --- | --- |
| `llama3` | Llama decoder with grouped-query attention. |
| `qwen3moe` | Qwen3 mixture-of-experts decoder. |
| `qwen35moe` | Qwen3.5 mixture-of-experts decoder with hybrid attention. |
| `olmoe` | OLMoE mixture-of-experts decoder. |
| `regression` | Small feed-forward model used in the [split-update example](advanced/training-workflow.md). |

For a text model, choose a preset and override architecture settings as needed:

```python
from shadowspill_client import Model, Weights

model = Model(
    "llama3",
    config={"preset": "tiny", "architecture": {"vocab_size": 32000}},
    initialize=Weights.random(seed=0),
)
```

The text models default to the small `tiny` preset. Initial weights are a separate
choice: use `Weights.random(seed=...)` to train from scratch, or
`Model.from_pretrained("/models/checkpoint")` for a supported checkpoint on the
server. Configuration, precision and initialization details are in the
[definition reference](advanced/definitions.md#supplied-definitions).

## Optimizers

| Name | Update rule | Per-update controls |
| --- | --- | --- |
| `adamw` | AdamW, supplied by the engine's MLOps integration. | `lr`, `beta1`, `beta2`, `eps`, `weight_decay` |

Choose `Optimizer("adamw")` during setup. Set the learning rate when applying an
update:

```python
training.step(batch, optim={"lr": 0.0003}).result()
# Or, after accumulating gradients with forward_backward:
training.optim_step(optim={"lr": 0.0003}).result()
```

Optimizer precision controls gradients, moments and optional master weights;
see the [client setup example](client.md#create-a-training-run). Optional
`defaults` supply omitted controls. A per-call value never changes those defaults.

## Objectives

| Name | Use |
| --- | --- |
| `cross_entropy` | Weighted next-token cross entropy for pretraining or supervised fine-tuning. |
| `clipped_policy` | Clipped policy objective using caller-supplied old logprobs and advantages. |
| `moe_balance` | Router load-balancing loss for a model exposing router statistics. |
| `weighted_squared_error` | Weighted squared error for the supplied regression model. |

Combine compatible objectives explicitly. For example, Qwen and OLMoE can use
cross entropy with a secondary router-balancing loss:

```python
from shadowspill_client import Objective

objective = Objective.sum(
    {
        "tokens": Objective("cross_entropy"),
        "balance": Objective("moe_balance"),
    },
    weights={"balance": 0.01},
)
```

Padding is excluded from balancing by default. Balancing statistics are computed
per physical microbatch, so splitting a batch can change that term. Token losses
are weighted sums; choose input weights or `grad_scale` for your normalization.
See [objective settings and inputs](advanced/definitions.md#text-objectives).

## LoRA and trainable parameters

LoRA is a model setting. For example, with an existing Llama checkpoint on the server:

```python
from shadowspill_client import LoRA, Model

model = Model.from_pretrained(
    "/models/llama",
    lora=LoRA(rank=8, alpha=16, targets=["blocks.*.attn.wq", "blocks.*.attn.wv"]),
)
```

`targets` matches module names. Omit it to use the architecture's default targets.
`alpha` defaults to `rank`; `dropout` defaults to zero. The supplied text LoRA
implementations require zero dropout. LoRA freezes base parameters and trains
the added adapters by default.

Use `trainable` to select parameter names or glob patterns and freeze everything
else. For example, `Model.from_pretrained("/models/llama", trainable=["lm_head.weight"])`
trains only the output head. To train adapters and the head together, add
`trainable=["*.lora_*", "lm_head.weight"]` to the LoRA example.

Selection happens after adapters are added. An explicit list replaces the default
selection, so include adapter patterns when you want to train them. Unmatched
patterns are errors; tied parameter names still refer to one shared parameter.
Choose this during setup, before preparation and the training loop.

## Custom definitions

Install your package on the server and register its definitions, then select
`Model("my_model")`, `Optimizer("my_optimizer")` or `Objective("my_objective")`.
Your client needs no model or optimizer implementation installed. The
[Advanced guide](advanced/definitions.md) covers authoring and registering all three.

# Training objectives

Choose an objective during preparation; submit its data with each batch. The
client owns tokenization, rollout generation, reward calculation and teacher data.
The server runs the prepared model, differentiates the objective and updates its
trainable parameters. These choices work with full-parameter training or LoRA.

| Task | Objective | Data beyond token sequences |
| --- | --- | --- |
| Pretraining | `cross_entropy` | None |
| Supervised fine-tuning (SFT) | `cross_entropy` | Assistant-target masks |
| GRPO | `clipped_policy` | Response masks, old logprobs and group-relative advantages; optional reference logprobs |
| PPO actor update | `clipped_policy` | Response masks, old logprobs and caller-computed per-token advantages |
| Response distillation | `cross_entropy` | Teacher-generated responses and their target masks |

## Prepare the objective

Use the ordinary [client setup](../client.md#create-a-training-run), with the
chosen `objective` in the creation call:

```python
training = service.create_training_client(
    model=model,
    objective=objective,
    optimizer=optimizer,
    batch=batch_config,
    memory=memory,
).result()
```

Here `model`, `optimizer`, `batch_config` and `memory` are the same named settings
as in the basic example. Changing the objective's inputs or computation requires
preparation. Learning rate and declared scalar loss controls remain per-call
settings.

## Cross entropy for pretraining

```python
from shadowspill_client import Objective
from shadowspill_client.text import text_batch

objective = Objective("cross_entropy")
# Prepare training with this objective, then for each update:
batch = text_batch(token_sequences)
result = training.step(batch, optim={"lr": 0.0003}).result()
```

Each sequence contains original token IDs. `text_batch` creates inputs `tokens[:-1]`
and targets `tokens[1:]`, then supplies weights for one mean over **all target
tokens in the global batch**. Different-length sequences receive no extra
per-sequence normalization. Service-added padding does not contribute to loss.

## Masked cross entropy for SFT

Use the same objective, but mark the assistant tokens in the original sequences:

```python
objective = Objective("cross_entropy")
batch = text_batch(conversation_tokens, loss_masks=assistant_masks)
result = training.step(batch, optim={"lr": 0.0001}).result()
```

Each mask has the same length as its original token sequence. The helper shifts
the mask with the targets, excludes prompt tokens from the loss, and divides by
the total number of supervised targets across the batch. Prompt tokens still
provide model context. Use the checkpoint's chat template when preparing tokens;
the [UltraChat example](../../examples/client-training.md#prepare-ultrachat)
shows corpus preparation.

## Policy inputs shared by GRPO and PPO

Both use the clipped policy surrogate from
[PPO](https://arxiv.org/abs/1707.06347). A rollout below is a dictionary containing:

| Field | Alignment |
| --- | --- |
| `tokens` | Complete prompt plus generated response, length `L` |
| `response_mask` | Boolean mask of length `L`, true on original response tokens |
| `old_logprobs` | Float32 logprob of each target `tokens[1:]` under the rollout policy, length `L-1` |
| `reference_logprobs` | Optional logprobs of those same targets under a fixed reference policy, length `L-1` |

The old logprobs must describe the behavior policy that generated the rollout,
including any sampling-policy transformations required by your algorithm. Keep
them fixed while reusing that rollout for updates. Prompt positions may carry
zero-filled auxiliary values because `response_mask` excludes them.

This small caller helper constructs already aligned `Datum` fields. `advantages`
contains one float32 vector of length `L-1` per rollout:

```python
import numpy as np
from shadowspill_client import Datum

def policy_batch(rollouts, advantages, *, sequence_mean=False, reference=False):
    masks = [np.asarray(row["response_mask"], dtype=bool)[1:] for row in rollouts]
    counts = [int(mask.sum()) for mask in masks]
    if not counts or min(counts) == 0:
        raise ValueError("each rollout needs at least one response target")
    batch = []
    for row, mask, count, advantage in zip(rollouts, masks, counts, advantages, strict=True):
        tokens = np.asarray(row["tokens"], dtype=np.int64)
        denominator = len(rollouts) * count if sequence_mean else sum(counts)
        fields = {
            "target_tokens": tokens[1:],
            "response_mask": mask,
            "weights": np.full(len(mask), 1 / denominator, dtype=np.float32),
            "old_logprobs": np.asarray(row["old_logprobs"], dtype=np.float32),
            "advantages": np.asarray(advantage, dtype=np.float32),
        }
        if reference:
            fields["reference_logprobs"] = np.asarray(row["reference_logprobs"], dtype=np.float32)
        batch.append(Datum(inputs={"tokens": tokens[:-1]}, loss_inputs=fields))
    return batch
```

Direct `Datum` fields are already aligned: the service does not shift them again.
The objective sums these explicit weights. Physical packing and distribution do
not introduce another mean.

## GRPO

Generate several responses to each prompt and score them. `groups` below is a list
of equally sized response groups; each rollout also has a scalar `reward`. This
example uses a mean over responses and then over each response's tokens, with
advantages standardized within its prompt group, following the outcome-reward
formulation in [DeepSeekMath](https://arxiv.org/abs/2402.03300).

```python
objective = Objective("clipped_policy", config={"reference_penalty": "k3"})
# Prepare training with this objective.
rollouts, advantages = [], []
for group in groups:
    rewards = np.asarray([row["reward"] for row in group], dtype=np.float32)
    relative = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
    for row, advantage in zip(group, relative, strict=True):
        rollouts.append(row)
        advantages.append(np.full(len(row["tokens"]) - 1, advantage, dtype=np.float32))

batch = policy_batch(rollouts, advantages, sequence_mean=True, reference=True)
result = training.step(
    batch, optim={"lr": 0.000001},
    loss_params={"clip_ratio": 0.2, "reference_beta": 0.01},
).result()
```

The optional sampled reference penalty is `exp(d) - d - 1`, with
`d = reference_logprob - current_logprob`, on response targets. Its coefficient is
`reference_beta`. To omit it, prepare `Objective("clipped_policy")`, omit reference
inputs, and pass only `clip_ratio`. Rewards with no within-group variation produce
zero policy advantages; an enabled reference penalty can still contribute.

Compute group statistics before dividing the global batch among workers or
backward calls. Once advantages are supplied, responses are independent model
inputs and can be packed separately.

## PPO actor update

Compute token advantages in your rollout/value-estimation code, for example using
generalized advantage estimation. Then submit the actor update:

```python
objective = Objective("clipped_policy")
# Prepare training with this objective.
batch = policy_batch(rollouts, token_advantages)
result = training.step(
    batch, optim={"lr": 0.000001}, loss_params={"clip_ratio": 0.2},
).result()
```

This example uses a global response-token mean. `clipped_policy` is the actor
objective; it does not generate rollouts, compute rewards/advantages or train a
value head. A value objective needs a model exposing value predictions and a
compatible custom objective. Include reward penalties once in the formulation
you choose; do not accidentally add the same reference penalty again here.

## Response distillation

Generate responses with a teacher, render them using the student's chat template,
and tokenize with the **student's tokenizer**. Train the student on the teacher's
response tokens using the same masked cross entropy:

```python
objective = Objective("cross_entropy")
# Prepare the student training client with this objective.
batch = text_batch(teacher_conversation_tokens, loss_masks=teacher_response_masks)
result = training.step(batch, optim={"lr": 0.0001}).result()
```

This transfers teacher-generated sequences, as in
[sequence-level distillation](https://arxiv.org/abs/1606.07947). The teacher may run
in a separate inference service or produce an offline dataset. Teacher and student
need not share token IDs for this recipe because the responses are retokenized.

Soft-target distillation instead needs teacher distributions aligned to the
student vocabulary and a model/output contract exposing the needed student
distribution. Implement that as a [custom objective](custom-definitions.md), with
explicit temperature, reduction and masking. The built-in `cross_entropy` recipe
above consumes token targets, not full teacher logits.

## Split updates and secondary terms

Construct and normalize a global batch once, then slice the resulting `Datum`
list if using `forward_backward` and `optim_step`. Prepare those operations first
as shown in the [training workflow](training-workflow.md). Recomputing weights
separately for each slice changes the objective.

Use `Objective.sum` to combine compatible declared objectives; the
[catalog](../catalog.md#objectives) shows cross entropy plus MoE balancing. With a
policy objective, configure balancing with `participation_field="response_mask"`
so both terms share the same update-participation contract. Router masks separately
control balancing tokens; padding is excluded by default. Each term retains its
own reduction, so select its coefficient explicitly.

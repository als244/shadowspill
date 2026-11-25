# Workload organization and adding a model family

`shadowspill_workloads` is the optional built-in model library. The engine runs
installed model components through the same hooks whether they come from this
library or an external package. An external integration can follow the
[component guide](../python/advanced/components.md) without editing this repository.

## Package layout

```text
packages/workloads/
  src/shadowspill_workloads/
    models/
      families.py            # The list of built-in family packages
      registration.py        # The small family declaration type
      <family>/
        registration.py      # Presets, reader, checkpoint IDs, training hook
        config.py            # Architecture and named shapes
        implementations/     # Model computation
        checkpoint.py        # Checkpoint names and tensor layouts
        lora.py              # Architecture-specific adapter targets
        training.py          # Family settings/hooks, when needed
        README.md
      shared/                # Reusable layers and integration shared by families
        glm/                 # Shared GLM computation, loading and setup hooks
      catalog/               # Lookups derived from family declarations
    components/              # Shared registration, objective and setup helpers
      decoder/               # Shared packed decoder registration and setup hooks
    inputs/                  # Schemas, processors and input adapters
    objectives/              # Reusable objective computations
    initialization/          # Bounded weight loading and parameter generation
    lora/                    # Shared adapter application and targeting
    policies/                # Chunking, retention and execution choices
    integrations/            # MLOps operator facts as generic build options
  scripts/
    setup_mlops.sh           # Installs MLOps: a checkout beside the repository, else the pin
```

The catalog's models and its AdamW component use MLOps, so this package owns
MLOps: its revision is pinned in `scripts/setup_mlops.sh`, which the repository's
setup runs. The engine neither imports nor installs it.

Large modules can use a package instead of a single file. Keep architecture-specific
computation, checkpoint mappings, initialization and setup hooks beside the model.
Code shared by related families belongs under `models/shared/<architecture>/`;
both GLM families use `models/shared/glm/`. Common registration, input, loading
and policy mechanisms stay in the top-level workload packages. Generic planning
and execution belong in the engine.

Put application loops in `examples/training/` and benchmark drivers in
`tools/planning/`. A model family supplies declarations and computation, not a
training loop or its own service startup path. This includes both GLM families;
their hooks reuse decoder setup and supply their checkpoint storage, attention
requirements and provider contracts. Family READMEs describe architecture details;
client configuration and runtime commands stay in the shared guides.

## Add a built-in family

1. Create `models/<family>/config.py` and `implementations/`. A config describes
   architecture dimensions; `PRESETS` gives named starting shapes. Each
   implementation exports `Model`. Reuse existing layers where their mathematics
   match, and keep an independent small reference for numerical checks. Configs
   must round-trip through JSON without importing Torch or model implementations;
   store dtype names such as `"bfloat16"` and resolve them in model construction.
2. Add `checkpoint.py` for pretrained weight mappings and bounded copying.
   Add `lora.py` if adapter targeting depends on the architecture. Keep parameter
   storage precision explicit. Declare selective defaults in
   `FAMILY.parameter_dtypes` and pass them into the public `ModelComponent`;
   shared setup applies the [dtype policy](../python/advanced/precision.md).
3. Supply a public model-registration hook. Reuse the existing input schema,
   adapter, objectives and preparation hooks when they describe the model's inputs
   correctly. A family-specific `training.py` supplies settings and hooks; it
   does not implement another training loop. The
   [Qwen3 settings](../../packages/workloads/src/shadowspill_workloads/models/qwen3/training.py)
   and [OLMoE settings](../../packages/workloads/src/shadowspill_workloads/models/olmoe/training.py)
   reuse [shared decoder hooks](../../packages/workloads/src/shadowspill_workloads/components/decoder).
   They supply the family, catalog and architecture-specific checks. A different
   constructor can be supplied directly, as in
   [Llama](../../packages/workloads/src/shadowspill_workloads/models/llama3/training.py).
   Use the public component hooks for architectures with different input or
   initialization requirements; do not force them into decoder-specific settings.
4. Declare `FAMILY` in `registration.py`. The
   [Qwen3 declaration](../../packages/workloads/src/shadowspill_workloads/models/qwen3/registration.py)
   shows preset descriptions, checkpoint reader, pretrained identifiers,
   initialization choices and the installed registration hook in one place.
   Add the package name once in
   [models/families.py](../../packages/workloads/src/shadowspill_workloads/models/families.py).
   Catalog lookups, pretrained identification and component loading derive from
   these declarations; do not add another family map elsewhere.
5. Include the family declaration and all code used by its hooks in its
   component source identity. Metadata inspection must not import Torch,
   kernels or model implementations. Import those inside worker hooks.
6. Check configuration resolution, meta construction, checkpoint tensor mappings,
   bounded initialization and losses/gradients against the independent reference.
   Then exercise public `step()`, split `forward_backward()`/`optim_step()` and
   checkpoint continuation. Test trainable selection and LoRA when supplied.

### Output head for a packed training decoder

Expose the final hidden states and the head through `token_features`. For a
decoder with `hidden(tokens, sequence_lengths)` and `lm_head`, the shared
`decoder_token_features` method already does this. Construct the head with
`mlops.modules.LanguageModelHead(hidden_size, vocab_size)`. It supplies
ordinary logits, the chunked scalar objective and the corresponding head LoRA
conversion. Tied embeddings can assign the embedding parameter to the head's
`weight`.

A decoder with a softcap and FP32 projection uses the same module.

```python
from mlops.modules import LanguageModelHead

self.lm_head = LanguageModelHead(
    config.d_model, config.vocab_size,
    fp32_projection=True, logit_softcap=config.logit_cap,
)
```

The projection settings apply to prediction, scalar loss and head LoRA. A
different output transformation needs a matching bounded `objective` method.
Packed decoder construction checks this contract before capture. The scalar
objective must consume each vocabulary block and its derivative before moving
to the next block. Chunking a differentiable list of per-token logprobs alone
can leave every block alive until backward.

The catalog head regression compares the scalar loss with ordinary logits and
inspects saved tensors on each small decoder, with and without head adapters.
On a representative compiled step, inspect the terminal saved-forward outputs.
A frozen head with upstream gradients ordinarily returns a hidden-width
gradient plus the declared metrics. A trainable weight or adapter also adds its
parameter-shaped gradients. Token-by-vocabulary blocks on that boundary merit
investigation. For example, a 200,058-entry vocabulary at 16,384 tokens costs
12.21 GiB if backward keeps one FP32 value per vocabulary entry and token.
Dividing this into 198 output tensors leaves the total unchanged. A frozen
2,304-wide BF16 head needs only 72 MiB for its hidden gradient at that shape.
[Catalog head tests](../../packages/workloads/tests/workload_tests/objectives/test_token_head.py)

## Input modalities and initialization

Input modality is a model/schema concern. The model handles its computation;
processors and input adapters describe tensors, alignment and padding; objectives
define losses and reductions. The service consumes those declarations and does
not choose a separate trainer for text, images or other modalities.

Weight initialization does not consume examples. `ParameterInitialization`
records reusable parameter-distribution choices for the supplied decoder loader;
it is optional family metadata, not an engine requirement. Other architectures
can supply their own bounded initialization hook. A multimodal model can reuse
decoder initialization for its language component and provide the appropriate
initialization or checkpoint mappings for its encoder.

Use text-specific schemas only for actual token inputs. An image/text model also
needs declarations for its image tensors, position metadata and alignment; naming
its checkpoint architecture does not supply those inputs automatically.

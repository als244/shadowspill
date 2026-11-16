# Shared text application

This package handles token-corpus training choices independently of model
selection. It imports only the lightweight client and shared application helpers.

[settings.py](settings.py) supplies batch limits, physical size/chunk candidates,
profiling/search controls, budgets, schedules and evaluation cadence. Its request
builder accepts a caller's `Model`, `Objective` and parallelism declaration.
An application can extend these settings with model configuration and defaults.
The token cap includes every scheduled input position, including padding.

[preparation.py](preparation.py) uses metadata-only previews to bind explicitly
declared model configuration values to objective controls and refine the existing
PressureFit search. It retains resolved weight-content pins. A model need not
supply any objective defaults: for example, OLMoE can use an explicitly weighted
MoE balancing objective, while Qwen applications can take that coefficient from
checkpoint metadata. Padding exclusion remains the objective's default.

[cli.py](cli.py) handles corpus loading, vocabulary validation, optional logging,
and the shared setup/runtime lifecycle. Model applications supply their settings
and metadata resolver. The runtime loop receives an already initialized client.

The Qwen applications now contain model/preset choices, supported precision/LoRA
combinations, model parameter/input names, and checkpoint metadata bindings.
Their batch/preparation configuration and training lifecycle live here and in
[application](../application/README.md). Llama and regression use that same
application lifecycle with their existing schedule and evaluation policies.

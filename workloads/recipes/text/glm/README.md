# Shared GLM recipes

Model definitions and checkpoint tensor mappings live in
[the model families](../../../models/). This package owns their caller-side policy.

- `quickstart.py`: GLM-5.3 and Flash factories for the ordinary phase runner.
- `factory.py`, `model.py`: meta model construction, precision and build options.
- `initialize.py`: bounded checkpoint loading or reproducible random values.
- `inputs.py`: synthetic inputs across candidate batch sizes.
- `data.py`: real-text batches, objectives and the fixed-batch learning check.
- `partition.py`: optional per-layer partition policy.
- `flash_trace.py`: Flash inputs for the generic stage tracer.

The [Flash LoRA training recipe](../glm53_flash_lora/README.md) uses these helpers
for checkpoint-backed training. No model implementation imports this package.

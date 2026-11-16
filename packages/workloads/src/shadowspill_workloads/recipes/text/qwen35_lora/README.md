# Qwen3.5 MoE LoRA

Use the [public-client application](../../../../../../../examples/training/qwen35_lora.md)
or [Python example](../../../../../../../examples/qwen35_lora.py). Its lightweight
client owns token batches and the training loop. Separate setup resolves the base
checkpoint, execution policies and worker initialization through the installed
model definitions. It supports whole steps, explicit backward accumulation,
evaluation and exact checkpoint resume.

The installed Qwen3.5 definition and shared bounded initializer own model-specific
construction and weight mappings. The remaining adapter in this directory serves
the old generic recipe caller while that caller is migrated; the standalone Qwen
phase CLI and its forwarding API have been removed.

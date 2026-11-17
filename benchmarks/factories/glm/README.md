# GLM phase benchmarks

`quickstart.py` supplies GLM-5.3 and Flash factories for the
[phase benchmark runner](../../quickstart.md). `factory.py` composes the model,
precision and bounded initializer; `inputs.py` supplies reproducible synthetic
inputs across microbatch sizes. Architecture definitions and checkpoint mappings
remain in the installed model families.

For real training, use the [public client](../../../docs/python/client.md).

# Locating nondeterminism

```bash
python -m qualification.nondeterminism llama3 --model-implementation mlops
```

The probe runs fixed inputs twice and compares the objective, module forward
outputs and incoming gradients bitwise. It reports the first divergence in
execution order and exits nonzero when the step is not reproducible.

Use the same `--seed`, `--model-config`, `--data-geometry`, `--case-factory` and
`--case-option` settings as the failing numerical case. `--no-modules` omits
module hooks to reduce memory and compares the objective/parameter gradients.
`--deterministic` requests ordered kernels where MLOps provides them.

A divergence that persists with ordered kernels needs investigation outside that
provider setting. Keep compiler caches isolated across deterministic-policy
changes, since the policy can affect generated code without a runtime guard.
The [numerical gate](../numerical/README.md) separately checks bounded error and
checkpoint replay.

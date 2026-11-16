# Regression benchmark

The training application now uses the [public service client](../../../../../../examples/training/regression/README.md), with separate setup and runtime modules.

## Single-GPU quickstart benchmark

`benchmark.py` supplies the model, synthetic data and bounded initializer to the
shared quickstart driver. It sweeps two microbatch sizes and the requested GPU
budgets using the same public phase APIs:

```bash
python -m benchmarks.quickstart \
  --factory shadowspill_workloads.recipes.regression.benchmark:experiment \
  --search-budget-gib 2 3 --spill-gib 0.125 --output-dir runs/regression-benchmark
```

See the [quickstart guide](../../../../../../benchmarks/quickstart.md) for independent
phases, CPU replanning and saved results.

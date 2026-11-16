# Regression client example

Run the application against an existing authenticated service. The client needs
NumPy and `shadowspill-client`; the server needs the engine, MLOps and installed
workload definitions. See the [service guide](../../../docs/python/client.md)
for server startup and credentials.

```bash
export SHADOWSPILL_TOKEN=...  # Same owner credential as the service.
python -m examples.regression_training --ssh chicago --server-port 8000 \
  --run-dir runs/regression --ranks 1 --steps 3 --checkpoint-every 1
```

Use `--ranks 2` on a service allocated two GPUs to shard weights, gradients and
optimizer state through the existing generic DP interface. The default is two
workers, eight samples per worker and ten updates. The caller supplies a single
global list; the engine distributes distinct samples and sums contributions.
Each sample has weight `1 / (global_samples * 8)`, preserving the mean over all
output elements. `--backward-pieces 3` submits unequal contributions followed by
one `optim_step()` with the same complete-update normalization. The model's
integer buffer counts training forwards, so split execution can increment it
more often than a whole step.

[setup.py](setup.py) supplies portable settings and input/parallelism declarations.
[Shared setup](../application/setup.py) owns client creation and resume validation.
[data.py](data.py) creates deterministic NumPy
PCG64 batches indexed by update. [loop.py](loop.py) receives a ready client and
owns only inputs, training calls and completed results. Model code and bounded
initialization live in the [installed catalog](../../../packages/workloads/src/shadowspill_workloads/models/regression).
The client never imports them or initializes workers.

The local run directory retains reconnect IDs before creation completes, scalar
metrics, a downloaded HTML report and checkpoint reference JSON files. Complete
checkpoint payloads stay in server storage until explicitly downloaded. Resume
with the same service storage and owner, using a fresh local directory:

```bash
python -m examples.regression_training --ssh chicago --server-port 8000 \
  --run-dir runs/regression-resumed \
  --resume runs/regression/checkpoints/step-00000001.json
```

Resume verifies saved settings, generator identity, NumPy version and update
cursor before creating workers. It loads saved Plans and native training state;
there is no automatic retry of updates. Optional `--wandb-project NAME` and
`--wandb-mode offline` export the recorded scalar history after training. An
export failure leaves the authoritative local/service results intact.

This replaces the old per-worker regression Trainer recipe. Synthetic samples
now use NumPy PCG64 instead of Torch's generator, so historical synthetic values
are different. Initialization, FP32 precision, AdamW defaults and loss
normalization are preserved. The old `--phase` and `--no-plots` CLI flags are
removed. The existing public build/profile/plan/execute APIs and the
[separate benchmark](../../../packages/workloads/src/shadowspill_workloads/recipes/regression/README.md)
still support saved phase artifacts and CPU replanning.

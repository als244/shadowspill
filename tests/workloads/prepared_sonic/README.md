# Prepared Sonic saved-Program regression

Run the optional real-provider integration from the repository root:

```bash
python -m pytest -m slow tests/workloads/test_prepared_sonic.py
```

It requires the CUDA native backend and MLOps with the pinned SonicMoE/Quack
extras. It runs two BF16 rank-32 cases: recompute with fresh/continuing/fresh
gradients, and saved activations with three AdamW updates. Both use shared
nonzero LoRA factors twice, two empty experts, batch 2, sequence lengths 65/48,
and nested 33/17/9 execution chunks. It calls the actual Sonic provider directly;
it does not expand that provider's automatic hardware dispatch.

Each phase runs in a fresh, bounded process. An independent CPU implementation
uses ordinary dense products, top-k, softmax and autograd. Public builds provide
the provider's chunk rules, gradient destinations and `OperatorPreparation`.
Cold profile, execution and resume use unrelated empty compiler caches and
forbid Triton/CuTe/native compilation and provider autotuning. Execution also
forbids model/build imports. Runtime uses 4 GiB device and 64 MiB pinned host with
strict physical admission and checks measured physical counters after every
invocation. These small cases need not force a spill transfer.

The checks compare all exported state and losses under the existing numerical
qualification bounds, inspect actual fused GEMM destination pointers, validate
the runtime trace, restore all checkpoint tensors bitwise, and resume two
invocations bitwise in another process. Saved Program/profile/Plan/reference
files must remain unchanged. Logs, artifacts and per-phase status stay in the
pytest temporary directory, including failures. A child timeout terminates its
whole process group. No planner or independent-reference phase may use CUDA.

`SHADOWSPILL_SONIC_BUILD_CACHE` can name an existing ordinary Quack compiler
cache to shorten builds. Its use is recorded and applies only to building;
profiling/execution/resume still have empty phase caches and compiler guards.
It never supplies a previously built Program or profile. Results with it are
not cold-build latency measurements.

This slow test is excluded from the default shared suite; explicitly run the
command above in addition to full gates when qualifying prepared Sonic changes.
It is not full-model, other-dtype, DP, expert-parallel or all-provider coverage.

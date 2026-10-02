# Generic profiling saved-value lifetimes

## Problem

Before this change, preparation ran all structurally distinct forward producers
and retained their saved tensors in the host spill pool until all profiles and
selected entrypoints were ready. This made preparation require the sum of those
snapshots, even when a feasible training schedule would release or recompute
them. The 16K EP2 example retained 36.34 GiB. Its 32K candidate exhausted an
80 GiB spill pool while requesting another 2.14 GiB with 1.82 GiB free.

This is a generic task-profiling problem. No model, precision, operator, or
process-group name is part of the fix.

## Lifetime contract

1. Register the actual graph-pair occurrence as a recipe, without executing it.
2. When a backward requires measurement, replay that occurrence's forward.
3. Copy only saved inputs that lack an authentic reference into the spill pool.
4. Measure the backward, then release its device examples and host snapshots.
5. Repeat independently for the next backward. Cached selected entrypoints use
   the same replay/release scope when they need warmup.

The producer replay runs outside the backward timing/workspace interval. Its
wall time contributes to preparation overhead. Structural/profile deduplication
is unchanged. Registration uses object identity to avoid confusing occurrences
that share compiled code but have different weights or declared metadata.

During distributed preparation, all ranks replay a producer if any rank needs
its values. Normal profiling phase coordination preserves communication order.
If profiling fails, allocator recovery happens before persistent snapshot
unregistration. A single snapshot that cannot fit still fails planning.

Thus the extra host capacity is bounded by **the largest single snapshot**,
rather than the sum over distinct forwards. No pool size was increased.

## Changed components

- `pytorch/graph_pairs/saved_values.py`: register recipes instead of rebuilding
  the entire captured model with retained representative activations.
- `pytorch/planning/training/profile.py`: register once and report peak/current
  snapshot bytes.
- `pytorch/profiling/profiler/`: on-demand replay, scoped snapshots, timing and
  cleanup.
- `pytorch/profiling/executables.py`: prepare authentic inputs around cached
  selected-entrypoint warmup.
- `tests/shadowspill/pytorch/api/test_03_planning_host_memory.py`: fresh and cached
  builds assert that every earlier snapshot is already released before another
  is retained. The no-free-spill-pool failure/cleanup test remains in place.

## Validation and evidence

- 79 existing profiler/planning CPU checks passed.
- Ruff and mypy passed for changed production modules.
- H100 fresh/cached-plan lifetime regression passed; peak is one snapshot and
  current bytes are zero before plan adoption.
- H100 no-free-spill failure/cleanup regression passed.
- EP2 ShadowSpill save/recompute training and independent model oracle passed:
  relative loss difference 7.27e-6; no parameter checks failed.
- Full suite: 1,104 passed, one skipped; all 48 CUDA CTests passed.
- Numerical rerun: **5/5 passed**, `della_saved_lifetimes_h100_1002`, using the existing H100
  configuration and October references. The first launcher omitted this config
  and used September's references: 179 optimizer-state structure mismatches,
  while model/loss/checkpoint replay checks passed. That failed attempt remains
  archived under `della_saved_lifetimes_1002`; no references were regenerated.
- Full 32K model planning and physical admission passed on both ranks at the
  unchanged 50 GiB execution / 80 GiB host-spill budgets. Predicted global-step
  times are 6.533 and 6.674 seconds; real training timing is still pending.
- Full 64K planning and physical admission also passed on both ranks, with
  predicted step times 6.564/6.573 seconds.
- 128K/256K retries continue in `codex:0.0`; fresh artifact stores live
  under `chicago-ep2-capacity-fixes-1002/` on scratch.

GPU logs: `~/storage/shadowspill/generic_training_0930/della_1001/` under
`saved-value-lifetimes/` and `moonep-planner-o2-validation/`. Small-model success
does not itself establish large-model feasibility.

The ordinary numerical cases also show the intended host-memory reduction:

| Case | Previously retained across profiling | New snapshot peak | Live after profiling |
| --- | ---: | ---: | ---: |
| PyTorch Llama3 | 2.41 GiB | 23.86 MiB | 0 |
| MLOps Llama3 | 2.37 GiB | 501.75 MiB | 0 |
| PyTorch Qwen3.5 | 3.38 GiB | 45.75 MiB | 0 |
| MLOps Qwen3.5 | 3.31 GiB | 728.06 MiB | 0 |
| MLOps OLMoE | 815.25 MiB | 98.62 MiB | 0 |

Sources: `gates_della_generic_refresh_14848333/numerical.log` versus
`gates_della_saved_lifetimes_h100_1002/numerical.log` in `qualification/results/`.
These are profiler snapshot bytes, not total host or device memory.

The MLOps Qwen3.5 shutdown still reports detaching eight retained 24/32-byte
workspace storages (224 bytes total), with zero orphan allocations reclaimed.
The same warning is present in the earlier successful H100 baseline, so it is
not introduced by this change. Sources: each run's `mlops_qwen35.log`, at callable
close. This remains a separate small-storage ownership/diagnostic follow-up.

Published on master: ShadowSpill `17c0a2d1`; accompanying MoonEP compatibility
fix MLOps `e61e7b2`. Both were pulled on Tübingen, where the complete gate set is
running as `tubingen_capacity_fixes_1002`.

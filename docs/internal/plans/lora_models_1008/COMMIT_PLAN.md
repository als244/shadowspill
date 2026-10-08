# Approved commit groups — 2026-10-08

The user approved committing and pushing both repositories, then updating the existing master checkouts on Chicago, Della, Tübingen and fatnode. No extra branches or worktrees are needed.

## MLOps

1. Keep auxiliary routing counts FP32 under every default dtype, with regression tests.
2. Respect precision, narrow matrix dimensions and device shared-memory limits in grouped expert GEMMs.
3. Add configurable LoRA modules, grouped expert projections and a bounded head loss; omit frozen weight gradients; include tests and API documentation.

## ShadowSpill

1. Materialize shared activation cotangents when independent task-boundary storage is required.
2. Account for empty views that retain nonempty allocations.
3. Organize the example architectures, optional expert parallel construction and full-model LoRA recipes, with catalog, examples and tests.
4. Record concise validation reports, graphpair tables and reproduction scripts. Raw compiler/build stores, large allocation timelines and tensor checkpoints remain outside Git.

Validation: 1,238 suite tests passed (one skipped), numerical 5/5 against existing references; 32 full-model GPU/reference LoRA cases; six default-model throughput comparisons and the earlier 28 performance cases. Multi-GPU validation of the consolidated optional-EP constructors and whole-model FP8 LoRA remain separate follow-up work.

The scripts in scripts/ record the approved source groups. SYNC.md records the resulting commits and machine updates. Existing unrelated machine-local gate configs, report symlinks and historical plan drafts are preserved.

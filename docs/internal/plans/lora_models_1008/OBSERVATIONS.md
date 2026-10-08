# Additional observations

## FLA metadata references at callable close

Dense Qwen 3.5 and Qwen 3.5 MoE, both full training and LoRA, print a close-time
notice about small INT64 metadata views (lengths and cumulative offsets).
The notice reports **zero unreleased allocations**, and detaches remaining
Python references to recycled workspace. This is independent of LoRA.

The installed FLA package has an identity-keyed `tensor_cache` decorator used
by `fla.ops.utils.index.prepare_lens` and related metadata functions; its
bounded queues default to four entries. The reported 4/5-element tensors match
the four packed sequences in these experiments. FLA exposes
`FLA_DISABLE_TENSOR_CACHE=1`, but that option was not changed for these results.
This is source evidence for the likely reference owner, not a separate
allocation-leak reproduction. Results retain the original console diagnostics.
No FLA source or global configuration was modified for this task.

## Timing variability

The initial 10-step 1.18B LoRA measurements each include one approximately
313 ms step; other save steps are around 100 ms, recompute around 123 ms.
Per-step timing is retained, and a longer follow-up records Python GC events
to distinguish execution cost from host-side pauses. Report median and
aggregate throughput separately until that follow-up is complete.

Follow-up: 40 measured steps are stable with GC enabled. Save: 99.03–100.78 ms; recompute: 121.77–123.72 ms. A generation-2 collection took 231.85 ms during save warmup. No generation-2 collections occurred within either measured window. The earlier outlier attribution remains an inference because those runs did not record GC events. See TIMING.md. An explicit dense addmm epilogue probe gave no consistent improvement and bitwise-identical outputs/gradients, so production retains its simpler expression.

2026-10-08 performance-gate-scale follow-up: full Qwen LoRA training completed 12 measured updates with stable timings and finite losses. At close, the existing FLA metadata retention diagnostic reported eight cached INT64 sequence-metadata views (four 16-element and four 17-element tensors; 1,056 bytes total), with zero unreleased allocations reclaimed. This is the same previously recorded close-time metadata-cache issue at a larger sequence count; no new allocator leak or hot-path driver allocation was observed. Case exited successfully.

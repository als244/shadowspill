# Engineering investigations

These reports preserve root-cause evidence for failures that shaped the current
runtime design. They are historical analyses, not normative API contracts.

- [Allocator retirement storm](allocator-retirement-storm.md) explains the
  callback storm and dispatcher contention that motivated isolated retirement
  processing.
- [Prefetch admission deadlock](prefetch-admission-deadlock.md) documents a
  timing-dependent schedule/runtime disagreement and its causal correction.
- [Qwen runtime overheads](qwen-runtime-overheads.md) reconciles the original
  standard-allocator, simulator, and ShadowSpill execution measurements.
- [Step prologue and terminal tail](step-prologue-and-terminal-tail.md)
  decomposes the per-invocation cost outside the selected span, explains the
  simulator's one-sided optimism, and records why first-use ordering was
  adopted over steady-state residency.

Current behavior is defined by the [architecture](../architecture/overview.md),
[Python](../python/README.md), and [C](../c/README.md) documentation.

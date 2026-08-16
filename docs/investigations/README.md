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

Current behavior is defined by the architecture, runtime, planner, simulator,
PyTorch frontend, and memory-budget documents in the parent directory.

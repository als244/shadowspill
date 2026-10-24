# Implementation readiness

The user accepted the phase-based layout, small top-level API and autonomous
implementation order. Chicago preparation was completed and pushed; the refactor
branch is `codex/offline-pipeline-refactor`, based on clean master `bec7950b`.
Implementation runs in the adjacent worktree and `codex:offline-refactor` tmux
window. See [progress](progress.md) for completed milestones and active work.

Follow the [approved implementation order](implementation-order.md):
offline planning first, the single-GPU pipeline next, and
distributed execution/sharding last. Do not make distributed implementation a
prerequisite of the offline or single-GPU milestones.

## Selected migration policy

Migrate repository callers, then remove superseded public APIs and duplicate
implementations. The user selected removal rather than deprecated public
compatibility wrappers. Temporary internal adapters may support the migration,
but must not become a second long-term API or implementation. Preserve existing
capabilities through the new owners before removing their old entry points.

## Selected forward/backward gradient default

For a `build_forward_backward` Plan, each invocation starts fresh gradients by
default (`accumulate_gradients=False`). The user selected explicit opt-in through
`accumulate_gradients=True` to add to stored gradients from an earlier invocation.
Accumulation across `num_examples` within one invocation is unchanged. Previously
reduced gradients must not be reduced again; both behaviors must be represented
and covered by the saved Plan's memory requirements.

The two identified behavioral choices are now settled. Complete the final API
consistency review below before implementation; this does not imply that every
public field or extension protocol has already been agreed.

## Final API consistency pass

The [consolidated public review](public-api-review.md) now shows the ordinary DP
workflow, supplied-profile CPU planning, shared-plan execution with a complete
rank map, fresh/accumulating forward-backward calls, and the mixed-sharding
example together. It records the selected defaults and remaining exact protocols.

Finish the complete signature/record specification against those examples. Check
the ordinary DP workflow, supplied-profile CPU simulation, shared-plan
distributed execution with a complete rank map, and
forward/backward calls with persistent gradients. Present substantive unresolved
design issues together if this pass reveals them, rather than restarting a
field-by-field questionnaire or silently selecting new behavior.

Finish the precise profile/memory record, initializer/export callback, optimizer
runtime-value, communication-calibration, archive metadata, and custom-provider
contracts under their already selected responsibilities. Minimize public types
and keep plain values/metadata. Do not add general registries, factories, or
extra public wrappers merely for implementation convenience.

## Engineering work, rather than preference questions

- Determine whether normal allocator observation provides the complete evidence
  required for memory admission; preserve the permitted fallback if necessary.
- Inventory provider build preparation and allocation/effect contracts. One-GPU
  DP measurement must stay separate from distributed calibration.
- Make persistent executable payloads independent of original compiler-cache
  paths, while enforcing selected dependency/build compatibility.
- Track existing advanced behavior before deleting superseded wrappers. User
  cleanup authorization does not implicitly discard tied/shared storage,
  parameter metrics, partial checkpoint support, or precision controls.

## Code organization standard

Small public surface; short, cohesive functions and files; hierarchy within a
phase only when it improves navigation or separates real components. Each phase
owns its implementation. Shared IR/artifacts define facts once. Trainer and
quickstart compose public calls. Keep optional caches, GPU measurement, and
runtime initialization out of CPU planning/reporting paths.

# Naming conventions

## General

- Python modules and functions use `snake_case`; public types use `PascalCase`.
- Public C symbols use the complete `shadowspill_` prefix.
- `MemoryPool` owns a bounded arena; `MemoryLease` owns one range in a pool.
- `execution` and `spill` name the two plan-selected pool roles.
- `fetch` means spill to execution; `evict` means execution to spill.
- `worker` names the C background thread; `handle_*` names its processing
  operations.
- `recomputation` names graph alternatives; `PressureFit` names the search
  algorithm and nothing else.
- an artifact store has a `build` tree and a `planning` tree: `artifact_store`
  roots both, `build_store` and `plan_store` override either, and
  `build_store_mode` and `plan_store_mode` say what a run does with each.
- a store mode is one of `contribute`, `reuse`, `require`, `refresh`, and the
  same four words mean the same four things on every surface.
- `execution_XXXXXX` is the primary chronological task identity; semantic task
  name and canonical IR task ID are separate fields.
- `profiling_metadata` is cache identity for value-sensitive task measurement,
  not a runtime model argument.
- `transfer_bandwidths` names the calibrated fetch/evict rates consumed by
  planning and simulation.

## Provider boundaries

Provider and hardware API names are used only for concrete pool/backend
implementations, hardware identity, physical-accounting reports, or framework
adapter edges. They do not define generic pool, lease, route, planner,
simulator, or runtime semantics.

The supported default execution-pool factory is named `device()` because its
contract is accelerator memory usable by PyTorch. Provider-specific APIs
remain in their backend. PyTorch allocator callback symbols retain the
provider spelling required by the framework hook.

## Avoid

- `backing` for the secondary pool role; use `spill`.
- `topology` for something that is not one; admission takes facts.
- `progress` for the runtime thread; use `worker`.
- `native` as a synonym for the C library.
- `core` for the invariant part of something; say what makes it invariant.
- `compiled` for the C library. It means what torch.compile produced.
- `context` for a search's own input, which is a problem. The word has exactly
  two uses: a driver context, and `ShadowSpillScheduleContext`, the part of a
  planning problem that is not about how it is searched. A backend's opaque
  handle is its state.
- `host` for the secondary pool, which is generic; use `spill`. Keep it only
  where it means the CPU a backend runs on - `pinned_host()` names a pool
  that really is host memory, and a driver call that synchronizes the host
  really does.
- `host` for the dispatching thread's own work; use `dispatch`.
- `h2d`/`d2h` for schedule or lane policy; use `fetch`/`evict`.
- model-family or provider names in framework-neutral policy.
- `utils`, `helpers`, or `common` modules without one precise domain.
- `cache` for a store that is planning evidence; the planning tree is the
  *plan store*, and what a run does with it is a mode, not a temperature.
- a `_dir` suffix on a store argument; it is `artifact_store` and
  `--artifact-store`.
- strategy names in public types when selection is an internal implementation
  detail.

Canonical serialized IR action kinds are `fetch` and `evict`; public
explanations and runtime labels use fetch and evict.

## Backend and device, never a provider name

A provider's platform name belongs to that provider's backend directory under
`csrc/backends/` and to the few PyTorch attributes that carry it (`torch.<provider>.*`,
`<provider>_stream`, `<provider>_event`, `is_<provider>`). The one place the
PyTorch layer names the accelerator's device type is
`shadowspill.pytorch.accelerator`. Everything else says *backend* for streams,
events, allocators, the provider library, and its statistics and capabilities,
and *device* for tensors, placements, and ordinals.

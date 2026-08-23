# Naming conventions

## General

- Python modules and functions use `snake_case`; public types use `PascalCase`.
- Public C symbols use the complete `shadowspill_` prefix.
- `MemoryPool` owns a bounded arena; `MemoryLease` owns one range in a pool.
- `execution` and `spill` name the two plan-selected pool roles.
- `fetch` means spill to execution; `evict` means execution to spill.
- `worker` names the C background thread; `handle_*` names its processing
  operations.
- `recomputation` names graph alternatives; `PressureFit` names planner policy.
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
- `progress` for the runtime thread; use `worker`.
- `native` as a synonym for the C library.
- `core` for the invariant part of something; say what makes it invariant.
- `compiled` for the C library. It means what torch.compile produced.
- `context` unless it is a driver context; the PressureFit one is a
  problem, a backend's opaque handle is its state.
- `topology` for something that is not one; admission takes facts.
- `host` for the secondary pool, which is generic; use `spill`. Keep it only
  where it means the CPU a backend runs on - `pinned_host()` names a pool
  that really is host memory, and a driver call that synchronizes the host
  really does.
- `host` for the dispatching thread's own work; use `dispatch`.
- `h2d`/`d2h` for schedule or lane policy; use `fetch`/`evict`.
- model-family or provider names in framework-neutral policy.
- `utils`, `helpers`, or `common` modules without one precise domain.
- strategy names in public types when selection is an internal implementation
  detail.

Canonical serialized IR action kinds are `prefetch` and `offload`; public
explanations and runtime labels use fetch and evict.

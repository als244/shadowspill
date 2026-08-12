# Naming conventions

## Public and neutral names

- Python modules use `snake_case`; public types use `PascalCase`.
- Public C symbols use the full `shadowspill_` prefix.
- `MemoryPool` owns a bounded arena; `MemoryLease` owns one range in a pool.
- `execution` and `spill` describe plan-selected pool roles.
- `fetch` means spill-to-execution; `evict` means execution-to-spill.
- `worker` names ShadowSpill's background runtime thread.
- `release`, `evict`, and `fetch` describe memory actions in explanatory text;
  the canonical IR action spellings remain versioned schema values.
- `recomputation` names graph variants; `PressureFit` names the planner.

`host`, `device`, `CUDA`, `ROCm`, and provider API names are permitted only in
concrete backend/provider code, physical-accounting reports, framework adapter
edges, or explicit hardware identities. They must not define generic pool,
lease, route, planner, or runtime semantics. Exact legacy-oracle field names
may appear only in the isolated external comparison adapter.

## Prohibited production names

- Names inherited from the prior project or prototypes.
- Provider names in neutral core policy.
- `backing` as the secondary-memory role; use `spill`.
- `progress` for the background thread or its work; use `worker` for the thread
  and `handle_*` for processing functions.
- `native` as a synonym for compiled code.
- `utils`, `helpers`, or `common` modules without one precise domain purpose.
- Strategy names in public types when the strategy is an implementation detail.

The executable naming audit enforces legacy-name and neutral provider-boundary
rules. Historical evidence documents are exempt from terminology rewrites.

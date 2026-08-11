# Naming conventions

## Public names

- Python modules use `snake_case`; public types use `PascalCase`.
- Public C symbols use the full `shadowspill_` prefix.
- Opaque identifiers are described by role: object, task, allocation, plan,
  runtime, device, or resource.
- Use `device` and `host` for memory locations.
- Use `offload`, `prefetch`, and `release` for memory actions.
- Use `recomputation` for graph variants and `PressureFit` for the planner.

## Prohibited production names

- Names inherited from the prior project or prototypes.
- Provider names in core logic.
- `native` as a synonym for compiled code.
- `utils`, `helpers`, or `common` modules without one precise domain purpose.
- Strategy names in public types when the strategy is an implementation
  detail.

The executable naming audit checks production Python and compiled sources.

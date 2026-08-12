# Memory-budget semantics

The execution pool's `physical_capacity` is the complete process-attributable
physical-memory cap for its provider device. It is not merely a limit on
planner-visible tensors.

Runtime initialization accounts for context and bounded provider headroom,
then creates the execution pool arena from the remainder. `execution_budget`
in `plan_step()` or `plan_forward()` defaults to that initialized arena
capacity and may only reduce the capacity exposed to this plan. It cannot
increase the physical cap. `spill_budget` has equivalent semantics for the
selected spill pool.

Physical accounting includes:

- provider context and fixed service cost;
- the complete execution-pool arena;
- bounded provider allocations outside that arena;
- caller-retained outputs and anonymous task workspace inside the arena;
- fragmentation inside the arena.

Objects, anonymous workspace, fragmentation, and caller-retained outputs are
logical subdivisions of the same physical arena and are not double-counted.
Every category and subtraction appears in `PlanReport`.

PressureFit receives usable object capacity after an explicit workspace
reserve. Before sealing, ShadowSpill replays the ordered allocation/free
timeline through the same aligned, coalescing range policy used by the runtime.
Sequential temporaries therefore contribute their maximum simultaneous live
extent, not allocation volume. Unknown or unbounded provider growth is rejected
rather than hidden as leeway.

The initial `pinned_host` provider allocates the spill pool once during
`Runtime` construction. Planning does not resize it. Future peer, remote, or
storage spill providers obey the same selected-pool budget contract even when
their physical accounting mechanism differs.

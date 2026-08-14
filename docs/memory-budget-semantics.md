# Memory-budget semantics

The execution pool's `physical_capacity` is the complete process-attributable
physical-memory cap for its provider device. It is not merely a limit on
planner-visible tensors.

Runtime initialization accounts for context and bounded provider headroom,
then creates the execution pool arena from the remainder. `execution_budget`
in `plan_step()` or `plan_forward()` may be omitted or set to the same complete
physical cap used to configure the runtime; both select the complete derived
execution pool. The fixed runtime cost is therefore subtracted exactly once.
A value no larger than the derived pool capacity remains available as a
logical per-plan limit. It cannot increase or physically shrink the arena that
was allocated when the runtime was initialized. `spill_budget` has equivalent
logical-limit semantics for the selected spill pool.

`device()` reserves 1,280 MiB of provider headroom by default. This value is
reported and can be overridden before runtime initialization; it is never
added above `physical_capacity`. Reserving it up front is required by the
single conventional-slab design because provider-retained allocator pointers
make later slab relocation unsafe.

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

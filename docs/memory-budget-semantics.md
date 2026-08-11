# Memory-budget semantics

`device_budget` is the physical GPU-memory cap attributable to the ShadowSpill
process on the participating device. It is not only a limit on planner-visible
tensors.

The reported accounting includes:

- device context and fixed provider cost;
- the conventional CUDA slab;
- anonymous task workspace;
- slab fragmentation;
- caller-retained outputs;
- bounded allocations that bypass the framework allocator.

The admitted slab is computed from the physical cap after context and provider
headroom. PressureFit receives a smaller usable object capacity after an
explicit workspace reserve. Every subtraction is present in `PlanReport`.

There is no hidden layout allowance. An operation with unbounded direct device
allocation is rejected during planning. Unexpected steady-state physical
growth latches a runtime failure rather than silently exceeding the cap.

`host_budget` caps ShadowSpill-owned pinned-host memory. It does not claim to
limit unrelated pageable memory in the process or system.

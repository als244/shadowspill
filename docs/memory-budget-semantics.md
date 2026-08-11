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

The default admission policy is deliberately visible:

- provider headroom is at least 512 MiB, or the measured external high-water
  plus 64 MiB, rounded to 64 MiB;
- workspace admission is at least 512 MiB, or 125% of the largest measured
  task workspace, rounded to 2 MiB;
- pinned-host reservation is the simulated host peak plus the larger of
  256 MiB or 10%, rounded to 64 KiB.

Objects, anonymous workspace, fragmentation, and caller-retained outputs are
logical subdivisions of the slab. Reports show each category, but physical
usage is `context + slab + external provider bytes`; slab subdivisions are not
added again.

Before sealing, the complete ordered allocation/free timeline is replayed with
the runtime's aligned best-fit and coalescing policy. This catches spatial
fragmentation even when aggregate free bytes appear sufficient. Sequential
temporaries contribute their net simultaneously-live peak, not allocation
volume.

Simulator workspace is a scalar capacity charge, while slab admission requires
the physical extent multiset. Structural allocator telemetry supplies task
temporary extents. Frontend work performed inside a task boundary but outside
the compiled graph must also declare its individual extents; ShadowSpill never
turns an aggregate byte charge into one fictitious contiguous allocation. For
gradient accumulation, each returned per-parameter contribution remains one
extent even though the simulator charges their sum.

There is no hidden layout allowance. An operation with unbounded direct device
allocation is rejected during planning. Unexpected steady-state physical
growth latches a runtime failure rather than silently exceeding the cap.

`host_budget` caps ShadowSpill-owned pinned-host memory. It does not claim to
limit unrelated pageable memory in the process or system.

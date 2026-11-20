Engine
======

``shadowspill`` builds, profiles, plans and runs a training step within device
and spill memory budgets, and serves training runs to clients. Each phase
returns an artifact the next one takes, and each artifact saves to one archive
and loads back (`Artifact.save`, `Artifact.load`), so a phase can run in a
process of its own:

- `build_step`, `build_forward_backward` or `build_forward` captures a model
  as a `Program`, on a representative device.
- `measure_program` measures the Program's tasks as `Profiles`, and
  `apply_profiles` attaches them, giving a `ProfiledProgram`.
  `measure_communication`, `measure_overlap` and `measure_pool_memory` measure
  transfers, their contention with tasks, and a spill pool's device cost.
- `~shadowspill.plan_program` chooses a `Plan` within `MemoryBudgets`, on the
  CPU.
- An `Executor` runs the Plan in a fresh worker process, where it opens the
  execution and spill pools.

The pages follow those phases; the framework-neutral core they share is
documented under Internals.

.. toctree::
   :maxdepth: 1

   build
   profiling
   planning
   execution
   parallelism
   service
   frontend
   diagnostics

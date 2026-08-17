# Examples

These examples are task-oriented recipes built from the public
`shadowspill.memory` and `shadowspill.pytorch` APIs. They complement the
[Python API reference](../python/README.md): examples show complete workflows,
while API pages define individual contracts.

## Suggested order

1. [Training loop](training-lifecycle.md) — end-to-end training and checkpoint.
2. [Forward-only execution](forward-only.md) — plan and execute a fixed-shape
   inference graph.
3. [Concurrent planned callables](concurrent-callables.md) — dispatch several
   admitted callables before synchronizing their results.
4. [Reusable planning and budget sweeps](reusable-planning.md) — separate
   capture/profiling from PressureFit evaluation.
5. [Diagnosing a plan and real step](diagnostics.md) — join PlanReport and
   StepResult evidence by execution ID.
6. [Custom stage partitioning](custom-partitioning.md) — supply a validated
   contiguous FX partition policy.

## Assumptions

- ShadowSpill was installed with `./scripts/setup.sh`.
- The process has a supported accelerator and enough memory for the declared
  runtime pools.
- Runtime construction occurs before any accelerator allocation in the
  process.
- Example tensors remain on CPU until ShadowSpill owns execution placement.
- Planning caches use a fast local filesystem.

The training example is self-contained. The other pages focus on the changed
portion of the workflow and refer back to it for common model, runtime, and
lifecycle setup.

Failure propagation and automatic rollback are documented separately in
[Errors, failures, and cleanup](../python/failures.md); the happy-path examples
stay linear so ownership order remains easy to see.

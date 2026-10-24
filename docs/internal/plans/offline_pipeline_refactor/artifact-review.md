# Artifact portability and compatibility: review proposal

Status: fifth of the six grouped reviews. One archive per artifact, explicit
save/load, self-contained ProfiledProgram and executable Plan bundles, external
checkpoint values, and separate disposable compiler caches are already selected.
This document consolidates their proposed contents and compatibility behavior.
The user selected exact matching of relevant software builds initially.
Detailed archive schemas and packaging remain proposals for the final API review.

## Proposed contents

| Artifact | Included information |
| --- | --- |
| Program | Built task definitions and alternatives, graph/storage/input structure, semantic source names, logical parallelism, required graph/code payloads, finalized preparation choices, and target/provider requirements. |
| Profiles | Compatible task identities, timing and missing/dynamic memory records, source/classification metadata, and the collection's hardware/runtime setup. It need not duplicate task code. |
| ProfiledProgram | Its Program and the applied profile records needed to use it independently of original folders and caches. |
| Plan | Selected task code and declarations, schedule, memory layout and evidence, required resource descriptions, logical participants/groups, input/persistent-tensor descriptions, simulated timeline, and relevant profile/communication assumptions. Unselected alternatives are not execution dependencies. |

Each archive contains readable metadata and packed payloads. JSON metadata and
ZIP remain candidate formats, not a new user-selected requirement. Required
executable dependencies must be copied into the archive or explicitly named as
external runtime/provider libraries. A path into the original Inductor/Triton
cache is not a portable dependency. Do not save the whole compiler cache.

All workers load the same Plan archive. Their complete launch rank map is a
runtime input. Checkpoint tensors, input data, process groups, streams, device
pointers, and live provider buffers are not saved executable objects. Saved
resource descriptions are recreated through provider initialization. Reports
and checkpoint file formats remain separate from phase-artifact save/load.

## CPU loading and execution validation

Recommend separating these operations:

```python
# CPU process: inspect the saved result and render its simulated timeline.
plan = Plan.load("training.plan.zip")
print(plan.summary)
write_report(plan, "reports/training")

# Fresh execution worker: bind and validate the actual runtime environment.
plan = Plan.load("training.plan.zip")
with Executor(plan, rank_map=rank_map, device=device) as executor:
    executor.initialize(fill_tensor)
    result = executor.run(args=examples).result()
```

CPU load checks the archive schema and required payload integrity without
initializing CUDA, importing GPU provider implementations, or recreating live
compiled callables. An artifact whose schema cannot be understood still reports
a format incompatibility; inspection is not permission to reinterpret an
unknown schema. Execution payloads are resolved only when needed.

Executor validates relevant executable requirements and shared plan/rank mapping
before dispatch. Checks needing the device occur after allocator installation;
do not initialize CUDA early solely to run a compatibility check. Software,
hardware, resource, and input mismatches should identify the failed requirement
and its recorded/actual values. Startup does not silently retune, rebuild,
measure, or replan. These phase boundaries were already selected.

## Selected software compatibility policy

The user selected conservative exact matching initially for the recorded versions/build
identities of dependencies that can change task code, layout, allocation, or
runtime behavior: relevant PyTorch/compiler builds, ShadowSpill runtime ABI, and
the providers used by the selected tasks. A version label alone is insufficient
for locally modified kernels; record relevant implementation/build identities.
CPU inspection/reporting remains independent of executable dependency validation.

The match should concern actual executable dependencies, not every installed
Python package, an unrelated documentation commit, the original absolute path,
hostname, or GPU ordinal. A bundled fixed code payload may have a content identity
independent of the source checkout. Hardware features, supported device targets,
driver/binary interfaces, and available capacity still need their own executable
requirements; exact equality of every environment field is not proposed.

Example: a task compiled/profiled with one PyTorch/provider build should not run
against a different unqualified build merely because it imports successfully.
The user would explicitly build, supply compatible memory/timing evidence, and
plan for the new environment through the normal public phases. CPU inspection of
the original artifact can still work with a supported archive reader.

Accepting declared version ranges from the outset was not selected. A future
relaxation would need evidence that the selected code/ABI, allocation
bounds/lifetimes, and required effects remain compatible. Do not assume that a
minor/patch version change preserves these properties or add an arbitrary
ignore-checks flag.

## Execution requirements versus performance assumptions

Execution requirements cover code and provider interfaces, logical geometry,
target features, memory capacity, required allocations, and communication/resource
behavior. Failure is an execution incompatibility. Estimated memory remains
simulation-only under the already-selected rule.

Measured durations and supplied link rates describe a performance scenario.
Where a difference affects only timing and all execution requirements are still
satisfied, it need not invalidate the executable. Report the mismatch and retain
the original simulated predictions rather than silently claiming they were
measured on the new setup. This treatment needs to be reflected in the final
compatibility schema; it is not permission to relabel a memory-affecting provider
or topology change as a timing-only change.

## Compilation, extraction, and disposable caches

The selected build boundary already permits library-internal lazy binary
compilation or dispatch when the finalized task and allocation requirements are
preserved. Loading an archive must not start a new autotuning search which can
invalidate the profiled task. Required source/binary payloads and fixed choices
must suffice independently of the originating compiler cache.

Some backends may require files on local disk. Extract only required payloads to
local temporary/cache storage, with archive identities keeping copies reusable.
Removing those local copies must not destroy the saved artifact's dependencies.
Runtime/provider code installation remains external. Whether each current
provider can meet this contract is an implementation investigation; source/graph
packaging alone is not proof that its saved preparation choices are reproducible.

The remaining concrete container/version metadata and source layout will be
reviewed with the final API examples and migration. No additional public manifest
class or mandatory ArtifactStore is required merely to move between phases.

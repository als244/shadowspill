# Build completion and measurement

Status: the user selected finalizing task definitions in build, with measurement
warming up and recording timing/memory without changing those definitions. This
is the second review's agreed boundary, not an implemented API. GPU-assisted build
is allowed where required; one-GPU profiling for local DP compute must not need
live process groups. Build remains independent of ShadowSpill runtime pools.
Apply/plan/sweep/report remain CPU-only. Exact provider hooks and target descriptors
remain implementation/interface details to review before adoption.

## Selected successful-build boundary

A returned Program contains finalized task definitions for its target:
the exported computation, selected implementation/configuration where it affects
the task, input/output layouts and effects, executable code or supported code
payloads, and compiler-visible memory information. Missing dynamic/provider
allocation evidence remains explicitly listed for profiling; build does not
promise that all memory requirements are statically known or attach task timings.

Complete preparation that determines those task definitions inside build. Use
CPU compilation when supported. If the actual compiler/provider needs
a representative GPU to finish such preparation, build requires that device and
reports a clear missing requirement when unavailable. Ordinary compiler errors
must remain errors rather than be disguised as a GPU request. A failed build
does not return an unresolved Program as though construction succeeded.

For example, GPU autotuning that selects a task implementation with different
layout or allocation behavior must be resolved as part of preparation before
that implementation is profiled and planned. Build may share input-preparation
utilities with measurement to do bounded task/pair preparation, without requiring
the whole model on device or assigning measured task durations to Program.

This is not a requirement to expose or freeze every library-internal CUDA kernel
name. Lazy binary compilation or internal dispatch remains possible when it
preserves the established executable compatibility and allocation requirements.
Whether a specific provider can make that guarantee needs implementation
investigation; no universal unresolved-kernel flag is assumed. Any choice that
changes the planned allocation behavior or invalidates the applied profiles
cannot silently change during execution. Exact persistence of preparation
choices is part of the artifact review.

## What measurement does

`measure_program` takes the built task definitions, prepares valid inputs and
corresponding forward-produced saved values, performs warmup, and measures task
durations and required allocation information. Missing dynamic workspace and
external/persistent provider memory must be accounted for with sufficient
evidence for execution. Timing and detailed allocation observation can run as
separate passes. Measurement does not change the Program's graph, public tensor
layouts, selected task definitions, or identity behind the caller's back.

Forward/backward task pairs are prepared as needed, preserving numerical and
saved-value semantics. Pure DP local compute uses one representative target GPU
with the actual shard shapes implied by the logical DP size. It does not pretend
world_size is one or require the full model resident. DP reductions and storage
collectives remain outside local optimizer compute. Direct collective measurement
or a task that itself contains distributed work requires its participants unless
that provider explicitly supports a valid alternative measurement method.

CPU-only planning remains possible from saved Program and profile artifacts, or
from externally supplied profiles. Building a new Program without a GPU is
supported where its compiler/provider path can finish offline; the initial
design does not promise offline construction for every GPU backend.

## Minimal information at custom/provider boundaries

Use ordinary PyTorch export/fake/autograd information where it is sufficient.
There is no proposed new mandatory model base class or blanket second custom-op
registry. A backend integration supplies information that remains hidden behind
its operation boundary:

| Information | Purpose and phase |
| --- | --- |
| Tensor geometry, aliases, mutations, differentiation, and relevant replay effects | Build must know the graph and storage/dependency relationships. Ordinary PyTorch metadata is reused. |
| Target/configuration and supported preparation | Build selects compatible task definitions; preparation that needs a GPU must report that requirement. |
| Logical group and resource references | Saved descriptions replace process-local handles. Workers initialize/bind actual groups and resources later, as already selected. |
| Shared/fixed allocations and dynamic scratch | Static descriptions or profiling supply the memory information. Shared resources have one identity and are counted once. Unknown memory is not zero. |
| Completion behavior | The provider preserves its caller-stream completion contract, including joins from private streams. ShadowSpill does not take ownership of those private streams. |

This table explains necessary information, not a new registration API or final
method names. Resource initialization/cleanup signatures and binding descriptors
still need to be made concrete with the execution and artifact interfaces.

## Alternative and scope of this decision

The unselected alternative allows build to return task definitions whose implementation
or layout is finalized during measurement. That can make some builds easier to
run offline, but requires a clear way to return changed executable definitions
and associate profiles with the updated identity while Program stays read-only.
It is more than ordinary warmup or recording previously unknown scratch sizes.

The user selected the completed-task-definition boundary above. This does not decide a
particular `target` descriptor class, the exact GPU-device argument, dtype
defaults, provider hook names, native allocator-observer feasibility, or archive
codec. Those details must implement the agreed boundary, and any public choices
still need review rather than being inferred from this proposal.

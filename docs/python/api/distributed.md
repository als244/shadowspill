# Distributed planning and training

For ordinary distributed training, pass `parallelism` during public-client setup
and submit global batches to one service endpoint. See the
[eight-GPU example](../../../README.md#pretrain-qwen3-30b-a3b-on-eight-gpus).

This reference explains the underlying phase APIs. Declare logical ownership
when building a Program, measure DP compute on a representative GPU, calibrate
communication separately and plan on CPU. Homogeneous workers load the same
complete Plan and bind their logical ranks. Different hardware requires
compatible worker-specific Plans, as described below. Only execution opens
ShadowSpill's runtime pools.

## Simple DP

```python
from shadowspill import Parallelism, build_step, measure_program, apply_profiles

parallelism = Parallelism.dp(ranks_per_replica=8)
program = build_step(
    model,
    objective=objective,
    optimizer=optimizer,
    example_args=example_args,
    example_kwargs=example_kwargs,
    num_examples=4,
    parallelism=parallelism,
    target="cuda:0",
)
program.save("program.ssp")
profiles = measure_program(program)
profiles.save("profiles.ssp")
apply_profiles(program, profiles).save("profiled.ssp")
```

The application supplies the model, optimizer, objective and rank-local example
inputs. The model can start on CPU or meta; build does not initialize persistent
weights. Plain DP needs neither live process groups nor eight GPUs for build or
compute profiling. Build finalizes task code; measurement records its duration
and allocation requirements using the ordinary PyTorch allocator.

A **model replica** collectively owns the complete model state. Eight ranks in
one replica can each compute an ordinary DP batch while storing one eighth of
the weights, gradients and optimizer data. `Parallelism.dp` shards all three by
default. Optional masters follow optimizer ownership; scalar counters and
hyperparameters are not parameter-sized shards.

`Parallelism.dp(ranks_per_replica=8, num_replicas=2)` describes two copies of
that eight-rank layout. It uses 16 ranks without increasing the storage-sharding
factor. Corresponding copies still require gradient reduction. Global sharding
across all 16 ranks instead uses `ranks_per_replica=16, num_replicas=1`.
The declared replica relationship also determines corresponding-copy reductions;
it does not duplicate data or launch workers by itself.

## Ownership and setup

`Parallelism` is immutable, serializable metadata. It does not create process
groups, divide model code, or rewrite a local layer into a distributed layer.

| Declaration | Meaning |
| --- | --- |
| `groups` | Named, ordered groups of logical ranks within one complete-model replica |
| `parameter_placements` | Which logical parameter or parameter slice the local model computes with |
| `parameter_storage` | Where weights, gradients and optimizer data are persistently stored |
| `inputs` | Independent input partitions, field delivery, loss and output ownership |
| `parallelism.group(name)` | A typed `GroupRef` for a compatible distributed layer/backend |

Each named group collection covers the replica exactly once, with equal-sized
memberships. `num_replicas` repeats those memberships at successive logical-rank
offsets. `GroupRef.size`, `.members(logical_rank)` and `.index(logical_rank)` are
available without CUDA or a communicator. A compatible provider must explicitly
bind that reference to its resources later; an arbitrary layer expecting a live
PyTorch ProcessGroup cannot accept it automatically.

Parameter declarations use module-qualified names and wildcard patterns.
Exact names win over patterns; conflicting patterns and unused selectors are
errors. Either name of a tied parameter can select it, but conflicting tied-name
declarations are rejected. One selected storage entry inherits omitted fields
only from `default_parameter_storage`.

`ParameterStorage` has a common `group` and flat `weights`, `gradients` and
`optimizer` modes (`"sharded"` or `"replicated"`). Optional `weight_group`,
`gradient_group` and `optimizer_group` override the common group. Optional
`min_shard_bytes`, also an argument of `Parallelism.dp` and of the client's
`data_parallel`, where it is 2 MiB by default, keeps every component of a
parameter whose unpadded physical weight storage is smaller replicated: a shard
of a small tensor saves little memory and costs a collective at every fetch.
Zero shards every parameter. Masters follow optimizer ownership. A storage group may contain only
ranks that compute with copies of the same logical parameter slice.

The modes govern parameters. A buffer is stored whole on every rank: it has no
gradient or optimizer state to share a shard with, and a program that mutates
one (a step counter, a running statistic) does so on each rank's own copy.
Optimizer state that is not parameter-element storage, such as a column norm,
keeps the placement its optimizer gave it: replicated optimizer data loads and
exports it by its own geometry, and sharded optimizer data cannot hold it.

For example, the ownership resolver can describe two eight-way model-shard
groups inside one 16-rank replica, with corresponding expert slices stored across
pairs and dense attention weights stored across all 16 ranks:

```python
from shadowspill.parallelism import Parallelism, ParameterStorage, Replicated, Sharded

parallelism = Parallelism(
    ranks_per_replica=16,
    groups={
        "all": [tuple(range(16))],
        "partition": [tuple(range(8)), tuple(range(8, 16))],
        "copies": [(rank, rank + 8) for rank in range(8)],
    },
    default_parameter_placement=Replicated("all"),
    parameter_placements={"layers.*.experts.*": Sharded("partition", dim=0)},
    default_parameter_storage=ParameterStorage(
        group="all", weights="sharded", gradients="sharded", optimizer="sharded",
    ),
    parameter_storage={"layers.*.experts.*": ParameterStorage(group="copies")},
)
exchange = parallelism.group("partition")
```

This describes ownership, not an executable EP recipe. The model supplies each
local expert slice and the distributed operation. Builders lower local
parameter shards and replicated optimizer updates alongside sharded DP. The
16-rank, two-group example above is an ownership declaration; it still needs a
model and compatible execution Plans. Model-owned communication binds
through [provider resources](../../architecture/resources.md), whose group
references execution resolves to the process groups it creates.
Weights and gradients may be replicated while optimizer data is sharded; when
weights or gradients are sharded, their storage groups must match the
optimizer group. Other layouts need explicit redistribution.
Equal contiguous model slices are represented; noncontiguous parameter maps
are not, and ranks' Programs differ only as
[Ranks on different hardware](#ranks-on-different-hardware) describes. The
declaration API cannot establish that arbitrary model operators implement the
layout it describes.

## Data loading and objective normalization

The public service packs and routes global batches using the prepared input
placement. Direct Executor callers supply rank-local tensors themselves, using
the application's data-replica relationship. This can differ from parameter
storage or model-communication groups. Parameter placement alone cannot infer
input-layout semantics.

`Parallelism.inputs` can declare that relationship explicitly. It defaults to
`None`, preserving existing rank-local callers and saved declarations. These
framework-free records describe delivery and ownership; Executor still receives
rank-local tensors and does not route a global batch itself.

```python
from shadowspill.parallelism import (
    GroupRef, InputFieldPlacement, InputPlacement, OutputPlacement, Parallelism,
)

inputs = InputPlacement(
    partitions=GroupRef("input_pairs", ((0, 1), (2, 3))),
    fields=(
        InputFieldPlacement(("inputs", "tokens"), "replicate"),
        InputFieldPlacement(("loss_inputs", "mask"), "replicate"),
    ),
    loss="leader",
    outputs={"predictions": OutputPlacement("leader")},
)
parallelism = Parallelism.dp(ranks_per_replica=4, inputs=inputs)
assert inputs.partition_count == 2
assert inputs.scheduled_capacity([[65536], [65536]]) == 131072
```

Input groups use global logical ranks and cover every worker exactly once.
Their declared order determines the leader and contiguous slice order. Groups
currently have equal size, following `GroupRef`. `InputPlacement.independent(4)`
instead declares four distinct partitions, irrespective of parameter storage.
Fields can be shared metadata, replicated payload or a balanced contiguous
`"slice"` on an explicit axis. The definition must declare every routed field
and use matching slices for aligned model/loss fields, including empty slices.
`region(path, shape, logical_rank)` supplies those slice bounds.

`loss="leader"` assigns a replicated contribution to the first member only;
`"partitioned"` sums disjoint contributions from all members. Output rules select
the leader, sum partial results or concatenate them along a declared axis. The
model/objective and routing code must honor these declarations and collective
ordering; declaring sliced inputs does not implement a distributed model.

`scheduled_capacity` counts actual padded buffer positions once per input
partition. Three 64K buffers exceed a 128K step cap even if only 120K positions
are valid. Replicated copies on cooperating workers count once; masked model
buffers still count in full. This arithmetic does not by itself admit a schedule
or enforce an application's cap. The service's `training.inputs` pipeline packs
the global batch into prepared buffers, routes declared views and restores output
order. `TrainingClient.validate_batch()` checks whether a batch fits locally
before submission. Sliced delivery still requires model operators that implement
the corresponding distributed computation.

Gradient contributions are **summed**, without implicit world-size division.
For an additive objective, every rank/example can contribute
`loss_sum / global_normalizer`, where the denominator counts the complete
update's examples, valid tokens or weighted observations. A local mean needs
the corresponding contribution weight. The simple baseline divides its
rank-local normalized loss by world size.

Ranks on one kind of GPU run the same Program and input signatures. Unequal
example shapes *within* that Program use `example_calls`; ranks on different GPUs
run Programs of their own, described in
[Ranks on different hardware](#ranks-on-different-hardware). DataLoader workers
should return CPU tensors. Account for their buffers/prefetch in host capacity,
and preserve source progress for exact checkpoint continuation.

## Communication calibration and CPU planning

Communication has its own reusable artifact. In a calibration process on every
rank, initialize CPU rendezvous and call:

```python
import torch.distributed as dist
from shadowspill import ProfiledProgram, apply_profiles, measure_communication, measure_overlap

dist.init_process_group("gloo")
try:
    profiled = ProfiledProgram.load("profiled.ssp")
    info = measure_communication(
        program=profiled.program,
        rank_map={rank: rank for rank in range(8)},
        device="cuda:0",
    )
    overlap = measure_overlap(
        profiled, info, rank_map={rank: rank for rank in range(8)}, device="cuda:0",
    )
    if dist.get_rank() == 0:
        info.save("communication.ssp")
        overlap.save("overlap.ssp")
        apply_profiles(profiled.program, profiled.profiles.merge(overlap)).save("profiled.ssp")
finally:
    dist.destroy_process_group()
```

Here each worker exposes its assigned physical GPU as process-local `cuda:0`.
Calibration derives all-gather, reduce-scatter and all-reduce requests from the
Program. An explicit `collectives=[...]` overrides those probes. It measures host
copies concurrently across workers, then collective groups separately; each
worker receives the same combined result. It runs no compiled compute tasks.
The subsequent `measure_overlap` call runs saved task/collective pairs on those
workers and adds supported interference measurements to the compute profiles.
This preserves single-GPU build/ordinary profiling and CPU-only planning.
Service preparation reuses compatible paired evidence
across search candidates through `OverlapProfileCache`. Standalone callers can pass the same
cache instance to each `measure_overlap` call. By default, cheap probes take three
trials and expensive probes take one; pass `trials=N` to override. Allocator
cleanup precedes warmup, so timed trials reuse the conditioned workspace.

`CommunicationInfo` contains node membership, rates/latencies, measured collective
costs and communication memory evidence. Collective bytes describe the full
padded gather output/scatter input, or each participant's all-reduce tensor.
Compatibility includes operation, ordered members, size, dtype, in-place layout
and backend setup. Supplied rates/topology fill timing gaps. Measured costs do
not automatically estimate contention between arbitrary overlapping groups.

Planning is a separate CPU call:

```python
from shadowspill import CommunicationInfo, MemoryBudgets, ProfiledProgram, plan_program, write_report

plan = plan_program(
    ProfiledProgram.load("profiled.ssp"),
    memory_budgets=MemoryBudgets(execution_bytes=40 << 30, spill_bytes=128 << 30),
    communication_info=CommunicationInfo.load("communication.ssp"),
)
plan.save("training-plan.ssp")
write_report(plan, "report")
```

Budgets are per rank. The GPU allowance includes compute/provider overhead and
communicator memory. Measured communication allocation contracts are distinct
from task profiles. Unknown memory is never zero; estimated memory permits
simulation only. Timing estimates can still be used with complete memory
evidence. Inspect `plan.simulation_only_reasons` before execution.

## Ranks on different hardware

A Plan's executables and profiles belong to the GPU that built them, so ranks on
different GPUs cannot share one. Each kind of GPU instead builds, profiles and
plans its own Program, with its own example shape, accumulation count and
budgets, and the ranks train as one group when their Programs agree on what the
ranks exchange: the parallelism, which tensors exist, how they are placed and
sharded, and their geometry. Calibration takes each worker's own Program, and
calibration and Executor startup compare that agreement rather than the Programs
themselves; ranks whose Programs disagree are refused before any collective
runs. Ranks pair collectives by the order they issue them, so their Plans must
also issue the same collectives in the same order. Sharded weight fetches and
gradient evictions are collectives each rank's planner places for its own
hardware, so ranks on different GPUs store weights and gradients replicated and
shard the optimizer state: a step's collectives are then its gradient reduction
and weight publication tasks, in program order on every rank. Preparation and
Executor startup compare the combined issue order of communication tasks and
sharded transfers at their actual task boundaries, including write-back before
fetch. Entries identify the operation and tensor names; a mismatch reports the
first difference. Opaque model/provider collectives remain the provider's
responsibility. A rank's contribution stays `loss_sum / global_normalizer` with the
normalizer counting the whole update over every rank, so ranks taking different
shares of a step are weighted correctly. Backend settings that belong to one
node, such as the interface it reaches the network through, may differ between
workers. These are worker setup choices. The built-in public service launcher
uses GPUs on one host, behind one endpoint receiving global batches. Using the
direct phase APIs across hosts requires a launcher supplying all assigned nodes
and their worker mappings. See the
[service deployment reference](../advanced/service-client.md#connection-and-setup).

## Fresh execution workers

Every worker loads its Plan, the same one unless ranks differ in hardware, and
supplies the same complete **launcher-rank → logical-rank** map. A permutation
changes physical assignment, not logical group order or shard identity. Startup
validates the assignment, distinct devices, the Plans' agreement on what the
ranks exchange and the order they issue collectives in, backend setup and node
membership. Executor creates and closes its own communication groups;
caller-owned rendezvous remains the caller's.

Saved Inductor code retains its compilation device ordinal. If build used
`cuda:0`, each worker must expose its GPU under that local ordinal before CUDA
initialization. The installed `shadowspill.orchestration.launch.rank_device` performs that
selection before importing the target worker module:

```bash
torchrun --standalone --nproc-per-node=8 --no-python \
  python -m shadowspill.orchestration.launch.rank_device -m my_training_worker
```

An application can provide the equivalent environment setup in its own launcher.
Inside `my_training_worker`, after defining its loader and rank-local inputs:

```python
import torch.distributed as dist
from shadowspill import Executor, Plan, save_checkpoint

dist.init_process_group("gloo")
try:
    with Executor(
        Plan.load("training-plan.ssp"),
        device=0,
        rank_map={rank: rank for rank in range(8)},
    ) as executor:
        executor.initialize(fill_owned_chunk)
        result = executor.run(args=local_args, kwargs=local_kwargs).result()
        save_checkpoint(executor, f"checkpoint-rank-{executor.logical_rank}.ssp")
finally:
    dist.destroy_process_group()
```

The public service coordinates these Executors behind `TrainingClient` commands.
A prebuilt evaluation Plan can share current model tensors.
Every invocation of a Plan that issues collectives first checks that all workers
selected the same Plan and options. A Plan that issues none, such as an
evaluation Plan over replicated weights, runs on each worker by itself, so
workers may invoke it different numbers of times. A Plan whose model's provider
resources hold communicator groups issues collectives inside its own tasks, so
it always meets the other workers.
Runtime commands and Executor do not build, profile or replan during execution.

## Transfer lifetimes and optimizer ordering

| Work | Completion boundary |
| --- | --- |
| Weight fetch | Copy the owned host shard, then all-gather the full execution tensor |
| Gradient eviction | Reduce-scatter, any corresponding-owner reduction, then copy the owned result to host |
| Optimizer | Ordered reduction/local update/publication sequence on the compute lane |

Fetch/evict readiness and buffer retirement cover the complete compound transfer.
Their host-copy bytes and collective costs are separate simulator phases.
Transfers can overlap unrelated compute. Generated optimizer communication stays
on the ordered compute lane; it does not introduce independent readers whose
completion could outlive the recorded last task. Planning places updates after
all required old-weight reads, interleaved with stages or at the tail as
`plan_program(optimizer_ordering=...)` selects (`"stage_interleaved"` by default).

The reduction logic counts ranks holding the same logical parameter, including
copies of model shards across repeated groups. When a reduced gradient is fetched
for further accumulation, the previous contribution is injected once before the
next reduction; it must not be multiplied by the number of participants.

A provider's private streams remain its responsibility. Its work must join the
task's current stream before that task's completion event. Tensor/mutation and
backing-storage contracts must describe actual allocations, including padding.
Execution warms saved collectives before admission. NCCL-private allocations use
external headroom; PyTorch-visible scratch retains explicit memory accounting.

## Checkpoints and initialization

Executor's loader fills bounded owned chunks using original tensor names,
geometry and global element offsets. It must fill the destination completely;
it need not construct full model tensors. Matching logical elements must have
consistent values across copies. Random initialization should therefore use
stable parameter/element identities, not a rank-dependent traversal RNG.

Checkpoint helpers stream owned tensor chunks. Use distinct archive paths per
logical rank. A checkpoint rejects a different logical rank, but can be restored
by another launcher rank assigned to that same logical rank. The helpers
assume unchanged topology and do not reshard a checkpoint automatically.
The default saves each rank's authoritative master shards. Sharded compute
weights are reconstructed locally and gathered by the normal Plan when needed.
For replicated compute weights with sharded masters, all workers participate
in restore: after validating their local archives, they exchange bounded CPU
chunks to reconstruct the complete replicas before the first forward. This
does not allocate a full weight tensor on the GPU or run an optimizer step.
`weights="compute"` instead saves compute weights and recreates local masters
from their rounded values. Parameters without masters retain their normal
owned storage. Encoded or incompatible physical layouts still keep their
required physical copies and record the reason.
The public service publishes rank archives as one checkpoint generation and
supports frozen-base omission and compatible state conversion. See
[client checkpoints](../advanced/service-client.md#checkpoints).

The [phase guide](../phases.md) describes phase composition;
[phase artifacts](artifacts.md) describes persistence.

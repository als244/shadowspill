# Distributed planning and training

Declare logical ownership when building a Program. Measure DP compute on one
representative GPU, calibrate communication separately, and plan on CPU. Every
execution worker then loads the same saved Plan and binds its logical rank.
Only execution opens ShadowSpill pools. The current CUDA training path supports
symmetric DP with weights, gradients and optimizer data sharded over the same
groups; broader declarations are described below with their implementation limits.

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
Repeated replicas have ownership/reduction tests; their complete training path
remains separate GPU qualification work.

## Ownership and setup

`Parallelism` is immutable, serializable metadata. It does not create process
groups, divide model code, or rewrite a local layer into a distributed layer.

| Declaration | Meaning |
| --- | --- |
| `groups` | Named, ordered groups of logical ranks within one complete-model replica |
| `parameter_placements` | Which logical parameter or parameter slice the local model computes with |
| `parameter_storage` | Where weights, gradients and optimizer data are persistently stored |
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
`gradient_group` and `optimizer_group` override the common group. Masters follow
optimizer ownership. A storage group may contain only ranks that compute with
copies of the same logical parameter slice.

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
local expert slice and the distributed operation. Builders now lower local
parameter shards and replicated optimizer updates alongside sharded DP. Their
build/profile and CPU planning checks pass; multi-GPU execution qualification is
pending. Model-owned communication still needs portable provider-resource binding.
Weights and gradients may be replicated while optimizer data is sharded; when
weights or gradients are sharded, their storage groups must currently match the
optimizer group. Other layouts need explicit redistribution.
Equal contiguous model slices are represented today; noncontiguous maps,
distributed input-placement maps and different graphs per logical rank remain
unimplemented. The declaration API must not be mistaken for qualification of
every layout it can describe.

## Data loading and objective normalization

Inputs are rank-local. Trainer/Executor do not distribute a global iterable.
Select data using the application's data-replica relationship, which can differ
from parameter-storage or model-communication groups. Model-parallel operations
also need explicit input-layout semantics; parameter placement alone cannot
infer them.

Gradient contributions are **summed**, without implicit world-size division.
For an additive objective, every rank/example can contribute
`loss_sum / global_normalizer`, where the denominator counts the complete
update's examples, valid tokens or weighted observations. A local mean needs
the corresponding contribution weight. The simple baseline divides its
rank-local normalized loss by world size.

All workers currently run the same Program and input signatures. Unequal example
shapes *within* that Program use `example_calls`; this does not provide different
Programs or arbitrary shapes on different ranks. DataLoader workers should return
CPU tensors. Account for their buffers/prefetch in host capacity, and preserve
source progress for exact checkpoint continuation.

## Communication calibration and CPU planning

Communication has its own reusable artifact. In a calibration process on every
rank, initialize CPU rendezvous and call:

```python
import torch.distributed as dist
from shadowspill import Program, measure_communication

dist.init_process_group("gloo")
try:
    info = measure_communication(
        program=Program.load("program.ssp"),
        rank_map={rank: rank for rank in range(8)},
        device="cuda:0",
    )
    if dist.get_rank() == 0:
        info.save("communication.ssp")
finally:
    dist.destroy_process_group()
```

Here each worker exposes its assigned physical GPU as process-local `cuda:0`.
Calibration derives all-gather, reduce-scatter and all-reduce requests from the
Program. An explicit `collectives=[...]` overrides those probes. It measures host
copies concurrently across workers, then collective groups separately; each
worker receives the same combined result. It runs no compiled compute tasks.

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

## Fresh execution workers

Every worker loads the same Plan and supplies the same complete
**launcher-rank → logical-rank** map. A permutation changes physical assignment,
not logical group order or shard identity. Startup validates the assignment,
distinct devices, backend setup and node membership. Executor creates and closes
its own communication groups; caller-owned rendezvous remains the caller's.

Saved Inductor code retains its compilation device ordinal. If build used
`cuda:0`, each worker must expose its GPU under that local ordinal before CUDA
initialization. From this checkout, `benchmarking.device_worker` performs that
selection before importing the target worker module:

```bash
torchrun --standalone --nproc-per-node=8 --no-python \
  python -m benchmarking.device_worker my_training_worker
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

`Trainer(plan, device=0, rank_map=...)` provides the loop and schedules over this
same Executor. A prebuilt evaluation Plan can share its current model tensors.
Neither API builds, profiles or replans during execution.

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
completion could outlive the recorded last task. Planning chooses either tail
updates or stage-interleaved updates after all required old-weight reads.

The reduction logic counts ranks holding the same logical parameter, including
copies of model shards across repeated groups. When a reduced gradient is fetched
for further accumulation, the previous contribution is injected once before the
next reduction; it must not be multiplied by the number of participants.
Broader repeated-replica execution still needs qualification.

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
by another launcher rank assigned to that same logical rank. Current helpers
assume unchanged topology and do not reshard a checkpoint automatically.
See [Trainer checkpoints](training.md) for loop/source state and frozen-base
partial checkpoints.

## Qualification scope

Saved-plan two-H100 checks cover common sharded DP, reversed logical-rank maps,
forward/backward/AdamW execution, save/recompute choices, bounded initialization,
checkpoint restoration and shared evaluation. Tiny catalog full-training and
LoRA also have separate distributed checks. These do not qualify every model,
provider, repeated-replica layout or arbitrary model-owned communication path.
The migrated baseline controller has also passed full-training and LoRA checks
on two H100s. The [phase guide](../../architecture/offline-pipeline.md) describes the
implementation boundary; [phase artifacts](artifacts.md) describes persistence.

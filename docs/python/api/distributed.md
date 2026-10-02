# Distributed planning and training

Two-device DP has been checked with actual compiled Runtime execution: save and
recompute, full and sharded optimizer state, FP32/FP16, optional FP32 masters,
checkpoint restore, Forward evaluation, and distributed quickstart. Real-data
Llama training also completed 50 updates at DP=1, 2 and 8 with the same global
batches and initialization. Mixed expert-parallel integration remains to be
qualified.

## Ownership and setup

Each process selects one device and owns one Runtime. Create a CPU Gloo control
group first. Runtime checks all participants' combined host reservations before
pinning spill memory. Create accelerator communication groups and group-backed model resources after
entering Runtime or the `ShadowSpill` backend. Process groups belong to the caller.

```python
import torch
import torch.distributed as dist
from shadowspill.training import Distributed, Trainer
from shadowspill.training.backends import ShadowSpill

# Launch with torchrun. device="auto" resolves LOCAL_RANK against visible GPUs.
dist.init_process_group("gloo")
try:
    with ShadowSpill(
        device="auto", execution_gib=20, spill_gib=40,
        control_group=dist.group.WORLD,
        host_headroom_gib=2, preparation_timeout=1800,
        artifact_store=f"runs/example/rank-{dist.get_rank():05d}/artifacts",
    ) as backend:
        group = dist.new_group(backend="nccl", device_id=backend.device)
        try:
            model = make_model()  # ordinary initialized CPU model
            with Trainer(
                model, objective=objective, optimizer=torch.optim.AdamW,
                optimizer_args={"lr": 3e-4, "foreach": False},
                backend=backend, distributed=Distributed(group),
            ) as trainer:
                trainer.prepare(example)
                trainer.fit(source, steps=1000)
        finally:
            dist.destroy_process_group(group)
finally:
    dist.destroy_process_group()
```

`make_model`, `objective`, `example`, and `source` are application code. Sources
supply rank-local data; the trainer does not silently shard an iterable. An accelerator-backed
model or communication library can use `backend.device` and caller-created groups.
`host_headroom_gib` reserves process/staging room per participant in the host check;
execution and spill budgets are also per process.

The generic PyTorch eager/compiled backends currently accept single-process
training. Distributed ownership in this API is implemented by the ShadowSpill
backend and lower-level PyTorch planning APIs.

## Data loading

Each process can use a standard PyTorch Dataset/DataLoader. Select samples with
a sampler for the **data-replica group**, which may differ from the group that
owns or communicates model parameters. The trainer does not infer data ownership.

For equal local batch sizes, PyTorch's `DistributedSampler` is a conventional
choice; call `set_epoch(epoch)` when shuffling across epochs. Its epoch-tail
policy either drops samples or adds indices to make equally sized partitions.
It does not by itself preserve an arbitrary exact global batch size.

For an exact global update that does not divide evenly across data ranks, use a
caller-supplied batch sampler or iterable. For example, 512 sequences across
seven ranks can be partitioned as 74 on one rank and 73 on each other rank.
Supply matching microbatch/task counts, potentially with different tail shapes,
and scale each summed loss by the same global normalizer. This partition logic
is application code today; no dedicated sampler is required by the trainer API.

DataLoader workers should return CPU data. Choose worker counts per rank within
the host budget. Leave `pin_memory=False` unless an application has measured a
benefit alongside ShadowSpill's pinned spill pool. Exact checkpoint continuation
requires a source that saves and restores its progress, including any worker
prefetch state that affects the next item.

## Parameter replicas and gradients

`Distributed(group)` declares copies of every parameter across that participant
group. For mixed layouts, override registered parameter names or objects:

```python
ownership = Distributed(
    participants,
    replica_overrides=[(model.expert_parameters(), None)],
    groups={"experts": expert_group},
)
```

`None` means a unique local parameter. For replicated groups of expert shards,
use the group containing copies of that same shard instead. Shapes alone never
establish parameter identity. Tied names retain their single logical parameter.
Distinct Parameters with overlapping storage spans are rejected during setup;
use one Parameter object for tied weights. Disjoint slices of a flat bank work.
Ordinary mutable buffers remain rank-local; model code explicitly synchronizes
them if needed.

Remaining gradient contributions default to the parameter's replica group and
are **summed**. Advanced `gradient_group`/`gradient_overrides` describe cases where
model backward already completes some replica reductions. Those groups must lie
within the declared replicas. No automatic world-size division is applied.

The caller defines objective normalization. For an additive objective, every
rank/microbatch contributes `loss_sum / global_normalizer`. That denominator can
count examples, valid tokens, weighted observations, or another application-defined
quantity. For a microbatch mean, multiply by its contribution weight divided by
the global normalizer. The generic trainer does not infer this from tensor shapes.

## Optimizer tasks

Pass a local optimizer. ShadowSpill inserts the gradient SUM and updated-weight
exchange into its optimizer tasks. `shard_optimizer=True` is the default on
`Trainer` and the training planners; optional masters and moments use the same
owners. Each master element and its moments have exactly one owner within the
parameter's replica group. After updating its shard, the owner casts the weights
to the model's compute dtype **before** all-gather; master-precision weights are
not gathered. `False` keeps complete optimizer state on each parameter replica.

Optimizer definitions remain independent of ownership policy. The current
sharded policy uses flat slices, supported by Torch Adam/AdamW/SGD and optimizers
optionally declaring `supports_flat_parameter_shards=True`. That flag is an
optimization capability, not a requirement for supplying an optimizer. Local
MLOps AdamW exposes it; its kernels perform no communication.

Other optimizers use `shard_optimizer=False` today, preserving original tensor
shapes. Ownership of whole matrices or coupled update groups will be a separate
policy, reusing the generic capture/state/task machinery. It is not yet
implemented. Distributed updates currently need a traceable tensor graph.

Model and optimizer state are ordinary tensors exposed to capture and profiling.
Explicit collective outputs, owned gradients, and any temporary updated-weight
buffers contribute to task object/workspace measurements. Device allocations made
directly by communication libraries remain external memory.

## Preparation and execution boundaries

Preparation coordinates initial replica values, captures, cache decisions,
profiling calls, task choices, and physical admission. Profile calls have matching
collective participation even when one rank already has a cached result. Candidate
selection uses the slowest rank's predicted duration and requires all ranks to
admit the same ordered task sequence. Memory schedules remain local.

A geometry that exhausts device memory during preparation is rejected only
after every rank has unwound that attempt. Its failure does not poison the next
geometry. Non-memory failures remain fatal; an unrelated peer error is never
hidden by a simultaneous allocation failure. This recovery does not cover a
worker blocked in a GPU driver call or an incomplete device collective.
Preparation/control timeouts bound peer coordination, not uninterruptible
kernel calls. Launchers must supervise whole-process failures; a driver reset
may still be needed before its memory can be reclaimed.

During execution, there is no added global barrier between tasks or updates.
Fetch and eviction follow each rank's own schedule. Communication can overlap
computation inside a task, but it must finish on that task's current stream before
its completion event. Cross-task outstanding communication is outside this initial
contract. Model code must supply truthful custom-op tensor/mutation contracts.

Named `groups` preserve caller-owned handles during fake-model construction and
bind explicit functional collective names in compiled artifacts. Restarted
processes recreate their groups; live process-group handles are not checkpointed.

## Checkpoints

Use one shared checkpoint directory and the same save/load call on every rank.
The initial scope is shared storage and unchanged topology. Each rank writes its
own data/RNG/buffer/optimizer state; a manifest is published only after all writes
succeed. The schema remains version 1.

- `trainer.save(path)` defaults to `weights="master"`: save owned masters when
  configured, compute weights otherwise. Rebuild compute weights by casting.
- `trainer.save(path, weights="compute")` saves compute weights only and upcasts
  each owned master slice on restore. Lost precision is not recovered.
- `trainer.fit(..., checkpoint_weights="compute")` selects the same policy for
  periodic saves.

Exactly one weight representation is saved per logical parameter. Optimizer
moments are separate state. Restore uses bounded CPU transfers rather than
assembling full master replicas on every rank.

## Startup diagnostics

With an optimizer advertising `zero_lr_preserves_state=True` (such as MLOps
AdamW), `trainer.fit(..., run_dir="runs/example", startup_diagnostics=True)`
runs a warmup and a traced step on every rank before the first recorded update.
Each rank saves diagnostics and simulated/traced occupancy timelines under
`runs/example/rank-NNNNN/startup/`. These invocations do not advance the trainer,
data source, schedules, or W&B records. The first data item is reused for the
first real update.

The optimizer executes its normal kernels and planned communication at LR=0,
while retaining weights, masters, moments and step counters. The runner restores
model buffers changed by forward/backward and RNG state separately; it does not
copy all parameter or optimizer state. See the [training API](training.md#startup-diagnostics)
for limits and the explicit `trainer.diagnose` call.

### Initial replica synchronization and mapped checkpoints

Replica initialization for `import_model_state` runs after the weights have
been imported. It writes the imported, pool-backed weights and leaves the
caller's source tensors untouched. This also applies when Trainer imports a
model loaded with `torch.load(..., mmap=True)`: synchronizing the source would
turn file-backed pages into a full private host copy on every rank, outside the
reserved spill pools.

The default still synchronizes replicated parameters from their first rank.
Rank-local parameters and buffers keep their local values. Optional masters and
optimizer state are initialized only for each rank's owned shards. For a pool
that cannot be directly addressed by this process, synchronized host views are
published back to the pool before import returns.

# Distributed planning and training

For ordinary data parallel training under torchrun, the high-level
[`DataParallel` configuration](training.md#data-parallel-training-and-resource-ownership)
lets `Trainer.prepare()` create the groups and `Trainer.close()` release them.
It supports replicated compute weights or transfer-sharded weights/gradients,
with independent optimizer/master sharding enabled by default. The API below
provides explicit group and per-parameter ownership control.

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
Linux cgroup headroom includes clean, unmapped filesystem cache that the kernel
can reclaim. Mapped files, shared memory, dirty/writeback pages, and unevictable
memory remain charged. This avoids rejecting a repeated run solely because an
earlier checkpoint is still cached. Host and cgroup checks are estimates made
before allocation; concurrent users can change available memory afterward.

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

## Computation-weight and gradient spill sharding

`TransferSharding` describes the CPU representation of computation weights and,
when `shard_gradients=True`, gradient accumulators. It is independent of the
optimizer ownership described below. Keep `Distributed.group` and the parameter
replica group equal to the full data-parallel group; select a smaller local
group through `transfer_sharding`.

Frozen tensor representations inherit this layout for every physical component,
including compressed data and quantization scales. CPU admission counts those
components' physical shards; GPU placement still reserves their full padded
execution storage.

Gradient reduce-scatter reuses the source allocation, so intermediate gradient
offloads use eviction. A retained write-back with a collective eviction callback
is rejected at task admission.

| State | Ownership control | Typical per-rank persistent storage |
| --- | --- | --- |
| Computation weights in the spill pool | `TransferSharding.group` | Weight bytes / local shard-group size |
| Gradients in the spill pool | The same group, enabled by `shard_gradients` | Gradient bytes / local shard-group size |
| FP32 masters and optimizer moments | `shard_optimizer=True`, over parameter replicas | State bytes / full replica-group size |
| Computation weights and gradients used by a task | Full execution tensor | Full tensor bytes |

Shard sizes include alignment/tail padding. For example, four data-parallel
ranks can use local shard groups `[0, 1]` and `[2, 3]` while masters and optimizer
moments remain partitioned across `[0, 1, 2, 3]`. Each local shard group holds one
complete set of CPU weight/gradient shards. The optimizer has one globally
partitioned set of masters/moments; it is not replicated once per local shard group.

```python
from shadowspill.training import TransferSharding, Distributed, Trainer

# Groups are caller-created. Every process creates them in the same order.
# For rank 0 in this four-rank example:
# dp_group: [0, 1, 2, 3] (NCCL)
# local_shards: [0, 1] (NCCL), local_host: [0, 1] (Gloo)
# matching_owners: [0, 2] (NCCL)
distributed = Distributed(
    dp_group,
    transfer_sharding=TransferSharding(
        group=local_shards,
        replica_group=matching_owners,
        host_group=local_host,
    ),
    shard_gradients=True,
)
# Supply distributed=distributed, shard_optimizer=True, and (if desired)
# master_dtype=torch.float32 to Trainer or the training planning API.
```

`replica_group` here means corresponding shard owners in different local shard
groups, not the full data-parallel group. In the example, rank 1 uses `[1, 3]`.
With one local shard group spanning all parameter replicas, it is `None`.
`transfer_sharding_overrides` can select layouts by parameter names or objects.
All shard groups for a parameter must have equal sizes, partition its replicas,
and agree on corresponding owners. The current gradient protocol requires
unreduced contributions from all of that parameter's replicas; it rejects
gradient overrides that have already reduced those contributions.

A weight fetch copies the owned CPU shard and then all-gathers the execution
tensor within its backing group. A gradient eviction reduce-scatters local
contributions within that group, SUM-reduces the result across corresponding
owners, and copies only the owned shard to the CPU. These phases run through the
transfer lane; object readiness/reuse waits for the whole sequence. The simulator
prices the host-copy bytes and calibrated collective duration as separate phases.

Profiling also measures interference when transfer sharding is configured. It
observes each task's internal NCCL intervals and reserves the shared collective
queue only for those segments. For tasks without internal collectives, paired
probes run the compiled task alone and alongside the actual sharding operations.
Small, middle, and large message sizes come from the program's execution objects.
The extra probe buffer must fit beside the task's measured workspace. When a
copy is large relative to a task, the probe uses up to eight task calls, draining
each call's allocator scope before the next. Communication is submitted first
for these sequences, providing complete overlapping copies without retaining
multiple task workspaces. At most 128 copies are submitted per sample.

The simulator uses separate compute and collective progress rates during actual
overlap. Calibration compares copies wholly inside the compute interval;
post-task tail delays and unstable or nonphysical estimates are excluded. At
least three consistent trials are required. Equal-size groups use conservative
shared overlap envelopes while retaining their separate isolated transfer costs.
Missing measurements retain isolated timing. These pairwise rates approximate
GPU contention; they do not model arbitrary multiway interference or every
rank-arrival delay. Compare predictions with measured steps when qualifying a
sharded configuration; no full-step multiplier is fitted.

Distributed task conditioning and sampling use agreed fixed-count bursts of
roughly 100 ms, capped at 128 calls. Allocation scopes still drain after each
call. Host control collectives occur between bursts so they do not repeatedly
interrupt the sustained workload being measured.

Profile and planning cache identities include NCCL version, communicator resource
settings and stream priority, plus NCCL environment overrides. Changing a group's
CTA limits or policy therefore requires fresh costs even when its membership is
unchanged. The task measurement policy also participates in the cache key, so
old isolated-only profiles cannot bypass the new probes. Moving a diagnostic
output file alone does not invalidate profiles.

### Repeated gradient evictions and optimizer steps

Let $G$ be the gradient accumulated before an eviction and let $\Delta_r$ be
rank $r$'s contribution from the next set of microbatches. CPU shard copies in
different backing groups contain the same global $G$. On reload, exactly one
matching owner restores each element of $G$; the other owners start from zero.
The full GPU gradient buffer is zeroed before its owned carry is restored.
Consequently, the next two-level SUM gives

$$
G_{\mathrm{next}} = G + \sum_{r=0}^{D-1} \Delta_r,
$$

where $D$ is the parameter's full replica count. Restoring $G$ in every backing
group would incorrectly multiply it by the number of groups. Reduce-scatter
reuses the existing full GPU gradient buffer for its output shard; no second
full gradient buffer is retained for the carry.

At the optimizer boundary, the remaining sparse carry and local contributions
are SUM-reduced into the optimizer's independently sized ownership slices. Each
owned master/moment element is updated once per training step. Intermediate
gradient evictions do not advance the optimizer step counter or apply weight
decay. Updated computation weights are cast and gathered across parameter
replicas before their local backing shards are stored. Neither reduction adds
division by the local group size or number of backing groups: objective
normalization remains the caller's single global normalizer.

Text quickstart exposes `--transfer-shard-size N` and `--shard-gradients` (or
`--no-shard-gradients`). It creates contiguous local shard groups and matching
owner groups across them. `N` must divide the data-parallel world size.
These options leave optimizer/master ownership over the full replica group.
Sharded transfers currently require pinned host spill memory and coordinated
symmetric planning, so every member submits compatible collective sequences.

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

Ordered custom operators keep their compiler ordering dependencies inside each
captured task. These empty PyTorch tokens are not model inputs, saved activations,
or communication buffers. Recompute uses one backward ordering chain for replayed
forward calls and derivative calls. An operator that is eligible for replay must
remain correct when called again with the same explicit inputs; ordering alone
does not make an arbitrary state update safe to repeat.

For an operator that allocates its own outputs, its fake implementation must
also describe backing storage, including padding behind a returned view.
Compilation records that extent; profiling checks the actual allocation
against it. A smaller logical tensor shape does not imply a smaller allocation.

Named `groups` preserve caller-owned handles during fake-model construction and
bind explicit functional collective names in compiled artifacts. Restarted
processes recreate their groups; live process-group handles are not checkpointed.

## Verified symmetric planning

Opt in with `Distributed(group, symmetric_planning=True)`. The default is
`False` without transfer sharding; specifying transfer sharding enables and
requires it. This option applies to
`plan_step_search`, `plan_step`/Trainer, and `plan_forward`/Forward.

All ranks still capture and profile their own tasks, including communication.
After profiling, the fast path verifies matching task/object identities and
dependencies, sizes, aliasing, mutations, allocation traces, initial/final
residency, effective capacities, alignment, and scratch reservations. Local
device ordinals and compiled-code digests may differ; executable code and
allocator bindings are never shared. Unequal requirements produce a visible
fallback message and use the existing per-rank search, or fail preparation when
transfer sharding requires a common schedule.

Transfer-sharded compilation uses Inductor's baseline instruction order instead
of its independent peak-memory reordering pass. This avoids that pass selecting
different temporary allocation sequences across ranks; kernel autotuning remains
enabled. The effective ordering policy is part of the profile cache identity.
The agreement and physical-admission checks still apply to the measured result.

Autotuned kernels may use different anonymous temporary sizes on different
ranks. When allocation/reuse order, output ownership, capacities and workspace
peaks agree, shared placement reserves the maximum observed size of each such
temporary. The combined maxima must preserve the verified workspace peak;
otherwise sharing is refused. Runtime slots retain the certified offsets and
dependencies but use each rank's exact local allocation sizes and checks.
Original profiles are retained, and the planning evidence records this
reconciliation under `allocation_policy`.

For a verified problem, planning uses the maximum measured duration of each
task profile and transfer phase, minimum fetch/eviction bandwidths, and maximum transfer latencies
across participants. Original local profiling artifacts remain available.

- A sweep assigns each ordering's complete budget sequence to one rank. Budgets
  stay sequential, preserving reuse of smaller-budget plans. Different orderings
  search concurrently after all ranks finish the geometry's GPU profiling.
  Orderings that lower to the same problem reuse one answer.
- A standalone training plan distributes resolution candidates among ranks.
  A forward-only plan has one candidate, searched by one rank.
- Every received schedule is checked and physically admitted using local facts.
  The selected task sequence stays common; this adds no execution-time barriers.

Search-result messages reference the verified allocation facts by a
device-independent digest and the selected object capacity. Receivers reuse
their existing immutable task allocation records only when that digest matches,
then perform the usual schedule validation, placement admission, and simulation
replay. This avoids transmitting and reconstructing large allocation traces for
every ordering on every rank. Saved plan artifacts still contain the full facts.

The number of independent orderings/candidates limits parallelism. This reduces
duplicated CPU search; it does not remove capture or profiling on each device.
Each rank's artifact store records the verification/fallback decision, which
rank searched each result, and received schedules with retained resolutions under
`planning/distributed/`. Search reports retain the owning rank's search duration
and the transfer calibration used by the shared search.
Cache hits with `keep_resolutions=True` restore retained alternative plans so
resuming a sweep can distribute the same complete artifacts again.

Quickstart accepts `--symmetric-planning` and `--no-symmetric-planning`, or
`run(..., symmetric_planning=True)`. Omitting this override preserves the
factory's `Distributed` setting. Distributed factories and text presets use the
same option; model code needs no changes.

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
- `include_optimizer=False` omits optimizer state on every rank; use
  `checkpoint_include_optimizer=False` for periodic saves. A new Trainer resumes
  saved training progress with freshly initialized optimizer state on each rank.

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

When initializing sharded backing from a checkpoint, a whole-tensor copy from
ordinary CPU storage publishes its bytes directly. It does not read the old
destination or allocate a second full-sized destination buffer. The host-group
leader scatters the owned shards, including zero tail padding, instead of
broadcasting a complete tensor to every rank. Partial writes and copies from
an alias still read the previous storage contents. If one shard group covers
the entire replica control group, its authoritative shards already define one
shared value; initial synchronization does not reconstruct and repartition it.
Multiple shard groups still synchronize their values across replicas.

Checkpoint loaders can also avoid constructing the same source tensor on every
rank. Inside the initializer, `initialize_storage_(destination, fill)` allocates
scratch only on a whole sharded storage's host-group leader, calls `fill(scratch)`
there, and scatters the owned shards. Other ranks neither read the checkpoint
payload nor allocate that full scratch tensor. The callback must overwrite its
scratch completely and must not contain collectives or depend on executing on
every rank. A source-read error is propagated to all members before publication.
Ordinary random initialization can keep using its existing tensor operations.

```python
from shadowspill.pytorch import initialize_storage_

def initialize(model):
    initialize_storage_(model.weight, lambda value: value.copy_(read_weight()))
```

This helper takes a physical CPU tensor; compressed wrappers expose their data
and scale tensors separately. Partial views and nonsharded state use ordinary
copy semantics, preserving untouched storage. The GLM checkpoint loader accepts
an optional `write_tensor` callback with this signature, and both GLM quickstart
recipes use it. Cross-group replica synchronization still runs after import.

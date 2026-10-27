# SSD spill storage

For model-scale correctness checks, use the opt-in
[`numerical_ssd` qualification gate](../../../qualification/README.md#ssd-numerical-qualification).

`shadowspill.ssd.ssd()` configures an SSD-backed spill pool, temporary unless it
is opened on a pool image (below). It uses
normal Runtime ownership, object placement, fetch/evict routes, calibration and
completion events. The Linux extension is built with ShadowSpill; no separate
I/O library is required. Use an existing directory on a local filesystem that
supports direct I/O and file preallocation, such as ext4 on an NVMe drive.

## Configuration

<!-- source-signature: src/shadowspill/ssd.py:ssd -->
```text
ssd(
    *,
    capacity: int,
    directory: str | Path,
    staging_bytes: int = 256 << 20,
    chunk_bytes: int = 2 << 20,
    queue_depth: int = 16,
    image: str | Path | None = None,
    create_image: bool = False,
) -> SSDPool
```

The factory returns an immutable `SSDPool` configuration. Creating it validates
the directory and settings; Runtime later allocates and releases the storage.
`SSDPool.addressable` is `False`: its pool addresses are opaque tokens.

```python
from shadowspill import measure_communication, plan_program
from shadowspill.ssd import SSDPool

spill = SSDPool(capacity=64 << 30, directory="/local-nvme")
communication = measure_communication(spill=spill)
plan = plan_program(profiled_program, memory_budgets=budgets,
                    communication_info=communication)
plan.save("training.ssp")
```

In a **fresh execution worker**, reuse that configuration:

```python
from shadowspill import Plan, Trainer
from shadowspill.ssd import SSDPool

with Trainer(Plan.load("training.ssp"),
             spill=SSDPool(64 << 30, directory="/local-nvme")) as trainer:
    trainer.initialize(fill_tensor)
    result = trainer.step(args=examples)
```

`Executor(plan, spill=...)` exposes the same setting. Capacity must equal the
Plan's spill budget. Build and task profiling use ordinary allocation and do not
open the SSD pool. Planning only reads CommunicationInfo; it needs neither the
drive nor the SSD extension. Measurement creates a bounded scratch pool using the
real transfer lanes, without installing the execution allocator. A Plan measured
for another spill kind must be replanned before execution.

The phase [quickstart](../../../benchmarking/quickstart.md#spill-storage) accepts
`--ssd-spill /local-nvme`. Pinned host memory remains the default. Sharded
fetches read the owned payload through the selected backend before all-gather;
gradient evictions reduce before writing the owned payload. One completion event
covers storage I/O and collectives for pinned-host, SSD and remote storage.

| Argument | Default | Meaning |
|---|---|---|
| `capacity` | Required | SSD bytes reserved for pool objects |
| `directory` | Required | Existing directory on the target filesystem |
| `staging_bytes` | 256 MiB | Upper bound on host payload buffers owned by the pool and its lanes |
| `chunk_bytes` | 2 MiB | Pipeline chunk size; multiple of 4096, at most 1 GiB, subject to the filesystem's alignment |
| `queue_depth` | 16 | Number of host staging slots per direction; 1–1024 |
| `image` | None | A pool image: with `create_image` the file to create, which the pool then is; otherwise an existing image read in place as the pool's first `image_bytes` bytes, never written |
| `create_image` | False | Create `image` as the pool file and keep it at close, for a conversion that imports frozen state once |

With one fetch route and one evict route, 2 MiB chunks and depth 16, payload
staging uses **66 MiB + 8 KiB** at 4 KiB alignment: two 32 MiB rings, one 2 MiB
state-I/O scratch buffer and two edge buffers. Increasing the cap alone does
not allocate more slots. An insufficient cap fails at setup. Small control
counters, Python metadata, model initializer scratch and compiler/profiling
workspace are separate from this payload cap.

## A pool image: import frozen state once

Pool images remain a low-level facility while their adoption into saved Executor
plans is pending. `ssd(..., create_image=True, image=...)` creates the persistent
pool file. The image manifest describes named tensors and storage offsets;
`ssd(..., image=...)` opens those bytes read-only, with new state in a separate
scratch file. The existing GLM conversion tool uses this facility. Saved-plan
workers currently use bounded checkpoint loaders instead of adopting pool images.

## Bounded initialization and checkpoints

`fill_tensor(chunk, destination)` fills Executor-provided CPU buffers. Pinned
memory exposes the pool view directly; SSD and remote memory use bounded staging
and byte-range I/O. Master-weight initialization reads only the matching source
chunk. Model controls and checkpoint export use this same path, preserving tensor
dtypes, padding and local shard offsets without a full host model copy.

Checkpoint files are independent of temporary spill storage. Use Trainer's
checkpoint helpers or Executor's bounded tensor export. Never dereference an SSD
pool address as a CPU tensor: addresses are opaque tokens meaningful to the pool.

## Pipeline and lifetime

Fetch uses direct SSD reads into reusable pinned-host slots, then H2D copies.
Evict uses D2H copies into those slots, then direct SSD writes. Disk reads wait
for their producer dependency too: ordering only the later H2D copy would allow
a stale disk read. A slot becomes reusable after its previous consumer finishes.

Read-ahead spans object boundaries. Available slots can start the next queued
object's SSD reads while earlier H2D copies are pending. Queue depth bounds the
whole lane's staging ring, without allocating an extra buffer per object. A
separate readiness stream honors source dependencies independently of H2D
progress; data copies and their completion events retain route order.

I/O workers actively poll while transfers are pending and sleep when idle.
This trades CPU time (up to one thread per active lane) for lower handoff
latency. I/O workers never invoke GPU APIs. Device-visible counters bridge disk completion
and stream waits without making an I/O worker wait behind its own GPU work.
Fetch completion includes H2D completion; evict completion includes the SSD
write. Errors propagate through the ordinary runtime failure path and unblock
pending waits before teardown.

The pool preallocates an anonymous, unlinked file, or the named image file.
`Runtime.close()` drains and closes its lanes and files; process termination
also closes the file descriptors. A temporary pool cannot resume a job; an image
holds only the state imported into it once. Durable checkpoints are separate.
Direct I/O bypasses the filesystem page cache; SSD write completion is not a
claim of durable checkpoint publication.

## LoRA and measurement

Frozen parameters keep their unchanged SSD copy after a fetch, so they need not
be evicted after every use. Recomputation reduces saved intermediates, but its input activations remain
ordinary planned objects: the planner may keep them on GPU or evict them and
fetch them before recomputation. There is no activation-write prohibition or
LoRA-specific rule in the pool or lanes. Trainable LoRA weights, optimizer
moments and counters still require writes.

`measure_communication(spill=...)` measures the complete SSD/host/GPU route, both alone and with the
opposite direction active. Short write measurements describe cache-assisted
behavior; they do not establish sustained write throughput. For read-dominant
LoRA, emphasize initialized direct-read measurements and actual step traces.
Do not infer bandwidth from unwritten file extents, which may return zeros
without reading the drive.

The simulator prices solo and overlapping portions using their respective
calibrated bandwidths and explicit request latency. The saved communication
artifact records the probe sizes, raw rates and measurement method. Smaller probe sizes reduce calibration writes
and can affect the estimate; they do not change production transfers.

A Chicago RTX 5090 / Samsung 990 PRO validation measured about 7.09 GB/s for
SSD-to-GPU fetches, close to standalone direct reads around 7.05–7.12 GB/s.
These are machine-specific observations, not hard-coded planner bandwidths.

For extension loading, exact C configuration and ownership, see the
[C SSD API](../../c/ssd.md) and
[`shadowspill/ssd.h`](../../../csrc/include/shadowspill/ssd.h).

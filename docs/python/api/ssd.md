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
from pathlib import Path
from shadowspill.memory import device, transfer_route
from shadowspill.pytorch import Runtime
from shadowspill.ssd import ssd

spill_directory = Path("/local-nvme/shadowspill")
spill_directory.mkdir(parents=True, exist_ok=True)

with Runtime(
    pools={
        "execution": device(physical_capacity=8 << 30),
        "spill": ssd(
            capacity=64 << 30,
            directory=spill_directory,
            staging_bytes=256 << 20,
            chunk_bytes=2 << 20,
            queue_depth=16,
        ),
    },
    routes={
        "fetch": transfer_route(source="spill", destination="execution"),
        "evict": transfer_route(source="execution", destination="spill"),
    },
    calibrate=False,
) as runtime:
    # Small, initialized probes avoid a large SSD write benchmark during setup.
    runtime.calibrate_transfer_capabilities(
        large_copy_bytes=32 << 20, warmup_copies=1, measured_copies=3,
    )
    # Import model state, then plan and execute as with any spill pool.
```

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

`ShadowSpill`'s training backend accepts the same configuration:

```python
from shadowspill.training.backends import ShadowSpill

with ShadowSpill(
    execution_gib=8,
    spill_pool=ssd(capacity=64 << 30, directory=spill_directory),
    calibrate=False,
) as backend:
    backend.runtime.calibrate_transfer_capabilities(
        large_copy_bytes=32 << 20, warmup_copies=1, measured_copies=3,
    )
    # Trainer(..., backend=backend) or Forward(..., backend=backend)
```

`spill_gib`, if also supplied, is the planner's limit within the configured pool.
Otherwise it uses the pool's capacity. Existing `spill_gib=N` callers continue to
use pinned host memory.

## A pool image: import frozen state once

A model's frozen state is imported once into a pool created with
`create_image=True`; the pool file is the image, and
`write_pool_image(model, runtime=..., pool=..., path=..., frozen_state_id=...)`
writes `<image>.manifest.json`, which records every storage root's offset and
size and the tensors viewing it. After the runtime closes, `PoolImage.trim()`
cuts the file to the bytes the state reaches.

Every later runtime opens the pool on that image, `ssd(..., image=path)`, and
imports the model with `image=PoolImage.read(path)`: the tensors the manifest
names are verified by name, geometry and dtype and bound where they lie,
without a copy; the rest of the model, LoRA factors and all, is allocated and
initialized in the scratch file behind the image, as are optimizer state and
evicted activations. The initializer takes `adopted`, the names it must leave
alone, and a write into adopted state fails. The GLM-5.3-Flash workload's
`python -m workloads.models.glm53_flash.convert` is such a conversion, and the
quickstart's `--ssd-image` opens its result.

## Initialize without a complete host model

Construct on `meta`, then initialize after import. With the low-level API:

```python
import torch
from shadowspill.pytorch import import_model_state, release_model_state

with torch.device("meta"):
    model = make_model()
model = import_model_state(model, runtime=runtime, pool="spill")
# After every plan using the model has closed:
release_model_state(model, runtime=runtime)
```

Modules must support in-place `reset_parameters()`, or pass `initialize=fn` to
`import_model_state`. Trainer and Forward accept the initializer through
`prepare(..., initialize=fn)`. Checkpoint inputs can be memory-mapped. These
paths work for pinned-host and remote pools too; see
[state import](../../architecture/state-import.md).

Non-addressable pool tensors carry checked CPU metadata without a retained host
payload. Direct CPU arithmetic on them is invalid outside controlled setup.
Use an admitted callable to execute the model. Explicit state reads/exports
produce CPU copies; use `release_model_state` when those copies are unwanted.
`PlannedTrainStep.save()` streams pool objects to a separate checkpoint file.

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

Runtime owns calibration and measures the complete SSD/host/GPU route, both alone and with the
opposite direction active. Short write measurements describe cache-assisted
behavior; they do not establish sustained write throughput. For read-dominant
LoRA, emphasize initialized direct-read measurements and actual step traces.
Do not infer bandwidth from unwritten file extents, which may return zeros
without reading the drive.

The simulator prices solo and overlapping portions using their respective
calibrated bandwidths. Request latency remains diagnostic metadata and does
not add simulated transfer time. Smaller probe sizes reduce calibration writes
and can affect the estimate; they do not change production transfers.

A Chicago RTX 5090 / Samsung 990 PRO validation measured about 7.09 GB/s for
SSD-to-GPU fetches, close to standalone direct reads around 7.05–7.12 GB/s.
These are machine-specific observations, not hard-coded planner bandwidths.

For extension loading, exact C configuration and ownership, see the
[C SSD API](../../c/ssd.md) and
[`shadowspill/ssd.h`](../../../csrc/include/shadowspill/ssd.h).

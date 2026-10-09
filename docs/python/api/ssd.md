# SSD spill storage

For model-scale correctness checks, use the opt-in
[`numerical_ssd` qualification gate](../../../qualification/README.md#ssd-numerical-qualification).

`shadowspill.ssd.ssd()` configures a temporary SSD-backed spill pool. It uses
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

I/O workers never invoke GPU APIs. Device-visible counters bridge disk completion
and stream waits without making an I/O worker wait behind its own GPU work.
Fetch completion includes H2D completion; evict completion includes the SSD
write. Errors propagate through the ordinary runtime failure path and unblock
pending waits before teardown.

The pool preallocates an anonymous, unlinked file. `Runtime.close()` drains and
closes its lanes and file; process termination also closes the file descriptor.
The pool is temporary and cannot resume a job. Durable checkpoints are separate.
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

The existing simulator uses fixed effective rates from concurrent calibration,
with solo measurements retained for diagnostics. It does not dynamically switch
rates when only one lane is busy. Smaller probe sizes reduce calibration writes
and can affect the estimate; they do not change production transfers.

A Chicago RTX 5090 / Samsung 990 PRO validation measured about 7.09 GB/s for
SSD-to-GPU fetches, close to standalone direct reads around 7.05–7.12 GB/s.
These are machine-specific observations, not hard-coded planner bandwidths.

For extension loading, exact C configuration and ownership, see the
[C SSD API](../../c/ssd.md) and
[`shadowspill/ssd.h`](../../../csrc/include/shadowspill/ssd.h).

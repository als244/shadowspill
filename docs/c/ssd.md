# SSD pool and lanes

`libshadowspill_ssd.so` adds temporary SSD spill storage through the existing
[pool-memory](pool-memory.md) and [lane](lanes.md) contracts. It registers one
pool kind and two directional lanes: SSD → device (fetch) and device → SSD
(evict). Allocation, admission, object lifetimes and scheduling remain the
ordinary runtime mechanisms.

## Build and registration

The extension builds on Linux with the main project. It uses direct file I/O,
Linux AIO syscalls and pthreads; it needs neither `libaio` nor a GPU SDK. The
selected backend must support pinned-host registration and stream value
waits/writes. Unsupported filesystem or backend operations fail setup; there
is no silent buffered-I/O fallback.

Load the library and resolve `SHADOWSPILL_LIBRARY_DESCRIBE_SYMBOL`
(`shadowspill_library_describe`) using the types in
[`shadowspill/runtime/library.h`](../../csrc/include/shadowspill/runtime/library.h).
Check the returned `ShadowSpillLibraryDescription.abi_version` against
`SHADOWSPILL_ABI_VERSION`. Append its `pool_memory` and `lanes` entries to
`ShadowSpillRuntimeConfig`, preserving other registered extensions. Keep the
library loaded until the runtime has been destroyed. Loading or describing it
performs no allocation or device probing; runtime creation does that.

## Configuration

[`shadowspill/ssd.h`](../../csrc/include/shadowspill/ssd.h) declares:

```c
#define SHADOWSPILL_SSD_POOL_KIND 3U

typedef struct ShadowSpillSSDConfiguration {
    const char *directory;
    uint64_t staging_bytes;
    uint64_t chunk_bytes;
    uint32_t queue_depth;
} ShadowSpillSSDConfiguration;
```

Set a spill pool's `kind` to `SHADOWSPILL_SSD_POOL_KIND`, its `capacity` to the
SSD byte budget, and its `configuration` to this struct. Supply every field;
the C struct has no implicit defaults. These values match the Python defaults:

```c
ShadowSpillSSDConfiguration ssd = {
    .directory = "/local/ssd/shadowspill",
    .staging_bytes = UINT64_C(256) << 20,
    .chunk_bytes = UINT64_C(2) << 20,
    .queue_depth = 16,
};
```

The directory must already exist on the desired filesystem. Keep configuration
storage, including the directory string, alive for the runtime's lifetime.
Routes name pool IDs normally; their endpoint kinds select the SSD lanes.
No per-route SSD configuration is needed.

| Setting | Meaning |
|---|---|
| Pool `capacity` | Bytes reserved on SSD; the file reservation rounds up to direct-I/O alignment. Must fit a signed 64-bit file offset. |
| `directory` | Directory for the unlinked pool file. Choose a local SSD filesystem for local SSD performance. |
| `staging_bytes` | Shared cap on host payload buffers for state I/O and all lanes using this pool. |
| `chunk_bytes` | Positive pipeline chunk size, divisible by the filesystem's direct-I/O alignment (at least 4096 bytes). |
| `queue_depth` | Positive number of chunk slots per directional lane. |

For one fetch and one evict route, chunk size `C`, queue depth `Q`, and
alignment `A`, host payload storage is `C + 2 × (Q × C + A)`: one state-I/O
scratch buffer and two rings with edge scratch. The example above consumes
**66 MiB + 8 KiB** at 4 KiB alignment. Rings are pinned; state/edge scratch is
aligned host memory. A larger cap alone creates no extra slots. Setup fails if
the configured buffers exceed the cap.

Control counters, framework initialization scratch and compiler/profiling
workspace are additional. The cap is not a limit on whole-process host memory.
The [Python API](../python/api/ssd.md) applies additional configuration bounds
and supplies a complete runtime example.

## State access and lifecycle

Pool acquisition preallocates a direct-I/O file with no persistent name
(`O_TMPFILE`, or a temporary file unlinked immediately). It returns an address
token that cannot be dereferenced. `read` and `write` are synchronous pool-state
operations used for import, initialization and export, outside scheduled lanes.
They handle partial sectors using bounded scratch and preserve neighboring
bytes. Disk-space exhaustion and short or failed I/O are errors.

Normal runtime close drains transfers, destroys the lanes, then releases the
pool and closes its file. The OS also closes the file on process termination.
The pool cannot resume across process restarts; durable checkpoints use a
separate file. Direct-I/O completion does not itself promise checkpoint
durability.

## Ordering and completion

Fetch pipelines SSD reads into pinned slots with H2D copies. Evict pipelines
D2H copies with SSD writes. A producer dependency gates the disk read itself,
preventing stale reads before an earlier write finishes. A slot is reusable
only after its consumer completes: H2D for fetch, disk write for evict.

The runtime calling thread enqueues GPU operations on the route stream. The
I/O worker never calls the GPU backend; it uses host-visible counters to
publish disk progress and release stream waits. The event given to `signal`
covers the entire transfer. An I/O error latches runtime failure before
releasing pending waits, so failed work cannot silently pass as successful.

Normal close drains first. Abandon stops the disk worker without waiting for
an unfinished GPU producer; submitted OS I/O is canceled or retired before
its staging memory is freed. Per-transfer trace instants use the runtime's
host/device clock anchor. Byte and chunk counters are always available;
aggregate lane timing is optional and is not implemented by this extension.

## Calibration and planning

Runtime owns calibration through these same lanes and actual pool ranges.
It initializes non-addressable probe sources before measurement; unwritten
file extents can otherwise report unrealistic read rates. Runtime measures
solo and simultaneous fetch/evict traffic, then publishes effective rates for
the planner. The simulator currently uses fixed effective rates; it does not
switch between solo and concurrent rates during a step.

Probe sizes and iteration counts are caller-configurable. Smaller probes
reduce setup writes but can change the bandwidth estimate, especially for
cache-assisted SSD writes. They do not change chunk sizes or training
transfers. Neither pool nor lane inspects model roles: frozen parameters,
trainable state and recomputation inputs follow ordinary object lifetimes.

# Framework-neutral runtime

`libshadowspill_runtime.so` consumes explicit task-boundary actions and owns
device-slab allocation, host backing, object residency, transfers, readiness,
retirement, failures, and teardown. It contains no PyTorch, CUDA, HIP, tensor,
optimizer, model, or operation type.

## Ownership model

A runtime instance owns one device slab, one bounded host arena, one H2D stream,
one D2H stream, and one progress thread. The frontend owns compute streams and
submits numerical work.

An allocation is an ordinary lease from the coalescing slab. Logical free
records events on every associated stream; its range is not reusable until all
events complete. If a request cannot fit, it blocks only while a retirement,
release, or offload can create a suitable range. Otherwise it returns a
structured no-progress OOM.

An object record represents one complete alias group. It tracks the current
allocation, residency generation, authoritative/device/host versions, host
backing, and readiness event. Tensor views are deliberately absent: a frontend
maps every view of an alias group to this one record.

Frontends populate initial pinned backing with
`shadowspill_write_host_object` before device materialization. After an
explicit terminal writeback and idle boundary, `shadowspill_read_host_object`
copies current bytes into ordinary caller-owned CPU storage. Both operations
require exact object size and a `HOST_ONLY` state, so neither can race an
asynchronous transfer.

## Task protocol

`shadowspill_before_task` deduplicates input object identities, validates their
authoritative device generations, inserts one compute-stream wait for every
unfinished H2D, and returns current addresses. A host-only input is a plan
violation unless its annotated prefetch is already queued; in that case the
caller waits only until the progress service admits the destination and then
uses a stream-event dependency.

`shadowspill_after_task` records one compute fence, applies declared object
version updates, and enqueues actions in their exact supplied order. Release
waits for task completion. Offload and prefetch use transfer-stream waits and
events; they do not synchronize the host or device. A normal offload queued
immediately after consuming an object whose H2D is still in flight is valid:
the compute fence already orders both prerequisites.

The progress worker polls completions, returns ranges, updates versions and
residency, wakes blocked allocators, and preserves the first failure. It never
launches framework work.

The host poll is not the device timeline. Once `before_task` has inserted a
wait, the consumer may be queued and `after_task` may advance the device version
while the object still appears as `PREFETCHING` to the worker. H2D completion
therefore changes readiness only; it never overwrites a newer device version.

Runtime statistics expose total/free/peak slab bytes, largest free range,
external fragmentation, host-arena occupancy, blocked allocators, pending
retirements, transfer counts and bytes, and inserted wait counts. Allocation
failure snapshots preserve both total free bytes and largest contiguous range
so capacity exhaustion is distinguishable from fragmentation.

## Backend contract

`ShadowSpillBackend` is a versioned vtable over conventional device/host
allocation, opaque streams and events, asynchronous copies, event queries, and
stream waits. Opaque tokens contain no vendor type in the core ABI. Phase 4's
mock backend implements deterministic delays and failure injection; CUDA and
future HIP backends implement the same contract.

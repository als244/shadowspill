# Durable service records

`shadowspill.server.records.Journal` is the service's filesystem persistence
primitive. It owns ordered durable metadata, independently of training state,
worker dispatch and HTTP. Contiguous acceptance, dependency handling and restart
classification belong to the request coordinator; the journal does not make an
interrupted GPU update replayable.

## Authenticated transport

`server.http.HTTPService` adapts the existing `RunController`, `RequestRecords`
and `ArtifactStore` to authenticated HTTP. It does not own another execution queue.
Deployment-configured bearer credentials resolve to an owner; run URLs carry only
session/run/request IDs. Lookup contexts retain their owners during I/O, and the
handler verifies the returned owner/session/run identity before accessing records.
The credential table retains SHA-256 digests rather than plaintext tokens.

The service reserves scratch and a concurrent-handler slot before starting a
thread, including TLS handshakes. Request headers have a total 64 KiB bound.
Portable control metadata is at most 1 MiB with 16,384 JSON nodes and 64 levels
of nesting; a constant-space scan checks structural bounds before JSON allocation.
Binary bodies have one explicit uncompressed length and reuse admitted input
decoding. Each underlying socket read checks the whole exchange deadline, so
partial headers cannot retain an owner indefinitely. Network timeouts apply to
individual exchanges; live training runs still have no default idle timeout.

Routes register immutable request metadata, finish binary uploads, query or wait
up to ten seconds for recorded acceptance/completion, and attach a writer. A
pending writer attachment returns its same lease with HTTP 202 until ready.
Connections close after each exchange. Accepted execution continues independently;
a lost HTTP response has no inferred optimizer effect and never causes replay.
Known request IDs and outcomes remain queryable after worker close or record-store
reopening. Read-only stores refuse new submissions and cannot restore live state.

Artifact downloads resolve authenticated owner-scoped immutable references, stream
64 KiB blocks and advertise the SHA-256 digest as the ETag. The lightweight SDK
transport verifies length, type, digest and partial destination writes without
loading a whole artifact. It does not redirect credentials or retry a mutation.
Reachable listeners require TLS; loopback HTTP also supports SSH forwarding.
TLS verification, owner isolation, disconnects, incomplete uploads and capacity
limits have real loopback tests. Ordinary/provider GPU tests use these routes for
updates, discard, close and downloads after worker exit. Public service creation,
SDK training handles, owned SSH forwarding, range-resumable transfers and actual
multi-node deployment remain separate implementation work.

## Request and worker ownership

`server.owners.OwnerCache` bounds open record/artifact owners, including pending
opens and failed evictions. Live controllers pin a borrow for their lifetime;
HTTP lookups borrow through I/O. Only idle owners may be evicted. Factories and
cleanup run outside the cache lock, so a slow journal rebuild does not block
unrelated cached owners. Full or pending entries return prompt backpressure.
Shutdown refuses new borrows and retains active or failed owners for cleanup
retry. An owner supplies its own byte reservation; the cache bounds its count.
Request-record index cleanup now keeps the exclusive journal lock until it
succeeds, and a partially closed index cannot serve another lookup.

`server.records.runs.RunCatalog` records creation identity and run lifecycle under
the same durable journal rules. It stores bounded `RunInfo` metadata and immutable
configuration/Plan references; tensor state and update effects remain in their
existing owners. Every lookup and paginated listing includes owner/session scope.
An identical creation retry returns the original record, including after restart
or close; conflicting reuse fails before another job can be enqueued.

Creation stages advance through installed phases and initialization. Only the
caller that confirmed initialized worker readiness may record a ready run. Old
unfinished runs are enumerated in bounded pages for service startup to classify
their request journals and mark them stopped. Catalog recovery alone performs no
worker launch or replay. Failed/closed/stopped records remain immutable, and a
new server cannot advance an earlier incarnation's creation. The catalog owns a
fixed index reservation; durability/index failure requires reconciliation, and
failed index cleanup keeps the reservation and writer lock until retry succeeds.

Preparation uses the existing `training.preparation.phases.run_phases` driver,
which accepts an explicit environment snapshot and an installed launch callback.
`orchestration.launch.phases.NodePhases` implements the allocation-owned callback.
It reserves the selected node slots and an explicit whole-phase host bound, then
uses the existing `OwnedProcess` supervisor through `NodeLease.launch_command`.
An ordered assigned-device list becomes the child visibility mask before imports;
CPU planning reserves host capacity only and has no visible GPUs. Every owned
leader and descendant must finish before its reservation is released or the next
phase starts. Disjoint jobs never change the controller's process-wide environment.

Phase stdout/stderr use the same `LogFiles` ownership as persistent workers, capped
before launch and retained after process exit for publication. Truncation and write
failures appear in launcher events. Failure summaries read only bounded log tails.
A job can stop its own waiting reservation or running child without closing the
shared allocation or touching another run; failed termination retains its lease,
and failed log deletion retains disk and scratch claims for retry. The caller must
join a stopped launch before closing its log owners. This adapter currently runs
one node's installed phases; distributed agent bootstrap remains separate work.

The engine's `training.session.staging.OrderedStaging` grants complete input
pipeline reservations in submission order. A bounded metadata window leaves room
for an earlier request whose metadata has not arrived. A later upload cannot
consume the bytes needed by that earlier pipeline and then wait behind it forever.
After reservations are granted, producers may finish out of order. A ticket hands
its owner the existing staging arena; the owner closes all payloads before closing
the ticket. Timeouts preserve tickets, and shutdown wakes waiters while acquired
owners retain their storage until they release it. Reservation is not durable
request acceptance. The coordinator supplies the admitted pipeline bound, connects
validation to acceptance, and handles expired or fenced producers.

`training.session.inputs.RequestInputs` connects those reservations to binary
Datum decoding, checksums, complete-call preprocessing and durable acceptance.
The coordinator supplies a trusted prepared pipeline bound and input envelope.
Decoding reserves its complete prepared peak before reading the body. The frame,
header, tree, tensor-total and largest-leaf bounds, plus the exact Datum count,
are checked before tensor allocation. Metadata pressure is
checked before persisting a new ID; fenced or conflicting submissions consume no
sequence slot. A decoded batch must match the request's Datum count, and the
installed preprocessing callback must finish all fallible content work before
returning. Identical retries inspect the original status without copying again.
Execution borrows accepted prepared inputs once and explicitly releases them after
its consumers finish. Failed queued inputs close promptly; rejected active readers
retain storage until they return. Waiting for an arena does not hold the ordering
lock needed by earlier cleanup. Fatal shutdown can release idle owners even when
record storage is unavailable, while active owners retain their leases. HTTP body
reads enforce the exchange deadline described above without changing the request's
durable execution classification.

`training.session.prepared.PreparedRequests` supplies that callback and derives
registration bounds from the rechecked prepared family. Whole-call validation
runs once; actual capacity must fit one original requested batch option. It
selects saved schedules by predicted cost, retaining the admitted coverage
witness when bounded packing search stops. No shape, count or Plan is added at
execution time. Grouped definitions provide an installed `input_groups(config,
validated_batch)` hook; its bounded index lists must cover every Datum once.
Per-call controls resolve nonsticky defaults and validate their actual captured
float32/int64/bool representation before preprocessing or acceptance. Validity
determines participation, independently of numeric weights or final scale.
Prepared calls retain the assignment, controls and completed spool until their
consumers finish. Main, masked and optimizer-only Plan selection follows the
existing `OperationState` transition, including empty optimizer skips.

`training.inputs.spool.InputSpool` finishes every prepared buffer/rank adapter
once and stores its fixed input arrays under an explicit temporary cache root.
Workers consume ready binary Datum frames without rerunning content-dependent
preprocessing between GPU microbatches. The existing layout admission includes
frame metadata, a 64 KiB streaming scratch block, serial decode peaks and the
exact temporary file volume. Frames retain physical backing arrays plus generated
metadata so worker reconstruction preserves strided and empty rank slices and
bit-exact BF16 storage. The shared frame writer borrows contiguous arrays and can
stream strided arrays with bounded C-order scratch. File
writes and reads check layouts, sizes and checksums. Any preprocessing/write
failure deletes all earlier files. The owner removes files after input consumers
finish; they are ephemeral storage and never authorize replay after server loss.
`training.session.worker_inputs.ReadyInputLoader` reconstructs the saved input
views without rerunning model hooks and feeds the existing Executor input loader.
It keeps one decoded frame until its synchronous spill copy finishes. Executor
releases every prior example view before requesting the next frame; host bindings
retain only owned clones. Ordinary/provider cold-worker checks now cover actual
binary uploads, selection, spooling, shared updates, interposed evaluation,
invalid-input recovery and publication failure. Persistent worker/agent transport
and public service clients remain integration work.

`training.worker.protocol` uses the common bounded JSON codec with its own fixed
record allowlist. A worker command identifies the run, controller incarnation,
worker incarnation, logical rank, invocation, original request and saved candidate.
Input frames carry opaque artifact IDs, sizes and checksums. Commands contain no
filesystem paths, arbitrary Plan names or executable objects. Public request
decoding does not accept private worker records.
`training.worker.bindings.WorkerBindings` resolves commands against one rechecked
family and identity, using the same capability and captured-control validation as
controller preprocessing. It derives main, masked and optimizer forms from the
coordinator's decision, checks the complete rank-local frame ABI and verifies all
local input files before execution. The agent supplies an owned-file resolver;
network metadata cannot name local files. Worker reads use a separate bounded
arena and the existing serial input loader. This boundary does not duplicate
update/gradient state or provide command replay.

`training.worker.loop.WorkerLoop` owns one outstanding invocation on a private
control stream. A stage message checks every local frame and returns a ticket
bound to the complete command digest. A separate execute message consumes that
ticket before entering the existing Executor through `WorkerExecution`. Abort
releases a ready invocation without execution. Submission sequences can have
rejected gaps; worker invocation numbers are contiguous, and consumed submission
sequences cannot be reused. Duplicate, stale or mismatched messages invalidate
the channel. A staging error reports known unstarted effects and permits explicit
corrected work; an execute exception stops the worker with unknown effects.

The private channel checks its length prefix before reading bounded canonical
metadata and supports fragmented reads/writes. It has no idle timeout. EOF, a
truncated command or a broken reply leaves the worker invalid; its owning agent
must close the Executor/process. No reconnect or automatic replay can reuse that
live state. CPU tests cover actual child processes and death after execution but
before reply. The cold-worker GPU oracle exercises the same loop and Executor
adapter. Agent launch/monitoring, authenticated remote
transport and service output publication remain integration work.

`training.worker.group.WorkerGroup` coordinates the private connections: it checks
all command metadata before sending, waits for every staging response, then sends
execute to every participant before waiting for completion. A staging rejection
aborts every ready participant and waits for every abort acknowledgement. Only
that confirmed boundary raises `ExecutionNotStarted`, allowing the dispatcher to
retain prior gradients/counters while failing the already queued dependent suffix.
Corrected work uses a new ID. Missing staging/abort/completion replies or failures
after release invalidate the group and invoke its allocation agent's stop callback
for all owned children. No incomplete response is retried. Real socket barriers
and child-process failure tests cover the controller protocol; actual multi-node
GPU qualification remains required. Agents still need to provide admitted file
delivery, startup/supervision and the authenticated service connection.

`orchestration.launch.staging.FileDelivery` supplies the generic node-local copy
owner. It reserves complete disk volume and bounded metadata/copy memory before
opening any source or creating files. All artifact sizes and checksums must pass
before a worker can borrow the completed delivery. Consumers keep files alive
through execution; active producers retain reservations through cancellation, and
failed deletion retains quotas until cleanup succeeds. These owned files survive
sender cleanup. The launcher does not interpret models, objectives or tensor
layouts; the worker separately rechecks each prepared input ABI. Agent startup,
supervision and actual node-resource admission remain to be connected.

`orchestration.launch.supervision.OwnedProcess` supplies fresh process-group
ownership with bounded stdout/stderr draining and non-reaping exit observation.
The controller group closes private control streams before waiting on its stop
callback, permitting orderly worker/report finalization. Explicit close escalates
only the owned process group and removes descendants before reaping its leader;
there is no idle timeout. Agent launch, node-resource admission and service
authentication still need to be wired around these primitives.

`training.worker.runtime.PreparedWorker` describes a strictly rechecked saved
family before execution. It pins the immutable bundle reference, initialization
signature, launcher-to-logical rank permutation, device budget, existing spill
capacity (including its 5% margin) and serial initialization/input staging peak.
The CPU description imports neither Torch nor the optional workloads catalog.
An agent must verify artifact integrity and reserve decoded-artifact/process/node
resources separately; these byte fields describe the execution pools and staging,
not total process RSS. Worker startup requires one explicitly assigned visible
GPU and preserves saved compiler ordinal zero.

`WorkerStartupConnection` and `start_workers` use the same private framed stream.
Every worker first sends its CPU preflight evidence. The controller compares all
participants before releasing any into mutable startup, then releases all starts
before waiting on any ready response. A mismatch, premature ready, missing reply
or initialization failure closes all private channels and stops the owned group.
Readiness transfers each channel once to the ordinary `WorkerConnection`;
startup cannot be replayed. The worker's `PreparedWorker.serve` opens the existing
Executor, performs bounded installed initialization, checks actual spill capacity,
then reports ready and enters the existing invocation loop. EOF closes that
Executor. There is no second mutable-state engine or startup compilation.

Actual agent-launched GPU processes, node-wide host/device admission, RNG stream
registration, authenticated transports and public client readiness remain
integration requirements. Socket startup barriers and cold-worker Executor
composition do not qualify multi-node startup by themselves.

`orchestration.launch.connected.ConnectedProcess` gives the local agent an owned
child and an inherited private socket, independent of stdout/stderr. The existing
supervisor passes only explicitly granted private descriptors. Once all startup
responses pass, the caller marks the connection ready and removes its startup
I/O timeout. Normal live connections have no idle timeout. Closing the channel
precedes process-group cleanup and wakes blocked I/O; the parent retains no copy
of the child's socket end. Real CPU children cover the startup barrier and command
protocol under log pressure, lost readiness, lost completion and startup timeout.

`orchestration.launch.admission.NodeAdmission` reserves an explicit assigned node's
device slots and aggregate host demand atomically before launch. Separate file
owners retain disk reservations through cleanup or storage handoff; worker exit
alone does not release stored artifacts' quota.
Impossible configurations fail immediately; admissible but occupied resources
fail by default or wait only when requested. Its lease sets one assigned visibility
per child, prevents duplicate slot launch and retains the whole claim until all
owned children have disconnected and completed cleanup. Failed termination keeps
resources unavailable for reuse. Actual allocation discovery, node/cgroup limits
and complete decoded-artifact/process/training demand still require integration;
the ledger cannot establish those facts from worker pool budgets alone.

`server.records.requests.RequestRecords` now persists owner/session/run identity,
request metadata, acceptance, dispatch and terminal outcomes through that journal.
Canonical request digests include binary payload checksums. An identical ID/content
retry returns its original status; conflicting content or sequence reuse fails.
Acceptance records retain their original journal position. Completed/skipped/failed
outcomes are immutable, and a running request cannot be dispatched again.
After all workers and required terminal transfers finish, `executed` commits an
immutable `ExecutionReceipt` before requested output publication. It records the
operation, outcome, completed updates and accumulation. Terminal results and
publication failures must match those facts. A receipt does not promise published
outputs or live workers; it prevents a known update from becoming an unknown
effect merely because publication or the controller subsequently fails.

`training.session.acceptance.AcceptanceOrder` commits the contiguous validated
prefix using that record interface, without importing server storage. Later upload
completion waits for missing earlier metadata or validation. A bounded validation
window preserves space for earlier requests. Dependencies must name earlier calls;
failed dependencies reject delayed uploads too. Malformed inputs conservatively
fail the already queued suffix, preserving successful earlier contributions for
explicit corrected work under new IDs. Rejection waits for preceding execution to
settle so its outcome reports the actual surviving counters and accumulation.
The coordinator calls `advance` after completion and serializes it with writer
takeover. Accepted dependencies still require a check at dispatch, since they can
fail during execution. Durable record failure invalidates live state and requires
reconciliation. None of these primitives dispatches workers or releases input
owners; service integration supplies those lifecycle boundaries.

`training.session.dispatch.RequestDispatcher` connects this ordering to the
existing `OperationState`. It rechecks dependencies and operation invariants at
the accepted queue head, commits dispatch before calling workers, and commits an
execution receipt only after the worker adapter confirms every participant and
required terminal transfer. Validation may overlap the execution wait; short
record/state transitions share the ordering lock. Publication consumes owned host
outputs and closes them on every path. Required publication failure records a
known completed effect; optional artifact issues remain on successful results.
Worker failure records unknown effects and invalidates live state. Authoritative
storage failure stops further work. This is the internal dispatch boundary;
persistent worker transport and the complete upload lifecycle remain integration
work. The adapter must honor prepared participation and accumulation decisions,
including output-only masked execution and empty optimizer skips.
Before dispatch, the same state transition logic projects completion without
committing it. The receipt and known result envelope must fit the portable record
bound, including growing contribution history and submitted controls. An oversized
history rejects the new contribution while preserving earlier gradients, which
can still be optimized. The prepared publisher must additionally admit its own
metric and artifact descriptor bounds.

Writable attachment calls `claim_writer` with a fresh handle identity and the
current server incarnation. The durable `WriterLease` carries its epoch and next
submission sequence. New metadata and acceptance of pending uploads require the
current lease; already accepted requests retain their original identity and may
finish after takeover. Identical request retries remain read-only lookups. A lost
claim response can retrieve the same current lease; a retired handle ID cannot
silently take ownership back. The coordinator rejects fenced uploads and releases
their staging using the actual surviving training state. Claiming a writer does
not restore workers or make a run lost to server restart executable.

`training.session.writers.WriterOwnership` coordinates that durable claim with
input cleanup. A new claim retires only missing old sequence positions, preserving
live tickets and accepted work. It rejects the old writer's incomplete uploads
and gates new metadata until their ordered failures settle. This prevents those
failures from capturing a new writer's corrected requests. Attachment status is
idempotent and may become ready while an already rejected producer still owns
storage; the normal staging budget continues to apply until that producer exits.
If no incomplete uploads remain, accepted work may continue after attachment is
ready. A retried current claim does not reject that writer's own pending inputs.

The record layer rebuilds a disposable SQLite query index under the configured
node-local cache root on every open. Its journal and synchronization modes are
disabled, its page cache is bounded, and it uses no WAL. Authoritative commits
precede index updates; an index failure makes the open store unavailable until
reconciliation. The recovered journal may already contain the original completion.
Unfinished records remain inspectable without automatic replay. The engine's
`training.session.recovery.fail_interrupted_requests` invalidates the existing
operation state and commits restart failures in bounded pages. A durable running
record without an execution receipt has unknown execution effects; accepted work and
incomplete uploads are recorded as not started. This requires dispatch to be
journaled before sending any worker command. Completed outcomes stay unchanged.
An execution receipt with unfinished publication instead yields a failure with
known completed effects and its saved counters. Live state is still invalidated;
those saved facts are not restoration of weights, gradients or optimizer state.
Even a fully completed journal cannot establish that its former mutable workers
still exist. Service startup must invoke this classification; it is not a worker
restart or checkpoint restoration mechanism.

One process holds an exclusive advisory writer lock for a journal directory.
Threads share a serialized writer. A commit contains a bounded group of immutable
records with contiguous positions, canonical JSON and a SHA-256 checksum linked
to the previous commit. Tensor data belongs in separate artifacts.

Publication writes and fsyncs a temporary file, links it exclusively into the
numbered committed sequence, removes the temporary name, then fsyncs the directory.
Only then does `append` return. No existing commit can be overwritten. Any I/O
failure poisons the writer: close and reconcile durable records before continuing.
A failure after publication may leave a visible commit even though its caller
did not receive an acknowledgement.

Recovery checks each complete commit's checksum, schema, sequence and hash chain.
It removes only unpublished temporary files, and syncs the recovered prefix before
exposing it. Missing or corrupt committed files are errors; they are preserved for
diagnosis. Recovery never silently truncates acknowledged history. Readers retain
at most one commit group while scanning a prefix snapshot.

The default bounds are 1 MiB per record, 8 MiB per commit and 128 records per group.
These bound serialized reads and groups, not total retained history or Python object
overhead. A derived query index can be rebuilt from the journal. Correctness must
not depend on a SQLite WAL shared over a network filesystem.

This backend requires POSIX locking, exclusive hard-link publication and file and
directory sync semantics from the configured storage filesystem. The parent
directory is supplied by service storage; no records or caches default to a user's
home directory. Tests cover failure at each publication boundary, a killed writer,
concurrent producers, independent-process exclusion and framework-free recovery.

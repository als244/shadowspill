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
updates, discard, close and downloads after worker exit. Public server constructors
and SDK training handles compose these same components. Range-resumable transfers
and actual multi-node deployment remain separate implementation work.

`server.Server` separates the allocation lifetime from per-run creation and live
request submission. Construction performs no probes or worker startup. Starting
validates installed definitions, discovers the assigned local node, reserves
deployment envelopes, recovers storage, and starts the HTTP service. Description
has a separate bounded CPU reservation so an active training run does not occupy
its slot. `LocalCreationFactory` retains all build/profile/plan/initialization and
worker startup; the public wrapper adds no second execution path. Shutdown joins
HTTP handlers and creation/run owners before storage and allocation retirement.
Failed cleanup preserves downstream ownership for retry. Public record imports
load neither the service runtime nor the SDK training connection.

Creation now uses `POST /v1/runs/{session}/{run}` with the portable
`TrainingCreation` record, including exact resolved settings, initialization/Plan
references, resource-wait intent, name and labels. The handler publishes choices
only after registry admission and delegates startup to its configured factory.
Identical retries return the original run, including terminal failures or closes;
conflicting reuse cannot start another job. GET on the same URL returns scoped
creation progress or the saved terminal record. HTTP 202 continues until the
creator thread returns and initialized workers are available. No polling request
can start or replay work. Run listings use bounded owner/session pages with an
explicit continuation cursor. Registration must fit 8 KiB before any creation
artifact is published. Full lifecycle records have 32 KiB of space for the Plan
reference, history ID and worst-case encoded failure diagnostic. HTTP pages return
at most 32 complete rows to fit the shared 1 MiB control envelope.

`POST /v1/training/describe` accepts portable setup keywords or an existing
resolved configuration. Its configured callback uses the same CPU setup resolver
as creation, including installed definitions and generic parallelism. Defaults
resolve on the server; omitted initialization stays omitted. The deployment
reserves a CPU resolution envelope, while the HTTP handler retains its usual
metadata scratch. Description starts no workers, reserves no GPUs and publishes
no run or artifact. The returned `TrainingPreview` includes the immutable config
and per-operation geometry/schedule counts from that resolver. Capture expansion
and reuse remain separate from these counts. A successful preview does not promise
available resources.

The public `shadowspill_client.ServiceClient` composes this protocol and the owned
SSH forward. Its bounded connection pool returns query futures, and explicit
artifact handles stream verified downloads into new local files. Closing a
connection stops its polling/transfers and waits for their I/O before releasing
the forward; it never sends an ordered training close. Metadata futures retain
no task input closure after completion or failure. Preparation stays in the
server creation component, separate from the live training handle.

The SDK's `creation` module submits exact settings, polls confirmed creation
progress, and explicitly attaches a writer. Its `training` module owns snapshots,
bounded pending calls and ordered metadata/upload delivery; it imports no setup,
planner or worker launcher. All execution state remains in the existing server
controller. Runtime futures retain stable identities and original acceptance;
required outputs are explicit handles. Earlier pending IDs remain dependencies
even through client capacity waits. An uncertain exchange blocks new submissions
until explicit reattachment. Connection shutdown releases queued snapshots and
polling without issuing a training close. Reattachment and stopped-run reads load
the saved immutable configuration rather than running preparation again.

Portable raw setup serializes operations as ordered entries because canonical
JSON sorts object keys. This preserves direct resolver operation/default order.
An authenticated stale writer now receives the specific `writer_fenced` code;
credential or owner failures retain the generic access-denied response.

Creation publishes bounded immutable Plan-summary pages after the bundle's exact
recheck and before worker startup. The pages describe the selected candidate/rank
Plans using existing simulation summaries, preparation settings, representative
inputs and coverage certificates. Page IDs bind the complete bundle reference;
the existing owner-scoped artifact endpoint serves them as control metadata.
Reading does not import the planner, decode the full archive, require live workers
or claim a writer. Each page respects the shared byte/JSON bounds and a 32-row
limit; full bundle downloads remain a separate explicit transfer.

Checkpoint integration reuses `checkpoint.write_checkpoint`, the existing rank
archive encoder separated from atomic file publication. It writes bounded tensor
chunks to a borrowed blocking sink, handles partial writes, and returns the exact
written manifest. It does not flush, close or commit the destination. File saves
use this same encoder and retain temporary-file/fsync/rename behavior. The stream
owner must admit storage and confirm successful rank completion before publishing;
an incomplete or failed stream is discarded. Cross-layout conversion remains
unfinished. Archive metadata inspection can
import without loading the framework or worker-side tensor I/O.

`checkpoint.checkpoint_size` derives exact archive bytes from the same tensor
descriptors, weight selection, manifest encoding and TAR framing, before reading
tensor data. Metadata and state must stay unchanged between sizing and writing.
The output owner still enforces that exact size. This admits the complete rank
archive before streaming, including metadata, checksums and padding, without a
dry-run tensor export or a full-state snapshot.

`ArtifactStore.produce_group` reserves a complete generation's bytes and file
count before invoking any encoder. It streams files serially into persistent
storage using one bounded copy buffer, then commits all references and an optional
run-local alias in the same journal entry group. Reads expose either the complete
generation or none of it. Updating `latest` preserves earlier immutable files.
Failed production removes the entire unpublished group; uncertain publication
disables the writer until recovery establishes the committed result. This storage
primitive accepts up to 128 files per group and does not validate checkpoint state
or dispatch workers; those remain the checkpoint coordinator's responsibility.

`server.checkpoints.publish_checkpoint` validates a complete admitted rank set and
the existing writer's completion manifests, then publishes the rank archives and
portable `CheckpointManifest` through that group. Provenance includes the completed
update, history/request IDs, fixed configuration and Plans, weight selection,
frozen-base identity, and explicit caller metadata. Configuration and Plans must
already exist in the authenticated owner's storage. Global metadata bytes are
admitted exactly before starting any rank. This entry point accepts trusted
installed writers; uploaded files need separate validation. Its socket tests cover
multiple rank archives, a missing completion, mismatched identities and preserved
aliases. Logical cross-layout conversion and RNG restoration remain unfinished.

`stage_checkpoint` uses `ArtifactStore.stage_group` to return completed, private
files as a `StagedArtifacts` owner. The dispatcher can persist its execution receipt
before the publisher calls `commit`; its ordinary cleanup closes this owner on
every path. An uncommitted close discards the files. Commit and alias visibility
remain atomic, and an uncertain commit requires recovery. Storage shutdown refuses
to retire a live staged owner, while failed cleanup remains retryable. This adds
no execution path: the existing dispatcher still owns ordering and state effects.

The existing private worker connection can receive a bounded binary stream during
an already released invocation. `WorkerStream` frames borrowed encoder bytes in
chunks of at most 64 KiB, hashes while sending, and leaves tensor encoding to the
rank checkpoint writer. Binary and control frame prefixes are distinct. Receiving
the declared bytes still requires the normal ticketed completion reply; the
persistent artifact owner verifies its digest before publication. A lost stream
or reply invalidates the channel and cannot authorize replay. Real CPU children
exercise transfers directly into admitted artifact storage, process loss at both
boundaries, and consecutive invocations.

`WorkerGroup.staged` exposes the existing all-ready boundary to storage admission.
Ordinary `run` uses that same path. An admission failure aborts every ready worker
and confirms that none executed; release broadcasts to all ranks before any reply
is consumed. After release, missing completions or a false claim of unstarted
execution stop the group. Two real CPU workers exercise this path through one
staged persistent artifact group, including insufficient storage and loss of the
second completion. This is internal invocation ownership, separate from client
training settings and the server's setup component.

Ordered `save` requests now pass through the same PreparedRequests, dispatcher,
WorkerRequests and initialized WorkerExecution as training. Save settings contain
only a friendly name, JSON metadata and authoritative weight selection. The worker
reports exact archive bytes and state/Program identities while staged, and checks
both ready and completion envelopes before release. The server admits all files
before broadcasting execute, then receives native rank archives through the private
stream. Completed private files remain owned until the dispatcher records effects
and ResultPublisher commits the generation; later updates wait for that boundary.
Save results have a dedicated checkpoint reference, separate from tensor outputs.
Open accumulation is rejected before contacting workers. Admission failure preserves
state; missing completions stop the group without replay. Assembly failure after
all completions is a publication failure with known state. CPU subprocess tests
cover these boundaries and restore the emitted archives. The service currently
includes frozen tensors; declared base omission,
conversion and RNG integration still require implementation. Worker tensor copies
reuse admitted staging; checkpoint descriptor/manifest peaks are included in the
deployment's decoded-artifact allowance and still need scale qualification.

Native rank restore now inspects fixed TAR headers before opening the archive
parser. It rejects extension/link/sparse records, duplicate members, truncated
payload ranges and invalid end padding without reading tensor data. Metadata is
bounded to 16 MiB, 65,538 members, 64 JSON nesting levels and 1,048,576 JSON nodes;
duplicate JSON keys and nonfinite numbers are rejected. The writer observes the
same manifest limits. Complete tensor inventory, geometry and checksum descriptors
are checked before initialization can mutate state. The internal
`checkpoint.load.validate_checkpoint` exposes this preflight to worker staging.
It does not verify tensor contents: immutable delivery must verify the archive
digest, and the native loader still checks each tensor while reading bounded
chunks. A failure after initialization begins invalidates execution state.

Ordered restore uses the same queue and initialized workers. Prepared input owns
the bounded global metadata reservation; its name resolver runs exactly once at
ordered acceptance. Acceptance records the immutable checkpoint reference while
the original request and deduplication digest retain the submitted name. The
resolver verifies manifest bytes, complete rank membership, byte order and owner
access to configuration, Plans and rank archives. Existing FileDelivery and
FileGrants checksum and retain each rank archive through worker completion.
Every worker validates its complete native inventory and generation metadata
before any execute command is broadcast. Loading delegates to the native loader,
reserving space for its destination and optional cast chunk. Compatible native
ownership is required; cross-layout conversion remains pending.
Preflight rejection preserves all workers. A partial load or missing completion
invalidates the group; it cannot be replayed on that incarnation. Confirmed restore
commits its saved update count and new history before result publication.

The lightweight client's `transport.ssh.SSHTunnel` now owns a loopback OpenSSH
forward into these same routes. It delegates alias/key/agent/jump configuration
and host-key verification to OpenSSH. Existing masters receive only forward and
cancel commands through their pinned control socket; a private new master stays
in the foreground under an owned process group. Cleanup cannot terminate a
borrowed master or cancel accepted training. Configuration output, diagnostics,
transient commands and proxy-child cleanup are bounded by one client reservation.
Failed cleanup retains its owner for retry. SSH startup is distinct from the
authenticated HTTP readiness exchange. Real local daemon tests cover both
ownership modes, host-key rejection and disconnect/reconnect through the existing
controller; a CPU-only Della fixture verifies actual cross-host authenticated
control and binary transfer without touching an allocation. OpenSSH's
[control commands](https://man.openbsd.org/ssh#O) and
[connection-sharing configuration](https://man.openbsd.org/ssh_config#ControlMaster)
define the forwarding boundary.

## Request and worker ownership

History identity belongs to the existing `OperationState`, alongside completed
updates and accumulation. Live creation supplies the initial history; a confirmed
restore starts a distinct history even when its update count decreases. Admission
can project that history without changing state. Execution receipts persist it
before publication, and successful results and known-completion failures must
match it. Restart classification retains the receipt's history without claiming
the old workers survived. Checkpoint saves read history from the queue-head
boundary; result publishers cannot supply a different one.

`server.jobs.RunJobs` bounds creation jobs and live runs in one service-owned
table. It reserves a slot and the deployment's per-job host envelope before
durable registration, resource-free factory construction or thread launch. Exact
creation retries return the saved identity even at capacity or after retirement;
historical runs consume no live slot. Disconnects leave jobs and workers running.
There is no executor queue or idle eviction.

The optional metadata publisher runs after admission and before registration,
outside the registry lock. Concurrent attempts at that unpublished identity get
backpressure; capacity rejection never invokes publication. Publication failure
leaves no registered job, and shutdown retains an in-flight publisher until its
I/O returns. The portable `TrainingCreation` record separates resolved settings
from saved Plans, checkpoint and explicit resource-wait choices. `CreationRecord`
stores it by content digest alongside the resolved configuration; a factory
checks both scoped artifacts and their content before using them. This record
does not itself implement checkpoint loading or the public creation endpoint.

One maintenance thread joins completed creation threads, retires orderly closed
or failed controllers, and retries failed cleanup while retaining admission. Its
HTTP lookup context pins the controller through the exchange. Allocation stop
can quiesce children while a borrow remains, but cannot delete files, clear the
live owner or release its host envelope until the borrow returns. Shutdown also
retains an unfinished factory/creation thread. Record lookup remains available
after physical retirement. Public server/allocation constructors remain separate
integration work.

The node-backed `HostArena` can claim a job's combined controller capacity once,
then partition it into independently closed staging, transfer and publication
budgets. Empty partitions still count at their full capacity. This avoids partial
host-memory claims while jobs wait, and closing input staging cannot disable the
result publisher. The entire node claim survives until every partition and its
borrowed buffers retire. A private cancellation event can interrupt the initial
whole-arena wait without affecting another run's resources.

Temporary phase processes and persistent workers can use a same-node partition
for their host demand. The parent has already charged that capacity; the child
retains a slice through confirmed process exit and reports its full demand without
charging it twice. GPU admission remains exclusive. These are internal accounting
reservations, not an OS memory limit or an additional user-facing training API.

`server.local_creation.LocalCreationFactory` connects persisted creation choices
to the existing preparation driver and `LiveRun`. It validates installed setup and
initialization before GPU use, selects compatible assigned devices, reserves the
combined phase memory, and archives completed preparation artifacts and capped
logs. Fresh execution admission includes the workers and their controller buffers
in one host claim. Saved Plans skip preparation and require the exact configuration
and current installed identities. Failed cleanup retains the job and its resources
until retirement succeeds; stopped creation is never replayed.

Deployment supplies explicit process, decoded-family and temporary-disk bounds.
File-count and size checks run before publication and persistent startup; those
checks are not hard filesystem quotas or proofs of arbitrary definition peaks.
The local adapter does not discover remote capacity. Checkpoint initialization,
startup Nsight and optional idle expiry remain later implementation stages and
currently fail before phase launch when explicitly requested.

`server.live.LiveRun` composes the existing node startup barrier, worker request
adapter, controller and result publisher. A registered creation receives ready
status only after all workers initialize. Its caller reserves the decoded family
and node-backed arenas before construction and retains them through retirement.
The live owner pins request and artifact stores across this entire lifetime.

Orderly close joins the controller before releasing publishers and worker files.
Allocation stop uses the same confirmed-effect shutdown path, and can interrupt
its own resource wait without closing another job's admission. Finalized worker
logs and a session/run/rank manifest enter immutable artifact storage before node
files are removed. Cleanup failures retain the owners, storage pins and parent
reservations for retry. The parent service must also finish HTTP borrows before
dropping this owner. This composition does not yet implement public server
constructors or remote node agents.

Result artifact identities include session, run, request and output names inside
the authenticated owner's namespace. Reusing run/request IDs in another session
cannot replace or accidentally reuse an earlier session's tensor output.

`server.storage.ServiceStorage` composes the run catalog with bounded request and
artifact owners. Deployment supplies persistent/cache roots and a host envelope
for each open request journal/index, reserved before recovery. Names below those
roots are fixed or hashes of validated owner/session/run IDs. Another service
cannot acquire the catalog lock until all retained owners finish cleanup.

Startup calls the existing request recovery engine on each older unfinished run
before marking it stopped. Incomplete/accepted uploads never started; running
work without a receipt is uncertain; saved execution receipts keep their known
effect and update count. Completed results stay immutable. Creation without a
request journal stops without inventing one; missing or empty initialized-run
journals fail recovery. Read-only HTTP contexts reopen the same saved records
and artifact files. This storage layer neither creates workers nor replays work.
Transient resource failures can retry recovery; journal/index durability failures
require closing and reopening the storage namespace for reconciliation.

Allocation shutdown uses `RunController.stop` with the node's confirmed-child-exit
callback. It gates further dispatch, stops owned children, and joins the existing
dispatcher before releasing result files or invalidating its pending transition.
An execution receipt or completed result that won the race remains authoritative.
Only then does the same recovery engine classify unfinished work and retire input
owners. A failed stop, join timeout or blocked input producer retains ownership
for retry. This internal allocation-loss path may end unresolved accumulation
without an implicit update/discard; it does not relax ordered training close or
offer accepted-request cancellation.

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

The public TrainingClient now submits checkpoint save settings through its existing
runtime queue. Its typed future maps the original completion to a CheckpointHandle
without another task or request, preserving identity and acceptance observation.
Manifest inspection verifies bounded metadata; explicit directory download streams
rank files serially and writes the local manifest last. Download failure leaves
the server outcome unchanged. HTTP/CPU subprocess tests cover save/close ordering,
immutable metadata, post-worker downloads and a failed transfer followed by an
explicit retry. Setup remains in the ServiceClient creation/server factory modules.

Checkpoint preflight can also use `checkpoint.state.PlanState` to inspect a saved
Plan family without opening an Executor or allocating tensor values. It uses
the same persistent tensor layouts, owned regions, shared-state owners and host
control descriptors as execution. `validate_checkpoint` checks these descriptions
against the complete rank archive before initialization; payload checksums still
belong to immutable delivery and the streaming loader.

`PlanState.logical_tensors()` binds checkpointable `StateId` entries to those
same archive descriptors. Tied views retain separate geometry over one canonical
archive; gradients and call-local controls are excluded from logical resume
state. Saved parameter placement gives each view its global coordinates, while
the existing owned-region resolver supplies its stored range, excluding padding.
Logical intersections stream bounded descriptors even for strided views; they
allocate no tensor data. Factored optimizer state keeps its own shape. Undefined
encoded or sharded optimizer mappings fail explicitly, as does registered state
without an export binding. These descriptions do not yet implement conversion,
publication of a converted checkpoint.

The offline `checkpoint.reading.read_rank` context reuses the same parser,
geometry validation and payload checksums before exposing bounded random reads
of saved representations. Its owner keeps sources immutable and publishes only
after successful context exit. `checkpoint.transfer.transfer_tensor` verifies
complete source/target logical coverage, checks every overlapping source replica
bit for bit, and streams into unpublished target callbacks under an explicit
scratch limit. An ordinary floating-point weight cast requires an explicit
transformation; optimizer state is copied without reinterpretation. This layer
does not choose identities, prove target ties or commit a checkpoint generation.

New-run checkpoint initialization uses this CPU preflight before the existing
all-worker startup release, followed by the same bounded checkpoint reader used
for ordered restore. `LocalCreation` resolves the complete generation and checks
optimizer semantics; `PreparedWorker` selects either fresh weights or checkpoint
initialization. Old initializer provenance inside saved Plans does not require
granting or opening those weight artifacts. `LiveRun` starts `OperationState` at
the saved update and publishes parent lineage only after every worker is ready.
`RunInfo.parent_checkpoint` and `initial_completed_updates` are durable creation
provenance and remain fixed across later operations, restore segments and close.
Changing ownership or an optimizer's state interpretation still requires a
supported conversion; setup may change defaults without reinterpreting state.

Ordered in-place restore and new-run creation use the same optimizer compatibility
check: definition/source, state configuration, groups and precision must match.
Per-call defaults and objective provenance do not reinterpret tensor state. A
mismatch fails before acceptance or startup; the prepared restore reservation
covers both the global manifest and the bounded source configuration graph.

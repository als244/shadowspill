# Durable service records

`shadowspill.server.records.Journal` is the service's filesystem persistence
primitive. It owns ordered durable metadata, independently of training state,
worker dispatch and HTTP. Contiguous acceptance, dependency handling and restart
classification belong to the request coordinator; the journal does not make an
interrupted GPU update replayable.

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
The coordinator supplies a trusted prepared pipeline bound. Metadata pressure is
checked before persisting a new ID; fenced or conflicting submissions consume no
sequence slot. A decoded batch must match the request's Datum count, and the
installed preprocessing callback must finish all fallible content work before
returning. Identical retries inspect the original status without copying again.
Execution borrows accepted prepared inputs once and explicitly releases them after
its consumers finish. Failed queued inputs close promptly; rejected active readers
retain storage until they return. Waiting for an arena does not hold the ordering
lock needed by earlier cleanup. Fatal shutdown can release idle owners even when
record storage is unavailable, while active owners retain their leases. Transport
read deadlines, prepared model-specific preprocessing/spooling and writer takeover
coordination remain service integration work.

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

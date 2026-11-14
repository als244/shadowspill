# Durable service records

`shadowspill.server.records.Journal` is the service's filesystem persistence
primitive. It owns ordered durable metadata, independently of training state,
worker dispatch and HTTP. Request ordering, deduplication, writer leases and
restart outcomes still need the request coordinator; the journal does not make
an interrupted GPU update replayable.

`server.records.requests.RequestRecords` now persists owner/session/run identity,
request metadata, acceptance, dispatch and terminal outcomes through that journal.
Canonical request digests include binary payload checksums. An identical ID/content
retry returns its original status; conflicting content or sequence reuse fails.
Acceptance records retain their original journal position. Completed/skipped/failed
outcomes are immutable, and a running request cannot be dispatched again. The
training coordinator still must enforce contiguous acceptance, dependencies and
writer fencing before invoking these methods.

The record layer rebuilds a disposable SQLite query index under the configured
node-local cache root on every open. Its journal and synchronization modes are
disabled, its page cache is bounded, and it uses no WAL. Authoritative commits
precede index updates; an index failure makes the open store unavailable until
reconciliation. The recovered journal may already contain the original completion.
Unfinished records remain inspectable without automatic replay. Restart failure
classification belongs to the coordinator; persistent metadata cannot restore lost
mutable worker state.

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

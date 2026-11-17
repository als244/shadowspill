# Durable service records

The service separates persistent records from live training state. Records tell
the client whether a request was accepted, executed and published. They do not
make an interrupted GPU update replayable or replace a model checkpoint.

For deployment and client examples, see the
[advanced service guide](../python/advanced/service-client.md). The execution
components are described in [training composition](training.md).

## Authenticated transport

`server.http.HTTPService` exposes the existing run controller and artifact store
through authenticated HTTP. Bearer credentials resolve to an owner; every
session, run, request and artifact lookup checks that owner. Reachable listeners
require TLS. Loopback HTTP can use the client's OpenSSH forward, which retains
ordinary SSH key and host verification.

Control metadata and binary tensor payloads have separate bounded codecs.
Handlers reserve their scratch and concurrency slots before reading payloads.
Exchange deadlines bound incomplete network I/O without imposing a timeout on
accepted training. Artifact downloads stream immutable bytes with content hashes;
the client verifies the completed transfer.

Closing a client connection stops its polling and transfers. Accepted work
continues on the server. `training.close()` is a separate ordered operation that
waits for workers and their resources to retire.

## Creation and worker ownership

`Server` owns the allocation, installed definitions, storage and listener.
Construction does not start workers. Starting the service discovers its assigned
resources and recovers records. The allocation adapter accepts a local machine
or a single-node Slurm step; it does not request a scheduler allocation.

`LocalCreationFactory` resolves setup, runs preparation or admits saved Plans,
and creates a `LiveRun`. `describe_training()` resolves settings without starting
workers. `prepare_training()` publishes Plans without starting a training run.
An identical creation retry returns the original identity; conflicting reuse
cannot launch another job.

Phase processes and persistent workers use the installed orchestration launchers.
The service accounts for host capacity, device slots, staged files and logs
through their owners. Cleanup releases a reservation only after its consumers
and owned processes finish. A failed cleanup retains ownership for retry.
These controls are deployment infrastructure, separate from the client's model,
batch and memory settings.

Workers receive a verified Plan family, a logical-rank mapping and grants for
specific files. Private commands identify saved computations and opaque input
artifacts; they cannot name arbitrary local paths or execute supplied Python.
All workers complete startup validation before any begin initialization. A run
becomes ready only after every worker reports initialized state.

The client connects to one endpoint. Worker placement and communication belong
to the service deployment; ordinary training commands contain no node addresses.

## Request and worker ownership

`TrainingClient` owns input snapshots and submits commands in order.
`RequestInputs` reserves staging in that order, decodes inputs and finishes
validation and packing before durable acceptance. Later uploads can finish first,
but cannot consume the capacity an earlier request needs and then block it.

`RequestRecords` binds each request ID to its exact metadata, payload checksums
and sequence. Retrying identical content reads the original outcome. Reusing an
ID or sequence with different content fails. Dependencies name earlier requests;
a failed dependency prevents dependent work from running.

`RequestDispatcher` uses `OperationState` to check accumulation and update
boundaries at the head of the queue. It persists dispatch before contacting
workers. `WorkerGroup` then implements a two-stage invocation:

1. Every worker validates its command and inputs and returns a ready ticket.
2. The controller releases every participant before waiting for completion.

A staging rejection aborts all ready participants without execution. Missing
responses or failure after release invalidate the group; private worker commands
are never automatically replayed. The service stops the owned worker processes.

After all workers and required terminal transfers finish, an `ExecutionReceipt`
records the confirmed effect, update count, accumulation and history. Publishing
metrics and artifacts happens afterward. A publication failure preserves those
known effects. Optional report failures remain attached to the successful result.

Only one writer lease may submit new work to a run. Writable reattachment fences
the previous writer and rejects its incomplete uploads; already accepted work
retains its original identity. Read-only attachment can inspect records and
artifacts without taking that lease.

## Checkpoints and artifact publication

The artifact store reserves a generation's complete bytes and file count before
publication. Private staged files become visible together with their manifest;
failed production publishes none of them. Immutable earlier generations survive
updates to friendly names such as `latest`.

Checkpoint saves use the existing bounded rank encoder. Every worker reports its
state identity and exact archive size before the service releases encoding. The
server receives rank archives, records completion and atomically publishes the
complete checkpoint generation. Saves include registered RNG state and retain
verified frozen-base dependencies when unchanged weights are omitted.

Restore resolves a friendly name once to an immutable checkpoint and validates
all ranks before release. In-place restore requires compatible native ownership.
New-run initialization can first convert compatible logical state into the
target Plan family using bounded CPU staging. Conversion preserves optimizer
meaning, tied-state relationships and RNG provenance; incompatible mappings fail
before worker initialization.

The service keeps predicted reports, runtime traces, logs and checkpoints after
workers close. Downloading these artifacts does not repeat preparation or
training. A failed client transfer leaves the committed server artifact intact.

## Journal and recovery

`server.records.Journal` owns ordered durable metadata. One process holds its
exclusive writer lock. Each commit contains bounded canonical records, contiguous
positions and a checksum linked to the previous commit. Publication writes and
syncs a temporary file, links it exclusively into the committed sequence, then
syncs the directory before acknowledging success. Tensor data lives in separate
artifacts.

A failed write makes the open writer unavailable until reconciliation. Recovery
checks the complete committed sequence and removes only unpublished temporary
files. Missing or corrupt committed files are errors; acknowledged history is
never silently truncated. A disposable SQLite query index is rebuilt from the
journal under the node-local cache root. Correctness does not depend on a shared
SQLite WAL.

On service restart, unfinished requests are classified from durable evidence:

| Last durable state | Recovery meaning |
| --- | --- |
| Incomplete upload or accepted, undispatched request | Execution did not start |
| Dispatched request without an execution receipt | Execution effect is unknown |
| Execution receipt without a published result | Confirmed effect; publication did not finish |
| Terminal result | Original immutable outcome |

Recovery marks old runs stopped and preserves their results. It neither starts
workers nor replays updates. Resume uses an explicit checkpoint in a new run.
The storage filesystem must provide POSIX locking, exclusive hard-link
publication and file/directory synchronization semantics.

# ShadowSpill client

Lightweight service connections, training submissions, portable configuration and
explicit artifact downloads. Public `ServiceClient` owns description and creation;
`TrainingClient` submits runtime operations against ready server workers. See the
[complete example](../../docs/examples/client-training.md) and
[client guide](../../docs/python/client.md) for current APIs and remaining work.

```python
from shadowspill_client import BatchConfig, Datum, Model, Objective, Weights

model = Model("my_project/classifier@1", initialize=Weights.random(seed=123))
objective = Objective("my_project/classification@1")
batch = BatchConfig(limits={"max_examples": 64})
example = Datum(inputs={"pixels": pixels}, loss_inputs={"class_id": 7})
```

Configuration records own immutable copies of their JSON-compatible settings.
Datum records hold caller data; they do not themselves copy arrays. The binary
snapshot implementation in `protocol.payload` copies arrays and nested trees
under `BufferBudget` byte/request limits before submission returns. Server
acceptance and completion remain separate observations.
The separate `shadowspill_client.text.text_datum` helper constructs shifted,
aligned next-token inputs without a tokenizer or model import.

Install `./packages/client` for the service client. The package has no dependency on
the engine, model catalog or a GPU framework. Its optional `planning` extra adds
the existing engine; use a qualified CPU-only engine wheel for offline planning.

Portable input/control/output schemas live under `shadowspill_client.protocol`.
The engine uses these same records for its public
[definition hooks](../../docs/python/definitions.md).

Preparation settings expose fixed `BuildOptions`, explicit `BuildCandidates`,
`ProfilingOptions` and `PlanningOptions` under `Preparation`. `Policy` identifies
an installed server policy and its portable configuration. Omitted choices stay
distinct from explicit `None` (unchunked); candidate axes and complete coupled
`BuildCase` lists are mutually exclusive. These records preserve user choices;
server preset resolution, coverage and memory admission run on the server.

`service.prepare_training(**setup).result()` runs measurement without selecting
Plans, initializing weights or starting training workers. Its
`.export().result().download(path).result()` writes a standard ZIP with every
requested case, original phase archives and a manifest. Initialization is
optional for this path. `service.get_preparation(run_id).result()` reattaches to
the same measurement job without replaying it. The future completes after phase
resources retire; downloaded artifacts remain available after disconnect.
`Preparation(reuse=prepared.reuse, ...)` identifies those original measurements
for compatible new preparation or training. Build/profile reuse preserves exact
source and physical input identities; changed budgets/counts are replanned.

The protocol modules also define compute requests, acceptance, results and
failures. `protocol.codec` encodes a versioned, bounded canonical JSON envelope.
Tensor bytes use separate artifact references; Datum is deliberately not a wire
record. Failures say whether execution never started, completed, or had unknown
effects. Results include effective controls and immutable artifact references.
HTTP and OpenSSH connections use these same records and ordered request futures.

`SSH` holds local connection settings, including an OpenSSH alias, service host
and service port. `transport.ssh.SSHTunnel` creates a loopback forward for the same
authenticated HTTP protocol. It reuses a configured existing master or owns a
private foreground connection. Closing a borrowed forward sends only a forwarding
cancellation; it never terminates that master, an allocation or a training run.
Host-key verification, keys/agent and jump routing remain OpenSSH responsibilities.
SSH startup and HTTP service readiness are separate checks. Local diagnostics and
command ownership use a fixed client memory reservation; failed cleanup retains it
for explicit retry. The transport has real local OpenSSH and Chicago-to-Della
checks, including accepted execution continuing after tunnel closure. The public
SDK also runs from a NumPy-only Tubingen client against Chicago GPU workers.
Actual Apple Silicon qualification remains unfinished.

`protocol.payload.snapshot_batch` and `protocol.payload_io.read_payload` share
versioned binary framing, explicit dtype/shape metadata and SHA-256 integrity.
Arrays never expand into JSON. Snapshots survive caller mutation and release
their staging lease after transfer. Noncontiguous arrays, numeric lists/scalars,
empty arrays and optional installed CPU Torch tensors are supported; BF16 uses
bit-exact uint16 storage without adding Torch to the client dependencies. Snapshot
completion establishes memory ownership; it does not establish server acceptance.
Installed server processors and engine schemas own media preparation and complete
input validation.

`training.save_checkpoint(...).result()` returns a small checkpoint handle after
server persistence completes. `handle.manifest()` inspects its metadata and
`handle.download(path)` copies rank files into a new local directory, including
after worker exit. The client performs no build, planning, model initialization
or worker launch. `training.load_checkpoint(id_or_name)` restores through the same
ordered stream, returns saved caller metadata and starts a new history segment.
`service.create_training_client(plans=training.plans, checkpoint=handle)` starts
a new run from the saved update, with new history and parent-checkpoint lineage.
Setup and bounded archive validation remain separate from runtime submissions;
the checkpoint replaces fresh-weight initialization. Compatible changed layouts
use the existing bounded conversion path. Explicit uploads and resumable downloads
are implemented. Actual cross-GPU-count resume and broader deployment qualification
remain unfinished; see [current coverage](../../docs/python/client.md#current-coverage).

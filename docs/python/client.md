# Lightweight service client

`shadowspill-client` requires Python 3.12 or newer and NumPy. Model definitions,
preparation and GPU execution belong on the server. The client package does not
require the engine, PyTorch, Triton or the supplied workloads.

`ServiceClient` supports authenticated connections, CPU setup previews, creation,
reattachment and explicit artifact downloads. `TrainingClient` submits live
operations. `shadowspill.server.Server` owns startup and the assigned resources.
Checkpoint save, ordered restore, new-run resume and explicit directory downloads
are supported. New-run creation converts compatible checkpoint state when its
prepared Plans change. Named Plans can be saved and retrieved after worker close
or service restart. Explicit file and checkpoint downloads can retain verified
progress for resume; uploads are still being implemented.
Existing phase and Trainer APIs remain available during migration.

Start the service on Chicago or another machine with an existing GPU allocation:

```python
from shadowspill.server import Allocation, Server

# SHADOWSPILL_TOKEN supplies the same credential used by the client.
server = Server(
    allocation=Allocation.from_environment(),
    storage_root="/path/to/service-data",
    workload_packages=["shadowspill-workloads", "my-workloads"],
)
server.serve(host="127.0.0.1", port=8000)
```

List only installed definition packages. Construction snapshots deployment
settings; `serve` discovers assigned resources, recovers stored records and starts
the listener. The current allocation adapter supports a local machine or an
assigned single-node Slurm step. Multi-node agent bootstrap remains unfinished
and is rejected explicitly. The server never requests a scheduler allocation.

For embedding, `server.start(...)` returns after opening a background listener;
`server.address` reports its bound address. `server.close()` stops creation and
live runs, waits for owned workers and preserves stored records/artifacts. Failed
cleanup retains its owners for another `close()` attempt. Optional `ServerLimits`
configures deployment memory, concurrency and storage envelopes. These limits
belong to the deployment, independently of client training settings.

```python
from shadowspill_client import SSH, ServiceClient

# Read the deployment credential from SHADOWSPILL_TOKEN, or pass token explicitly.
with ServiceClient(ssh=SSH(host="training-host", server_port=8000)) as service:
    preview = service.describe_training(**setup).result()
    print(preview.summary())
    config = preview.config

    training = service.create_training_client(config=config).result()
    pending = training.step(global_batch, optim={"lr": 0.001})
    request_id = pending.request_id
    pending.accepted().result()  # Complete server acceptance, beyond the input copy.
    result = pending.result()
    print(result.completed_updates, result.loss)
    training.close().result()
```

`setup` contains `Model`, `Objective`, optional `Optimizer`, `BatchConfig`, `Memory`
and optional execution/preparation/operation settings. Definitions must already
be installed on the server. Omitted initialization remains omitted; a preview
does not choose fresh weights. `config=` rechecks an exact earlier configuration
and cannot be mixed with setup keywords. The preview reports physical geometry
cases and eligible/rejected schedules; capture expansion and reuse can change
the actual build work. It contains no fabricated benchmark or task count.

Use `ServiceClient(base_url="https://service.example", ...)` for a direct
connection. Loopback HTTP supports an existing or SDK-managed SSH forward.
`session_id` identifies the caller's record session and defaults to a new opaque
ID. SSH uses ordinary OpenSSH aliases, keys, host verification and jump settings.
Neither connection path starts a server or requests a GPU allocation.

Futures support `.result(timeout=...)`, `.exception(timeout=...)` and `.done()`.
A wait timeout does not cancel work or retry a mutation. `cancel()` returns false.
`close()` or leaving the context ends connection work and releases the owned SSH
forward; it does not stop accepted server work or close a training run.

`Execution(idle_timeout_seconds=...)` optionally releases idle workers. The default
is `None`, with no idle expiry. The timer runs only while the request queue and
input staging are empty, after worker initialization and after work finishes.
Inspection, duplicate queries and writer reattachment do not extend that timer.
Expiry records `idle_timeout` on the stopped run and preserves saved results and
artifacts. It does not save a checkpoint or apply pending gradients; resume from
a saved checkpoint in a new run.

Creation accepts raw setup keywords directly, or `config=preview.config`.
`plans=training.plans` with the same configuration skips preparation; weights
still come from the explicit initializer. To resume, supply
`service.create_training_client(plans=training.plans, checkpoint=checkpoint)`.
The client reads the saved configuration as bounded metadata; it does not download
or decode executable Plans. The server validates every rank before initialization
and loads the checkpoint directly, without opening the old fresh-weight source.
The new run starts at the saved update count with a new history and a durable
parent-checkpoint reference. Setup defaults remain the saved configuration's
defaults; per-call overrides in the checkpoint do not replace them.
The creation future exposes its
`session_id` and `run_id` before readiness. Reconnect using the same session ID
and `service.get_training_client(run_id).result()`. Writable reattachment fences
the earlier writer; `read_only=True` permits inspection without takeover.

Runtime methods are `step`, `forward_backward`, `optim_step`, `forward`,
`discard_gradients`, `save_checkpoint`, `load_checkpoint` and ordered `close`, all returning futures. Input arrays and
controls are copied before the submission returns. Bounded staging/pending counts
apply backpressure, and snapshots that cannot fit fail promptly. Earlier pending
requests become explicit dependencies, so a rejected backward contribution cannot
be silently omitted from an already submitted optimizer update. A transport error
blocks new submissions on that handle: inspect the saved IDs and explicitly
reattach. `training.requests.status(id).result()` queries lifecycle state;
`.get(id).result()` retrieves the original outcome without resubmission.

`training.get_config()` returns the cached immutable configuration. Tensor results
are handles in `result.outputs`; their `.download(path).result()` is explicit.
The durable outcome remains in `result.record`. A `TrainingRequestError` preserves
the server's failure record, including known execution effects and accumulation.

`training.plans.summary().result()` reads saved metadata, including the selected
operation/batch candidates and rank Plans, chunk/policy choices, representative
inputs, coverage, memory and simulated timings. Each page contains at most 32
rows and fits the same bounded metadata format. Continue explicitly with
`.summary(page=result.next_page).result()` until `next_page` is `None`. Rows have
a `kind` and immutable `values`. These reads do not load the full bundle, run the
planner or issue a training operation; they remain available after worker close
and server restart. `.plans.download(path).result()` explicitly exports the full
bundle. Simulated durations are estimates for the recorded representative inputs,
separate from observed training latency.

`saved = training.plans.save(name="baseline").result()` gives the stored bundle
a run-local name and returns an immutable Plan handle. It copies no weights or
bundle data and does not enter the training queue. Plan names are independent
of checkpoint names. Saving the same name again is idempotent.
After reconnecting with the same `session_id`, retrieve it with
`service.get_plans(run_id, name="baseline").result()`. Both operations work after
worker close and service restart, without acquiring a writer or GPU. Pass the
handle as `plans=saved` when creating another compatible training run; supply
explicit initialization or a checkpoint as usual. Downloads remain explicit.

An immutable artifact reference can be attached to the current connection with
`service.artifact(reference)`. Its `.download(local_path).result()` streams and
verifies bytes into a new local file. Existing files are preserved; a failed
transfer removes its partial destination. Tensor data is never implicitly loaded
by a metadata result.

For large files, use `artifact.download(local_path, resume=True).result()` from
the first attempt. An interrupted transfer retains bounded progress metadata and
partial bytes in `<local_path>.shadowspill-download`. After reconnecting, attach
the same reference with `service.artifact(reference)` and explicitly repeat the
download with `resume=True`. The client verifies the retained prefix before
requesting the remaining bytes and checks the complete file checksum before
publishing the destination. No exchange retries automatically. Changed references,
corrupt prefixes and concurrent writers are rejected; existing destination files
are preserved. The client removes its partial directory after success.

A checkpoint save occupies the same ordered stream as training. Finish an open
accumulation with `optim_step` or `discard_gradients` first. Later updates wait for
the complete checkpoint to commit; a completed save does not increment the update
counter. Reusing `name` updates a run-local alias while preserving older generations.

```python
pending = training.save_checkpoint(name="step-50", metadata={"data_cursor": 50})
checkpoint = pending.result()
print(checkpoint.id)
manifest = checkpoint.manifest().result()  # Small verified metadata only.
restored = training.load_checkpoint(checkpoint.id).result()
print(restored.metadata["data_cursor"])
training.close().result()
checkpoint.download("./step-50").result()  # New local directory, after worker exit.
```

`weights="master"` is the default; `weights="compute"` selects compute-precision
values under the existing checkpoint reconstruction rules. Caller metadata is
frozen before submission and returned in the manifest. The save future preserves
its request identity and `.accepted()` observation. Request lookup exposes the
same handle as `training.requests.get(request_id).result().checkpoint`.

`load_checkpoint` accepts an immutable ID or run-local name. Ordered acceptance
pins the name to an immutable checkpoint, exposed by the future's
`accepted().result().checkpoint`; moving the name later cannot retarget the load.
The server checks the complete rank set and dependencies, verifies delivered files,
and validates every worker's native state layout before releasing any restore.
An open accumulation must be optimized or discarded first. A confirmed load
restores the saved update count and starts a new history while request sequences
continue increasing. `result.metadata` contains the saved caller metadata.
Validation failure preserves state; failure after worker release invalidates it.
Publication failure reports the confirmed restore count and history without replay.

Directory download verifies each immutable rank file, streams serially and writes
`manifest.json` after the rank files. It does not overwrite existing destinations.
A failed download leaves the committed server checkpoint available and does not
repeat training or saving. Plans and frozen-base dependencies are separate
artifacts; downloading the checkpoint does not implicitly copy them. The current
service save includes frozen tensors. In-place restore requires compatible native
state ownership. New-run resume can convert compatible logical state into a new
Plan family before starting workers; preparation reports a `converting` stage.
The run keeps the selected source in `parent_checkpoint` and records the actual
startup artifact in `initial_checkpoint`. Converted manifests record their source
checkpoint, source Plans, logical state identity and conversion version. Conversion
preserves optimizer state and weight selection, and rejects incompatible state.
Worker checkpoints also retain Python, NumPy and Torch default generator state.
Fresh runs initialize those streams from `Execution.seed`; restores use the saved
seed and state. Evaluation and fully masked read-only calls advance separate
evaluation streams, leaving training randomness unchanged. Compatible conversion
preserves each matching logical stream. New streams use versioned seed derivation
from the saved seed, restore identity and logical stream, recorded in
`manifest.info.conversion.randomness`. Physical GPU identity does not affect seeds.
Custom model-owned generators still require explicit registered state support.

`checkpoint.download(path, resume=True)` retains an incomplete directory with its
checkpoint identity and completed/partial rank files. Repeat the call explicitly
after reconnecting. Completed ranks are verified locally and need no second
download; partial ranks use the same verified suffix transfer. `manifest.json`
appears only after every rank is complete. A directory with unrelated content or
a completed manifest is preserved and rejected as a new destination. Default
`resume=False` keeps the existing behavior of removing this attempt's incomplete
copy. File locking currently targets the Linux and macOS client platforms; Linux
qualification is recorded separately from the outstanding macOS release gate.

Pass `runtime_trace=True` to a compute call to record that invocation. Its
`result.traces` contains downloadable ZIP handles, one per participating worker.
Each archive contains a request/Plan/rank/update association in `manifest.json`,
the existing `step.json` diagnostic view, and its `step.events.npz` event columns.
The artifacts remain downloadable after the training client closes.

Optional trace assembly or publication failures appear in
`result.artifact_issues`; a confirmed update still succeeds and must not be
repeated. Required execution or cleanup failures retain their ordinary failure
semantics. A skipped optimizer with no Plan produces a trace issue explaining
that no invocation ran. Tracing is opt-in and its overhead belongs to that call.

Configure Nsight at creation with
`Execution(nsys={"start_step": 2, "steps": 2, "gpu_metrics": False})`.
The one-based window counts complete optimizer updates. For split training it
starts at the first contributing `forward_backward` and ends after the selected
terminal `optim_step`; evaluation between contributions stays inside the window.
Masked calls do not advance the count. Capture does not restart after a restore
or interrupted window. Device metrics default to enabled and require support and
permission on the assigned GPU; `gpu_metrics_frequency` selects their rate.
Omitting `nsys` launches ordinary workers without a collector.

`capture = training.nsys(rank=0).result()` reads bounded per-rank metadata.
`capture.record.status` is `pending`, `ready`, `failed` or `missing`; its `window`
records selected/completed updates, history and first/last request IDs. Native
reports finalize after worker exit, so a completed selected update can still have
pending report status. After `training.close().result()`, query until finalization
finishes. A ready report requires both complete worker evidence and successful
collector exit. If `capture.report` is present,
`capture.report.download("rank-0.nsys-rep").result()` explicitly downloads verified
native bytes. Failed captures may retain a partial report, identified by their
status and error. Capture publication failure does not repeat completed training.

After reconnecting with the same session, use
`service.get_nsys(run_id, rank=0).result()`. Final manifests and reports survive
worker close and service restart. A terminal run with only a pending record is
reported as missing finalization. These reads need no writer or live worker.
The deployment installs Nsight and provides collector host, disk and finalization
allowances through `ServerLimits.creation.nsys`; client settings contain no
server paths or executable names. Captured timings include profiler overhead.

Service base omission and resumable uploads remain under
implementation. Qualification evidence and remaining hardware coverage are
tracked separately from this API contract.

# Lightweight service client

`shadowspill-client` requires Python 3.12 or newer and NumPy. Model definitions,
preparation and GPU execution belong on the server. The client package does not
require the engine, PyTorch, Triton or the supplied workloads.

`ServiceClient` supports authenticated connections, CPU setup previews, creation,
reattachment and explicit artifact downloads. `TrainingClient` submits live
operations. `shadowspill.server.Server` owns startup and the assigned resources.
Checkpoint save, ordered restore, new-run resume and explicit directory downloads
are supported. New-run creation converts compatible checkpoint state when its
prepared Plans change. Resumable transfers and named Plan saves are still being
implemented.
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

An immutable artifact reference can be attached to the current connection with
`service.artifact(reference)`. Its `.download(local_path).result()` streams and
verifies bytes into a new local file. Existing files are preserved; a failed
transfer removes its partial destination. Tensor data is never implicitly loaded
by a metadata result.

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

Service base omission and resumable uploads or downloads remain under
implementation. Qualification evidence and remaining hardware coverage are
tracked separately from this API contract.

# Lightweight service client

`shadowspill-client` requires Python 3.12 or newer and NumPy. Model definitions,
preparation and GPU execution belong on the server. The client package does not
require the engine, PyTorch, Triton or the supplied workloads.

`ServiceClient` supports authenticated connections, CPU setup previews, creation,
reattachment and explicit artifact downloads. `TrainingClient` submits live
operations. `shadowspill.server.Server` owns startup and the assigned resources.
Checkpoint methods and richer Plan inspection are still being implemented.
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
still come from the explicit initializer. The creation future exposes its
`session_id` and `run_id` before readiness. Reconnect using the same session ID
and `service.get_training_client(run_id).result()`. Writable reattachment fences
the earlier writer; `read_only=True` permits inspection without takeover.

Runtime methods are `step`, `forward_backward`, `optim_step`, `forward`,
`discard_gradients` and ordered `close`, all returning futures. Input arrays and
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

An immutable artifact reference can be attached to the current connection with
`service.artifact(reference)`. Its `.download(local_path).result()` streams and
verifies bytes into a new local file. Existing files are preserved; a failed
transfer removes its partial destination. Tensor data is never implicitly loaded
by a metadata result.

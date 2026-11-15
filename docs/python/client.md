# Lightweight service client

`shadowspill-client` requires Python 3.12 or newer and NumPy. Model definitions,
preparation and GPU execution belong on the server. The client package does not
require the engine, PyTorch, Triton or the supplied workloads.

`ServiceClient` currently supports authenticated connections, CPU setup previews
and explicit artifact downloads. Training handles and public server/allocation
constructors are still being implemented. Existing phase and Trainer APIs remain
available during migration.

```python
from shadowspill_client import SSH, ServiceClient

# Read the deployment credential from SHADOWSPILL_TOKEN, or pass token explicitly.
with ServiceClient(ssh=SSH(host="training-host", server_port=8000)) as service:
    preview = service.describe_training(**setup).result()
    print(preview.summary())
    config = preview.config
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

An immutable artifact reference can be attached to the current connection with
`service.artifact(reference)`. Its `.download(local_path).result()` streams and
verifies bytes into a new local file. Existing files are preserved; a failed
transfer removes its partial destination. Tensor data is never implicitly loaded
by a metadata result.

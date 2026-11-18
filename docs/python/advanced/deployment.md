# Service deployment

A client connects to one endpoint. The operator supplies compute, credentials and
storage; client training scripts need only the endpoint and their training
settings. See [Compute resources](compute-resources.md) for inventory, preview
and optional hardware selection.

## Start a local service

Install the engine and required definition packages on the compute host. With
the service credential in `SHADOWSPILL_TOKEN`:

```python
from shadowspill.server import Allocation, Server

server = Server(
    service_root="/local-ssd/shadowspill/runs",
    model_root="/models",
    allocation=Allocation.from_environment(pool="lab"),
)
server.serve(host="127.0.0.1", port=8000)
```

Omit `model_root` if there is no existing checkpoint directory. Omit `allocation`
to use assigned resources from the current environment and pool `"default"`.
`Allocation.from_environment()` consumes a supplied local machine or single-node
Slurm step; it does not request capacity. Multiple assigned devices on that host
can serve one distributed run.

The laptop needs only `shadowspill-client`. Connect using the same credential:

```python
from shadowspill_client import SSH, ServiceClient

service = ServiceClient(ssh=SSH(host="service-host", server_port=8000))
print(service.list_hardware().result().summary())
```

`service-host` is the service's SSH alias. It is not a worker list. Direct HTTPS
uses `ServiceClient("https://training.example.org")` instead. The same training
scripts work through either connection.

## Storage and deployment controls

| Setting | Purpose |
| --- | --- |
| `service_root` | Mutable sessions, request outcomes, Plans, reports and checkpoints; preserve it across service restarts |
| `model_root` | Optional existing frozen checkpoints; separate from mutable run storage |
| `cache_root` | Temporary preparation/worker staging; defaults to `.cache` under `service_root`; use local SSD |
| `workload_packages` | Installed definition packages; omission enables engine definitions and the supplied workloads when installed |
| `allocation` | Already assigned environment and optional pool name |
| `authentication` | Explicit owner credentials instead of the default `SHADOWSPILL_TOKEN` owner |
| `limits` | `ServerLimits` for concurrency, host accounting, scratch, artifacts and request I/O |
| `host`, `port`, `tls` | Listener configuration passed to `serve()` or `start()` |

Datasets remain client-owned. `model_root` neither locates datasets nor stores
training outputs. Deployment limits are accounting envelopes, not OS quotas or
changes to a client's execution/spill budget. Incomplete uploads expire after
24 hours without a verified chunk by default (`ServerLimits.upload_ttl_seconds`);
saved artifacts do not expire under this policy.

For a directly reachable listener, pass a server TLS context:

```python
import ssl

tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
tls.load_cert_chain("/etc/shadowspill/server.crt", "/etc/shadowspill/server.key")
server.serve(host="0.0.0.0", port=8443, tls=tls)
```

Use a certificate trusted by clients for the endpoint's hostname. A private CA
can be supplied through the client's `ssl_context`. Each example starts the
listener once; choose the loopback/SSH or direct TLS deployment.

`server.start(...)` embeds a background listener; `server.address` reports its
address. `server.close()` stops owned work and retains persistent artifacts.
Closing a client connection does not stop accepted work. See
[failure handling](failures.md) for run recovery.

## Checkpoint lookup and definition packages

For an unpinned `Model.from_pretrained("Qwen/Qwen3.5-9B")`, the server looks for
`model_root/Qwen/Qwen3.5-9B` or `model_root/Qwen3.5-9B`. Distinct matches are an
error. A match uses the installed checkpoint reader, validates architecture and
contents, and records the absolute source and content checksums in resolved
settings. It never writes to that checkpoint directory. A missing match uses
the installed reference resolver. Explicit revisions and already resolved
configurations retain their exact source; lookup never substitutes a local
directory for an explicitly pinned remote revision.

An explicit `workload_packages` list replaces the defaults. List only installed
packages and include any [custom definitions](custom-definitions.md) used by
clients. Omitting it enables the engine and supplied workloads when installed;
it does not discover unrelated extension packages.

The public training deployment uses GPUs assigned to one host. The separate
[node-agent architecture](../../architecture/service/README.md) describes the
multi-node protocol; it is not an available public training deployment.

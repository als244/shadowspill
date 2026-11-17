# Compute resources

Normal training uses automatic hardware selection. The service chooses equivalent
assigned GPUs that fit your memory budget; the GPU count comes from your
parallelism configuration. `step()`, `forward_backward()`, `optim_step()` and
`forward()` still receive one global batch or command.

Use these optional setup controls to inspect hardware or choose a pool, GPU type
or particular devices. Start with a connected `ServiceClient` and the training
settings from the [client guide](../client.md).

## Inspect assigned hardware

```python
hardware = service.list_hardware().result()
print(hardware.summary())
```

The summary lists the service's compute pools, nodes, GPU types, device capacities
and availability. Each node also reports assigned CPU and host-memory capacity,
available host memory and its GPU driver. Device IDs are available in
`hardware.nodes[i].devices[j].id`.

Availability reflects this service's resource claims. Inspection reserves
nothing, and it does not measure bandwidth or predict training performance.

## Choose hardware during setup

`Resources` is separate from model, optimizer and preparation settings:

```python
from shadowspill_client import Resources

# Use names reported by list_hardware().
resources = Resources(pool="lab", gpu_type="NVIDIA GeForce RTX 5090")
```

Every field is optional. A pool groups hardware supplied by the service operator;
a GPU-type constraint permits the service to choose any matching devices. To
choose exact devices instead:

```python
resources = Resources(node="chicago", devices=[0])
```

Numeric indexes refer to that node's assigned inventory, not to physical PCI
slots or another process's device numbering. They require `node=`. Stable device
IDs can be supplied instead. The device list must match the logical worker count;
its order determines the mapping from logical ranks to physical devices.

Use `resources=` with `describe_training`, `plan_training`, `prepare_training` or
`create_training_client`. For example, refine an existing setup preview:

```python
preview = service.describe_training(
    config=preview.config,
    resources=resources,
).result()
print(preview.summary())
```

The preview shows proposed workers, per-worker memory budgets and preparation
candidate counts. It creates no GPU workers and reserves no hardware. Busy
devices remain visible so you can inspect the setup before deciding to wait.

Pass both parts of the preview when creating the run:

```python
training = service.create_training_client(
    config=preview.config,
    resources=preview.resources,
    progress=True,
).result()
```

`preview.config` contains portable computation settings. `preview.resources`
retains the selected pool/GPU type and any explicit device constraints. Automatic
selection can use equivalent available devices; an exact selection keeps those
devices. Creation checks availability again and reports its final worker mapping
in the progress log. It fails when resources are occupied unless you explicitly
set `wait_for_resources=True`.

## Inspect Plans before initialization

```python
plans = service.plan_training(
    config=preview.config,
    resources=preview.resources,
    progress=True,
).result()
plans.report().result().download("plan.html").result()

training = service.create_training_client(
    config=preview.config,
    resources=preview.resources,
    plans=plans,
    progress=True,
).result()
```

Planning uses temporary GPU processes for build/profiling, then releases them.
The HTML report contains predicted timelines and memory usage. Training creation
reuses those Plans and initializes the model without repeating preparation.

Resource selectors are not saved as physical placement requirements inside the
computation configuration. [Saved Plans](saved-plans.md) can start on other
compatible devices; target/source checks still apply. Resuming trained weights
and optimizer state additionally requires a checkpoint.

## Name a deployment's pool

The operator may name the assigned local machine or single-node Slurm step:

```python
from shadowspill.server import Allocation, Server

server = Server(
    service_root="/local-ssd/shadowspill",
    allocation=Allocation.from_environment(pool="lab"),
)
server.serve()
```

Omitting `allocation` uses the current environment and pool `"default"`. This
uses capacity already supplied to the service; it does not request a Slurm job.

# Offline memory and bandwidth sweeps

Export a preparation's measurements once, then explore memory budgets and
hypothetical transfer speeds on your laptop. The optional CPU planner needs no
GPU, model weights or running service.

## Measure once on the server

Start the service as in the [homepage](../../../README.md#start-the-service).
This prepares the same 32K LoRA workload as the [long-context example](long-context.md),
without loading training state or submitting updates. No dataset is needed for
preparation. Run on the client with the server's SSH alias and port:

```python
from shadowspill_client import (
    SSH, BatchConfig, LoRA, Memory, Model, Objective, Optimizer,
    Preparation, ServiceClient,
)
from shadowspill_client.text import build_candidates

service = ServiceClient(ssh=SSH("training-host", 8000))
prepared = service.prepare_training(
    model=Model.from_pretrained("Qwen/Qwen3.5-35B-A3B", lora=LoRA(rank=16)),
    objective=Objective("cross_entropy"),
    optimizer=Optimizer("adamw", precision={
        "gradient_dtype": "bfloat16", "opt_state_dtype": "bfloat16",
        "master_dtype": "bfloat16", "parameter_rounding": "stochastic",
        "opt_state_rounding": "stochastic",
    }),
    batch=BatchConfig(limits={
        "token_capacity": 32768, "max_sequence_length": 32768, "max_sequences": 8,
    }),
    memory=Memory(device_gib=12, spill_gib=128),
    prepare=Preparation(candidates=build_candidates(
        microbatch_tokens=[32768], projection_chunks=[8192], moe_chunks=[8192],
    )),
    progress=True,
).result()
prepared.export().result().download("measurements.zip").result()
service.close()
```

## Explore on your laptop

Install the CPU planner from a source checkout (Python 3.12+, CMake and a C/C++
compiler):

```bash
python -m pip install ./packages/client ./packages/engine \
    -Ccmake.define.SHADOWSPILL_CPU_ONLY=ON
```

Then run:

```python
from shadowspill import PlanningInputs

inputs = PlanningInputs.load("measurements.zip")
sweep = inputs.sweep(device_gib=[12, 16, 24], transfer_scale=[0.5, 1, 2])
sweep.write_report("sweep.html")
```

Open `sweep.html` locally. The comparison shows predicted server execution time,
GPU and spill requirements, build settings, microbatch counts and orderings.
Each feasible variant links to the interactive timeline, lanes and memory
breakdowns. Keep the adjacent `sweep_reports/` directory with the HTML file when
copying the report. Numeric rows are also saved there as `rows.json`, along with
the original settings, measurement identities and transfer rates in
`measurements.json`.

For just one variable:

```python
inputs.sweep(device_gib=[12, 16, 24]).write_report("memory.html")
inputs.sweep(transfer_scale=[0.5, 1, 2]).write_report("bandwidth.html")
```

Omitted budgets inherit the measured preparation. `spill_gib` accepts a number
or list too. Budgets are per worker. A transfer scale of `2` doubles the measured
host-to-device and device-to-host bandwidths, including their concurrent rates.
It leaves compute times, transfer latency, collective communication and memory
measurements unchanged.

Every point runs the ordinary planner again; it can choose a different schedule.
An infeasible point stays in the report with its reason. Best variants are marked
only within equal operation, batch limits and scheduled capacity. Capacity
includes padding; it is not the number of real or supervised tokens in a batch.
These are predictions, not runtime measurements.

The sweep uses the shapes, chunks and microbatch counts in the export. To explore
new build choices, prepare and export those choices first. Sweeps leave the
original measurements unchanged and do not start training. To submit compatible
local schedules, use the explicit [local planning workflow](service-client.md#preparation-and-local-planning);
the service still checks the complete prepared family's memory and input coverage.

For programmatic inspection, `sweep.rows` contains one row per point and variant:

```python
for row in sweep.rows:
    if row.best:
        print(row.device_gib, row.transfer_scale, row.predicted_seconds)
```

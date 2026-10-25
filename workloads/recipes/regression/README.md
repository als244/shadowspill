# Distributed regression example

A small non-text model trains with weights, gradients and optimizer state sharded
across a single DP replica. Inputs differ by rank. The objective divides each
local squared-error sum by the global element count; reductions sum gradients.
The model also mutates an integer buffer, exercising persistent model state.

Start once from the parent process on a node with at least two visible GPUs:

```bash
python -m workloads.recipes.distributed_training \
  --run-dir runs/regression --ranks 2 --steps 10
```

The driver runs separate processes for each phase and uses torchrun only for
communication calibration and execution. Do not wrap the driver in torchrun.
Each worker exposes its physical GPU as process-local `cuda:0`, matching the
saved code's device ordinal. The initial example uses launcher rank as logical
rank; the public Executor also accepts a complete rank mapping.

| Module | Work |
|---|---|
| `build.py` | Construct the meta model and compile a shared Program on one GPU. |
| `profile.py` | Measure local tasks on one GPU with the ordinary allocator. |
| `communicate.py` | Measure the Program's declared collectives on all ranks. |
| `plan.py` | Plan and write a simulated report on CPU, without importing Torch. |
| `execute.py` | Load the Plan into fresh Trainer workers; initialize bounded chunks, train, log and checkpoint. |
| `model.py`, `data.py`, `initialize.py` | Application computation, resumable rank-local inputs and small deterministic initialization. |

Run a phase independently with `--phase build`, `profile`, `communicate`, `plan`
or `execute`. Later phases read `config.json` and saved archives from `--run-dir`.
A new build needs a fresh directory. For example:

```bash
python -m workloads.recipes.distributed_training \
  --run-dir runs/regression --ranks 2 --phase build
python -m workloads.recipes.distributed_training --run-dir runs/regression --phase profile
python -m workloads.recipes.distributed_training --run-dir runs/regression --phase communicate
CUDA_VISIBLE_DEVICES= python -m workloads.recipes.distributed_training \
  --run-dir runs/regression --phase plan
python -m workloads.recipes.distributed_training --run-dir runs/regression --phase execute
```

Use `--phase plan --execution-gib ... --spill-gib ...` to replan the measured
Program under different memory budgets without repeating build or measurement.

The run directory contains one archive for each phase, `logs/`, `report/`, and
`training/rank-00000/` etc. Each rank writes its owned checkpoint shards and
metrics; DistributedLogger also combines records. Optional `--wandb-project`
enables the existing logger's device telemetry. Inductor/Triton caches remain
outside the run directory.

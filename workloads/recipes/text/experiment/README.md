# Configured packed-text training

The JSON text runner uses the public build → profile → plan → execution APIs.
Each phase runs in its own process. Planning reads saved artifacts without
importing Torch; execution loads Plans without constructing the model.

```bash
python -m workloads.recipes.text.run experiment.json
python -m workloads.recipes.text.run experiment.json --phase build
python -m workloads.recipes.text.run experiment.json --phase profile
python -m workloads.recipes.text.run experiment.json --phase communicate  # ranks > 1
python -m workloads.recipes.text.run experiment.json --phase plan
python -m workloads.recipes.text.run experiment.json --phase execute
```

Start build in a fresh run directory. Later phases use the same configuration
and artifacts. `run_config(path, overrides, phase=...)` provides the Python entry
point and returns the last CPU `StepResult` after execution.

## Configuration

The existing `@module:object`, `@call` and `@partial` JSON forms select Python
objects. The model, data and objective remain workload choices.

| Fields | Meaning |
| --- | --- |
| `model`, `data`, `objective`, `objective_args` | Model constructor, packed corpus and objective `(model, batch)`; batch is `[tokens, targets, seq_lens]`. `data` given as a directory alone is `PackedTokens` over it; the objective defaults to next-token prediction. |
| `checkpoint` | A family and a local Hugging Face checkpoint directory, standing for `model` built at the checkpoint's configuration and `initialize` streaming it in: `implementation`, `dtype` and `lora` go to the model, `seed` to the initializer (the run's seed otherwise), any other option such as `max_seq_len` (the run's otherwise) to both. |
| `optimizer`, `optimizer_args` | Optimizer constructor (default: the mlops AdamW) and arguments. |
| `initialize` | Callable `(request, destination)` filling Executor-provided bounded CPU chunks. It may also be a context manager, such as a checkpoint reader; `workloads.recipes.text.initialization:from_checkpoint` is the supplied one for a local Hugging Face checkpoint. |
| `max_seq_len`, `max_tokens_per_step` | Packed sequence cap and total token capacity per update. |
| `max_tokens_per_microbatch` | Fixed example capacity; alternatively use `planning_min_tokens_per_microbatch` and `planning_max_tokens_per_microbatch` for a geometry sweep. |
| `build` | Additional public `build_step` options, such as partitioning and operator recomputation. |
| `master_dtype`, `grad_dtype` | Optional explicit precision policies, resolved as Python objects. |
| `execution_gib`, `spill_gib` | Per-worker planning budgets; execution includes external headroom. Left out, the profiled device's memory less 2 GiB and three quarters of the host memory available when profiling ran (`hardware.json`); `planning.json` records the budgets planned. |
| `warmup_seconds`, `measure_seconds`, `external_headroom_mib` | Task/communication measurement controls; defaults 0.05 s, 0.18 s and 512 MiB. |
| `steps`, `schedules`, `resume` | Target total updates, runtime hyperparameter schedules and optional Trainer checkpoint. |
| `parameter_norms_every`, `metric_reducer`, `omit_parameter_metrics` | Optional runtime norms (off by default) and CPU metric formatting. No compiled parameter-observation tasks. |
| `eval_every`, `eval_microbatches` | Held-out evaluation cadence and packed examples per evaluation. |
| `checkpoint_every`, `checkpoint_dir`, `keep_last` | Full training-state checkpoint cadence and retention. |
| `run_dir`, `seed`, `settings`, `plots` | Output directory, build seed/settings and report plotting. |
| `shares` | One positive integer per rank: how the step's microbatches divide among the ranks (equal by default), for ranks of unequal speed; every rank's loss stays its share of the step's mean. |
| `hosts`, `port` | Several machines in one run, the first the one the runner is launched on: `{"address", "ranks", "ssh", "repo", "python", "env"}` each. Every host builds, profiles and plans for its own GPUs under `targets/<gpu>/`, since executables and profiles belong to the hardware that made them; the ranks agree on what they exchange, checked when the workers open. Communicate and execute launch every host's workers under one rendezvous at the first host's address (`port`, default 29400); `env` carries the hosts' network settings (such as `NCCL_SOCKET_IFNAME`). Needs a fixed `max_tokens_per_microbatch`. |
| `ranks` | Workers, one per GPU of one node (default 1). The build shards weights, gradients and optimizer state over them (`Parallelism.dp`, unless `build.parallelism` says otherwise), a communicate phase measures each Program's collectives on the workers, and every rank trains its share of each step's `max_tokens_per_step` tokens with the step's trained-token count as the shared normalizer. Evaluation runs distinct held-out batches per rank. |

A config needs `run_dir`, `data`, `max_seq_len`, `max_tokens_per_step` and
`steps`, and a `checkpoint` or a `model`; a missing one is named. A key the
runner does not know is refused, with the nearest known name. `training.run`
takes the same config as a path or a mapping from Python.

Schedule keys use the saved Plan's hyperparameter names. A hyperparameter a
schedule names is made a tensor by the runner, since a Python number is fixed
into the compiled step at capture; one no schedule names stays the constant
it is given as. For example:

```json
{
  "optimizer": "@torch.optim:AdamW",
  "optimizer_args": {
    "lr": 0.001,
    "foreach": false
  },
  "schedules": {
    "optimizer_group.0.lr": {
      "@call": "shadowspill.training.schedules:WarmupCosine",
      "lr": 0.001,
      "min_lr": 0.0001,
      "warmup_steps": 10,
      "total_steps": 100
    }
  }
}
```

From the command line, a schedule's field is reached through its quoted name,
`'schedules."optimizer_group.0.lr".total_steps=300'`; unquoted, the dots in
the hyperparameter name would read as nesting.

Packed token normalization is a scalar tensor input, so the number of valid
targets can change between updates without rebuilding. The resumable source
preserves document packing, pending documents and the trained-token count.

## Outputs and resume

`programs/`, `profiles/` and `profiled/` hold `.ssp` artifacts by geometry;
`hardware.json` records the profiled device's and host's memory.
`search/` retains feasible plans for each ordering; `search.json` records failed
cases too. `plans/training.ssp` and optional `plans/evaluation.ssp` are selected
for execution, with shared model tensors admitted by Executor; the pair shares
the worker's pool and its fixed overhead, so an evaluation plan costs only
what its own tensors need beyond the training plan's. `report/` holds
the simulated timeline; `logs/` contains one log per phase.

Execution writes `metrics.jsonl`, `packing.jsonl`, optional metric tables and
`checkpoints/step_XXXXXXXX.ssp`; with `ranks`, each rank writes those under
`rank-xxxxx/` (its records hold its share of each step's figures) and the
distributed logger writes `aggregate/metrics.jsonl`, the ranks' shares summed,
with one W&B run per rank and one aggregate; `resume=step_XXXXXXXX` names the
step and every rank loads its own save; with `wandb_project`, `wandb_run.json` records
the W&B run's id and URL, which the runner prints after execution. Norms/RMS/module shares use the same CPU reducer
and logging hooks as the reference loop. `result.json` contains only the final
CPU observations, not model tensors. Compiler caches remain external.

Optional `startup_diagnostics=true` writes `startup/` with measured and simulated
timelines before the first update. It requires an optimizer with the explicit
zero-LR preservation guarantee (MLOps AdamW) and CPU tensor learning rates.
Buffers, RNG and training progress are preserved; ordinary PyTorch AdamW is
rejected. See [startup diagnostics](../../../../docs/python/api/training.md#startup-diagnostics).

```bash
python -m workloads.recipes.text.run experiment.json \
  resume=runs/text/checkpoints/step_00000003.ssp steps=6 --phase execute
```

A checkpoint binds to the training state, not to the Plans: the tensors by
name and geometry must match, and a rebuilt directory, another microbatch
geometry or another budget resumes it. Resume restores
optimizer state, schedules, RNG and packing position. Partial/compute-only
checkpoint policies from the old runner are rejected explicitly here rather
than silently changing their meaning.

With `ranks` the runner launches that many workers on one node under
`torch.distributed.run`; the [Qwen3 distributed recipe](../qwen3/) is the
qualified sharded-DP reference the phases follow.
`engine="pytorch"` selects the ordinary [PyTorch reference loop](../reference/).
It uses `device` (default `"auto"`) and `compile` (default `true`), without
constructing ShadowSpill Plans or installing the execution allocator. It preserves
packing, schedules, optional observations, evaluation and local PyTorch checkpoints.
Select one fixed microbatch size; planning/search belongs to the phase runner.

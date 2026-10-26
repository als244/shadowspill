# Configured packed-text training

The JSON text runner uses the public build → profile → plan → execution APIs.
Each phase runs in its own process. Planning reads saved artifacts without
importing Torch; execution loads Plans without constructing the model.

```bash
python -m workloads.recipes.text.run experiment.json
python -m workloads.recipes.text.run experiment.json --phase build
python -m workloads.recipes.text.run experiment.json --phase profile
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
| `model`, `data`, `objective`, `objective_args` | Model constructor, packed corpus and objective `(model, batch)`; batch is `[tokens, targets, seq_lens]`. |
| `optimizer`, `optimizer_args` | Optimizer constructor and arguments. |
| `initialize` | Callable `(request, destination)` filling Executor-provided bounded CPU chunks. It may also be a context manager, such as a checkpoint reader. |
| `max_seq_len`, `max_tokens_per_step` | Packed sequence cap and total token capacity per update. |
| `max_tokens_per_microbatch` | Fixed example capacity; alternatively use `planning_min_tokens_per_microbatch` and `planning_max_tokens_per_microbatch` for a geometry sweep. |
| `build` | Additional public `build_step` options, such as partitioning and operator recomputation. |
| `master_dtype`, `grad_dtype` | Optional explicit precision policies, resolved as Python objects. |
| `execution_gib`, `spill_gib` | Per-worker planning budgets; execution includes external headroom. |
| `warmup_seconds`, `measure_seconds`, `external_headroom_mib` | Task/communication measurement controls; defaults 0.05 s, 0.18 s and 512 MiB. |
| `steps`, `schedules`, `resume` | Target total updates, runtime hyperparameter schedules and optional Trainer checkpoint. |
| `parameter_norms_every`, `metric_reducer`, `omit_parameter_metrics` | Optional runtime norms (off by default) and CPU metric formatting. No compiled parameter-observation tasks. |
| `eval_every`, `eval_microbatches` | Held-out evaluation cadence and packed examples per evaluation. |
| `checkpoint_every`, `checkpoint_dir`, `keep_last` | Full training-state checkpoint cadence and retention. |
| `run_dir`, `seed`, `settings`, `plots` | Output directory, build seed/settings and report plotting. |

Schedule keys use the saved Plan's hyperparameter names. Values that vary at
runtime must be tensor arguments during build; a Python numeric constant is
fixed by capture. For example:

```json
{
  "optimizer": "@torch.optim:AdamW",
  "optimizer_args": {
    "lr": {"@call": "torch:tensor", "data": 0.001},
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

Packed token normalization is a scalar tensor input, so the number of valid
targets can change between updates without rebuilding. The resumable source
preserves document packing, pending documents and the trained-token count.

## Outputs and resume

`programs/`, `profiles/` and `profiled/` hold `.ssp` artifacts by geometry.
`search/` retains feasible plans for each ordering; `search.json` records failed
cases too. `plans/training.ssp` and optional `plans/evaluation.ssp` are selected
for execution, with shared model tensors admitted by Executor. `report/` holds
the simulated timeline; `logs/` contains one log per phase.

Execution writes `metrics.jsonl`, `packing.jsonl`, optional metric tables and
`checkpoints/step_XXXXXXXX.ssp`. Norms/RMS/module shares use the same CPU reducer
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

The saved Plan and evaluation Plan must match the checkpoint. Resume restores
optimizer state, schedules, RNG and packing position. Partial/compute-only
checkpoint policies from the old runner are rejected explicitly here rather
than silently changing their meaning.

This initial configured recipe launches one worker. Use the separate
[Qwen3 distributed recipe](../qwen3/) for qualified sharded DP execution.
`engine="pytorch"` selects the ordinary [PyTorch reference loop](../reference/).
It uses `device` (default `"auto"`) and `compile` (default `true`), without
constructing ShadowSpill Plans or installing the execution allocator. It preserves
packing, schedules, optional observations, evaluation and local PyTorch checkpoints.
Select one fixed microbatch size; planning/search belongs to the phase runner.

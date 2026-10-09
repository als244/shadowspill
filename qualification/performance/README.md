# Full-model performance qualification

Each worker runs one large model entirely through ShadowSpill. It does not
construct a standard-allocator model, which would exceed the target physical
device budget.

The worker initializes, registers, and calibrates the runtime pools before it
constructs the model. It prints the exact solo and bidirectional-concurrent
route bandwidths before planning and persists that same runtime snapshot in
the result artifact. Model state is then constructed, imported into the spill
pool, and released from anonymous CPU storage.

The retained geometries all use sequence length 1,024 and 65,536 tokens per
optimizer step:

| Family | Tokens/microbatch | Accumulation |
|---|---:|---:|
| Llama 3 8B | 8,192 | 8 |
| Qwen 3.5 9B | 16,384 | 4 |
| OLMoE 7B | 32,768 | 2 |

Run one cell:

```bash
python -m qualification.performance.run llama3 mlops \
  qualification/results/full_model/mlops_llama3.json \
  --artifact-store qualification/results/full_model/artifact_store/mlops_llama3
```

`--artifact-store` roots both trees of the store, and `--build-store` and
`--plan-store` override either. `--build-store-mode` and `--plan-store-mode`
say what this cell does with each; the four modes are defined in [the artifact
store guide](../../docs/python/artifact-store.md#store-modes), and `contribute`
is the default. Omitting the store entirely puts it in the user
cache root, shared with every other run on the machine.

A host cannot hold both the full pinned spill arena and an anonymous
checkpoint copy. `--skip-checkpoint` runs the throughput protocol without that
copy, and the resulting artifact records that checkpoint qualification was
skipped. A single-cell `run` invocation still checkpoints and restores by
default; checkpoint/replay release coverage lives in the numerical matrix.

`--plan-only` plans and writes the cell's plan record without running a step,
which is how placement-bearing records are produced for replay. It is also a
matrix option, and covers every selected cell there.

`--spill-budget-gib` changes the configured runtime spill-pool capacity.
`--planning-spill-budget-gib` may set a smaller budget for planning without
shrinking that physical pool. The planning budget is rejected immediately if
it exceeds the configured capacity.

Run the matrix with:

```bash
python -m qualification.performance.matrix \
  --output-directory qualification/results/full_model \
  --keep-going \
  --planning-spill-budget-gib mlops_qwen35=100
```

That runs the three cells carrying a throughput floor, of the five defined.
The remaining two pure-PyTorch cells are available through `--cells`. Every
cell still checks runtime, memory and simulator behavior; a throughput floor
is used only when its recorded hardware and configuration match.

The matrix runs every cell as a checkpoint-free throughput probe: it forwards
`--skip-checkpoint` so the anonymous full-state copy never coexists with the
pinned spill arena. `--checkpoint` opts a matrix run back into the
checkpoint/restore protocol. The repeatable `--planning-spill-budget-gib
IDENTITY=GIB` option forwards a per-cell planning budget; the retained Qwen
setup plans against 100 GiB inside its 112-GiB pool.

With checkpointing, the protocol checkpoints the planned callable, performs
and diagnoses one warm step, restores the checkpoint, then measures three
groups of four steps. Without it, the warm step is kept rather than restored
and the same three groups follow. Planning, compilation, warmup, and restore
are outside timed execution.

## What a group measures

The gate submits four steps back to back, then waits for the runtime to go idle.
Each step records a device-clock cycle from its origin to the next step's
origin. An explicit end marker closes the last cycle after terminal work
finishes. Consecutive cycles include the gaps between steps, opening fetches,
and required terminal writeback.

The group line

```text
mlops_llama3 group 1: 18.710652s/step, 3502.60 tokens/s
```

reports the **average** of its four cycle durations, and total group tokens
divided by their sum. The saved `group_seconds` uses that sum;
`host_group_seconds` independently records the host wall-clock span.
Group-boundary reporting and collection happen after those measurements.

The cell's headline `median_step_seconds` is the **median of all twelve
individual cycle durations** across the three groups. Throughput is tokens per
step divided by that median. Equal group averages can therefore conceal
different individual timings; inspect the saved `cycle_seconds` to compare
warmup or drift.

A callable can return while terminal work is still queued. Its host call
duration, saved as `dispatch_seconds`, therefore does not measure the full
step. Before the next invocation starts, the callable drains the preceding
plan's required work; `prior_invocation_drain_seconds` records that wait.
The cycle already includes this work, so it must not be added a second time.

The traced warm step is outside these twelve samples. There is no deliberate
cooling pause or additional warmup between groups. The gate does not assert
that clocks or timings have reached a steady state; use the raw samples to
assess that separately.

## Hardware, precision and budgets

The default execution pool is 16 GiB on SM80+ and 10 GiB below SM80. The spill
pool is 112 GiB on every device. Within the execution cap, external headroom
is 512 MiB on every device; `--external-headroom-mib`
overrides that allowance. `--execution-budget-gib` overrides the former;
`--spill-budget-gib` overrides the latter (a scalar for the worker, repeatable
`IDENTITY=GIB` entries for the matrix).

SM80+ retains BF16 model weights, gradients and optimizer moments with no
masters. Below SM80 the gate uses FP16 weights and gradients, FP32 moments,
and no masters. The optimizer is `mlops.optim.AdamW`. All four choices are
independently configurable with `--model-dtype`, `--master-dtype`,
`--grad-dtype`, and `--opt-state-dtype`. Each case prints the resolved dtypes
beside its budgets and writes them to its manifest. For example:

```bash
python -m qualification.performance.matrix \
  --execution-budget-gib 10 \
  --model-dtype float16 --master-dtype none \
  --grad-dtype float16 --opt-state-dtype float32 \
  --output-directory qualification/results/full_model_fp16
```

## Throughput comparisons on another machine

The retained throughput floors were measured on an RTX 5090 with the original
16/112 GiB pools and BF16 training settings. The gate automatically checks
that scope. A different GPU, dtype, geometry or budget reports the throughput
regression check as **not applicable**, with the mismatch recorded in the
artifact. It still judges runtime correctness, physical budgets and simulator
accuracy. It also omits the predecessor ratio outside the matching scope.
The remote floor depends on a particular network path; without matching link
hardware metadata it is reported without enforcing that floor.

`--measure-only` remains an explicit way to collect measurements without
turning any diagnostic gate into a failing exit status. It is no longer needed
just because a machine has a different GPU. Artifacts retain the measurements,
gate fields and baseline applicability in both modes.

Each cell needs enough host memory for its spill pool plus runtime overhead.
A machine short of a requested capacity fails at runtime bootstrap. A measuring
run never silently adopts its own throughput as a passing baseline.

The numerical and performance launchers accept `--external-headroom-mib`
(default **512 MiB on every GPU**). This allowance is subtracted, together with
the initial process baseline, when sizing the execution pool. Zero reserves no
external allowance. It does not change model or reference identity.

The independent `--reject-overbudget` flag defaults to **off**. Without it,
external and whole-process memory overruns are measured and reported. With it,
they fail the run. `--no-reject-overbudget` explicitly restores reporting mode.
Neither setting resizes the pool or permits pool overflow; actual device OOMs
still fail. Banners and artifacts record both controls, including
`external_headroom_bytes`, `reject_overbudget`, `physical_budget_enforced`, and
`physical_budget_within_limit`.

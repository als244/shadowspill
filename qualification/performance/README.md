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

Run one cell with fresh planning artifacts:

```bash
python -m qualification.performance.run llama3 mlops \
  qualification/results/full_model/mlops_llama3.json \
  --force-fresh
```

A host cannot hold both the full pinned spill arena and an anonymous
checkpoint copy. `--skip-checkpoint` runs the throughput protocol without that
copy, and the resulting artifact records that checkpoint qualification was
skipped. A single-cell `run` invocation still checkpoints and restores by
default; checkpoint/replay release coverage lives in the numerical matrix.

`--plan-only` plans every cell and writes its PressureFit fixture without
running a step, which is how placement-bearing fixtures are produced for
replay.

`--spill-budget-gib` changes the configured runtime spill-pool capacity.
`--planning-spill-budget-gib` may set a smaller budget for PressureFit without
shrinking that physical pool. The planning budget is rejected immediately if
it exceeds the configured capacity.

Run the matrix with:

```bash
python -m qualification.performance.matrix \
  --output-directory qualification/results/full_model \
  --force-fresh \
  --keep-going \
  --planning-spill-budget-gib mlops_qwen35=100
```

That runs the three cells carrying a throughput floor, of the five defined.
A cell without a floor cannot pass or fail, so it is not in the default set;
`--cells` reaches the other two.

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

A group is timed as a whole, not as four timed steps. The clock starts before
the first step is submitted, all four are submitted back to back, the group
then waits for the runtime to go idle, and the clock stops. So the line

```text
mlops_llama3 group 1: 18.710652s/step, 3502.60 tokens/s
```

reports `elapsed / 4` and `tokens_per_step * 4 / elapsed`, where `elapsed` is
that whole wall-clock span. Neither number is a median or an average of
per-step timings, and nothing between the steps is excluded: a gap between one
step and the next is inside the measurement, because it would be inside a real
training loop too.

Timing the group rather than each step is deliberate. A step returns before
its work finishes, so a per-call duration measures dispatch, not execution.
Those per-call durations are recorded separately as `dispatch_seconds`, which
is how far ahead of the device the frontend ran, not throughput.

In practice the frontend runs very little ahead, because every invocation
after the first begins by waiting for the previous one's plan to go idle: a
plan assumes its initial objects are resident when it starts, and the
invocation before it ends by writing them back. That wait is recorded per step
as `prior_invocation_drain_seconds`, beside `dispatch_seconds`. It is measured
on every step rather than only on a traced one, because the first invocation
has nothing to wait for and a trace taken on a warm first step is the one step
that never pays it.

Waiting for idle at the end of each group makes the figure slightly
conservative. The last step's terminal writeback is charged to the group,
whereas a loop that kept going would overlap it with the next step. That is
the same assumption the simulator makes when it charges the terminal tail, so
the measured and predicted numbers describe the same regime.

The cell's headline figures come from the median across groups, not across
steps: `median_group_seconds` is the median of the three group spans, and
`median_step_seconds` is that divided by four. A stall inside one group raises
that group's span and cannot be averaged away by the steps around it.

## Measuring on another machine

The floors in `workloads.full_model` are throughput measured on one machine.
On any other machine a pass says nothing and a failure says only that the
hardware differs, so `--measure-only` runs the same protocol and reports the
measurement instead of judging it:

```bash
python -m qualification.performance.matrix \
  --output-directory qualification/results/<machine-name> \
  --force-fresh \
  --keep-going \
  --measure-only
```

Each cell prints its median step, throughput, predicted step with simulator
error, and planning time. The gate lines are gone, and so are the regression
and predecessor ratios, which divide by throughput from the floor machine and
would describe that gap rather than this run. Cells close as MEASURED or
ERROR rather than PASS or FAIL, and the matrix exits on whether the cells ran.
Every gate field is still written to the artifact and `summary.json` records
the mode, so a run stays judgeable later against floors that suit the machine
that produced it.

The cells need the device to themselves: each plans against a 16 GiB
execution budget and a 112 GiB pinned host spill arena, and the host cannot
hold that arena alongside another process's reservation. A machine that is
short of either fails at runtime bootstrap rather than measuring something
misleading. Adopting a machine's own numbers as floors means replacing the
`workloads.full_model` table, which is a deliberate edit and not something a
measuring run does.

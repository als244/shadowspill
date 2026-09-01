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

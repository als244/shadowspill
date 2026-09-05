# Qualification

`qualification/` is ShadowSpill's thin release-acceptance surface. It owns the
protocol descriptions and four launchers, but no alternate implementation of
planning, execution, diagnostics, serialization, or model state.

```text
qualification/
├── gates.py         suite, numerical, and performance in one run
├── numerical/
│   ├── README.md
│   ├── run.py       one reference/planned correctness cell
│   └── matrix.py    the five approximately-1B cells
└── performance/
    ├── README.md
    ├── run.py       one full-model throughput cell
    └── matrix.py    the retained full-model matrix
```

## Running the gates

The three gates answer different questions and are usually wanted together,
so one command runs them in order and reports what each found:

```bash
python -m qualification.gates
```

They always run unit suite, then numerical matrix, then performance matrix,
whatever order the command line names them in; each finishes before the next
begins, because the measured ones are timed and overlapping them would
corrupt both. Name a subset to run only those:

```bash
python -m qualification.gates suite numerical
```

Each gate's output streams to the terminal as it runs and is saved under
`qualification/results/gates_<run>/`. `--run NAME` names this run, which also
names the matrices' own output directories, `numerical_<run>` and
`performance_<run>`. It defaults to the commit being measured and the time,
`<revision>_<mmdd>_<hhmm>`, with `_dirty` inserted when the tree carries
uncommitted changes: a cell's saved numbers do not record which revision produced them, so
the directory name is what makes a result readable later as a reference, and a
revision alone does not identify a tree that was modified.

Both matrices write their usual artifacts into those directories -- per-cell
JSON and log, plan report, PressureFit fixture, `summary.json`, and the run's
artifact store -- so a gate run leaves exactly what a matrix run by hand
leaves, under a name that says what produced it.
A failing gate stops the ones that would follow unless `--continue-after-failure`
is given, and `--keep-going` lets a matrix finish its remaining cells after
one cell fails.

### Options, and which gate they reach

`--run`, `--keep-going`, and `--continue-after-failure` describe the run as a
whole and stay on the wrapper. Everything else belongs to one gate, and goes
in one config file with a section per gate:

```json
{
  "suite": ["-k", "not slow"],
  "numerical": [
    "--reference-dir", "qualification/results/references/h100/approximately_1b"
  ],
  "performance": []
}
```

```bash
python -m qualification.gates --config qualification/gates.json
```

Each section is that gate's own command line, forwarded verbatim and
unread by the wrapper. That is deliberate: an option the wrapper understood
would be one it had to gain whenever a matrix gained one, and the two would
drift. It also means the sections accept whatever their matrix accepts today,
including `pytest` arguments for the suite, with no change here. A missing
section means no extra arguments; an unknown section name is an error rather
than a silently ignored typo.

Keeping all three in one file is what lets a run be reproduced from a single
artifact rather than from a remembered command line.

References are specific to the machine that recorded them, so a set recorded
elsewhere belongs in a directory named for what recorded it, and the gate is
pointed at the one that matches. Record a set once, then read it:

```json
{"numerical": ["--reference-dir", "<root>/h100/approximately_1b",
               "--regenerate-reference"]}
```

then drop `--regenerate-reference` from every run after. A run that records
its own baseline minutes before comparing against it still checks that the
planned step agrees with the unplanned one, but it cannot notice that either
has changed since the baseline was blessed.

`--budget FAMILY=BYTES` in the numerical section is how to ask whether moving
data changes what is computed: the budget decides what spills, prefetches, and
recomputes, so the same cell run at several budgets against one reference
should report the same numbers.

Run gates through this wrapper rather than calling a matrix directly, so that
the order, the run naming, and the per-gate logs all hold. If it cannot say
something a matrix can, that is a reason to give it the option.

The closing summary reports each gate's verdict and wall time, then what it
found: the suite's test counts and the name of every test that failed or
errored, which correctness cells agreed with their reference and which did
not, and for the performance matrix a table of real and simulated step time
and throughput per cell with the simulator's error, planning time, and where
that planning time went by phase.

Of the suite's counts, the deselected ones are the `fresh_process` tests,
which need a process where nothing has touched the device yet and so are run
one per process by CTest rather than in the shared pytest process. Nothing is
skipped.

The launchers delegate to `src/tools/qualification/`, which in turn uses the
public `src/shadowspill/` APIs and workload definitions under `workloads/`.
Generated reference states, compact result summaries, and optional detailed
reports are written beneath `qualification/results/`, which is ignored by Git.
The numerical matrix reuses one identity-checked compiled reference under
`qualification/results/references/approximately_1b/<model>/<provider>/reference.pt`.
Its neighboring `inputs.pt` contains the exact input microbatches, while the
reference contains only the final model and optimizer state; repeated matrix
runs do not create duplicate checkpoints.

Run the numerical matrix:

```bash
python -m qualification.numerical.matrix \
  --keep-going
```

Compact correctness evidence is the default. A `--cold` run uses temporary
compiler and planning caches and removes them afterward. Use
`--detailed-artifacts` only for an investigation that needs full PlanReports,
PressureFit fixtures, and per-task runtime traces. Use
`--regenerate-reference` only when intentionally replacing the canonical
compiled references and input sidecars.

Run the full-model matrix:

```bash
python -m qualification.performance.matrix \
  --output-directory qualification/results/full_model \
  --force-fresh \
  --keep-going
```

Both matrix launchers follow the planning-evaluation logging protocol: every
cell opens with a labeled START block (model, data geometry, budgets), the
cell subprocess streams live under a `[cell/N]` prefix, and every cell closes
with a PASS/FAIL block carrying per-gate status and UTC START, STOP, and
DURATION records. The console stream is duplicated with timestamps into
`matrix.log` beside `summary.json`, and each cell keeps one timestamped log.

Before its first group each cell prints the plan's own prediction, in the
same units the measured lines use: seconds per step from the simulated
makespan, and tokens per second from the manifest's tokens per step divided
by it. It then names the fetch and evict bandwidths the plan was made
against, because a prediction is only as good as the rates behind it and a
machine whose lanes do not deliver them explains its own error, and closes
with the same two units for the unconstrained step: every graph-pair group at
its cheapest option and no waiting at all, which is the ceiling the budget is
being traded against. The measured lines that follow are the whole group's wall clock
divided by the steps in it, not a median of per-step times, so a group's
number includes everything between submitting its first step and the device
finishing its last.

The performance matrix judges throughput against floors measured on one
machine. Its `--measure-only` reports the measurement without those floors,
closing cells as MEASURED rather than PASS, which is the mode to run on a
machine the floors did not come from.

Framework-free PressureFit benchmarking belongs in
`benchmarking/planning_eval/fixture_benchmark.py`. Step inspection reads the
saved step diagnostics (`docs/python/step-diagnostics.md`), and the gap report
below summarizes them across a matrix.

`--profiler-annotations` on the performance launcher emits profiler ranges around
task boundaries and compiled calls, so an external profiler can attribute time
to the task that spent it. It is off by default because the ranges cost
something to emit and a gate run should not pay for them.

The real-versus-simulated gap report reads the traced warm steps a
performance matrix saved and prints, per model, where the step's time went
against the simulation: span, task-duration, and idle deltas; task-duration
error by phase; task start drift along the compute lane; each transfer lane's
assumed versus effective bandwidth; and every measured transfer's achieved
rate classified by its overlap with the opposite lane and bucketed by size.
It is the acceptance experiment for simulator changes:

```bash
python -m tools.qualification.gap_report qualification/results/full_model
```

## Finding where a step stops being reproducible

The numerical gate runs every case with mlops's `deterministic_kernels`
in effect, which asks each operation that offers the choice for the kernel
whose accumulation order is fixed. Without it a kernel that sums with atomics
returns a slightly different answer each run -- one llama3 step run twice
differed in 78 of its 111 gradient tensors -- and no comparison against a
reference or against a replay can mean anything. The ordered kernels cost
throughput, so they are not the default outside this gate. The request covers
reference generation as well as the planned run, so a regenerated reference is
itself reproducible.

The request is baked into a compiled graph without a guard, so a graph cached
from a run without it would be reused rather than recompiled. The matrix gives
each case its own compile cache and deletes it afterwards, which is what makes
that safe here; a tool that reuses a cache across the boundary would need to
key it on the setting.

The gate requires a checkpoint replay to agree with the uninterrupted run
within tolerance, and records whether it agreed bit for bit besides. When it
did not, the useful question is which stage of the step is not reproducible,
and the nondeterminism probe answers that rather than leaving it at
"somewhere in the backward":

```bash
python -m tools.qualification.nondeterminism llama3 --model-implementation mlops
```

It runs the same fixed input through the same model twice with nothing changed
in between and compares bitwise at three widening levels -- the objective,
every module's forward output, and every module's incoming gradient -- then
names the first divergence in execution order. It takes the same geometry
knobs as the numerical matrix (`--seed`, `--model-config`, `--data-geometry`,
`--case-factory`, `--case-option`), so a failing cell can be probed with the
same shape and data that failed. `--no-modules` drops the per-module hooks,
which cost memory on a large model, and compares only the objective and the
parameter gradients. `--deterministic` asks for the ordered kernels first, so
a divergence that survives it comes from somewhere the request does not reach.
It exits non-zero when the step is not reproducible.

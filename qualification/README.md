# Qualification

`qualification/` contains the gate orchestrator, case runners and acceptance
checks. Start with `python -m qualification.gates`; the orchestrator calls the
matrix runners, which launch each case in a fresh process. All planning and
execution use the public `shadowspill` APIs.

```text
qualification/
├── gates.py         the five gates in one run
├── numerical/
│   ├── README.md
│   ├── run.py       one reference/planned correctness cell
│   └── matrix.py    the five numerical cells (smaller presets below SM80)
├── performance/
│   ├── README.md
│   ├── run.py       one full-model throughput cell
│   └── matrix.py    the retained full-model matrix
├── remote/
│   ├── README.md
│   └── matrix.py    the numerical matrix, spilling to another machine
└── remote_perf/
    ├── README.md
    └── matrix.py    the performance matrix, spilling to another machine
```

## Running the gates

The gates answer different questions and are usually wanted together, so one
command runs them in order and reports what each found:

```bash
python -m qualification.gates
```

They always run unit suite, then numerical matrix, then performance matrix,
whatever order the command line names them in; each finishes before the next
begins, because the measured ones are timed and overlapping them would
corrupt both.

**`numerical_ssd` is opt-in** and runs the same numerical matrix against a local
SSD spill pool. See [SSD numerical qualification](#ssd-numerical-qualification).

**`remote` and `remote_perf` are not in the default run**, because each needs
a memory daemon reachable over RDMA. `remote` is the numerical matrix with the
spill pool on a peer, and `remote_perf` the performance matrix the same way,
judged against floors measured with the pool on a peer; both skip and succeed
when no peer is named. See [remote/README.md](remote/README.md) and
[remote_perf/README.md](remote_perf/README.md). Name a subset to run only
those:

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
JSON and log, plan report, plan record, `summary.json`, and the run's artifact
store -- so a gate run leaves exactly what a matrix run by hand leaves, under a
name that says what produced it.
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
python -m qualification.gates --config qualification/gates_h100.json
```

**A config file is named for the machine its references were recorded on.**
`gates_h100.json` points the numerical gate at `references/h100/`, and running
it anywhere else fails the first cell on optimizer-state structure rather than
on anything numerical -- the two machines' runs enumerate optimizer parameter
groups differently, so every ordinal mismatches while the model tensors are
exact and the replay is bitwise. That reads as a broken tree and is not one.
Passing no config at all uses each gate's own defaults, which is what a run on
the machine that recorded the default set wants.

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
planned step agrees with the reference one, which is PyTorch alone compiled
fullgraph, but it cannot notice that either
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
one per process by CTest rather than in the shared pytest process. These
*deselected* tests still run. Actual platform/capability skips are listed by
pytest separately.

Suite planning calls use fresh temporary artifact stores, including the
fresh-process CTest cases. Tests may reuse artifacts within their own store
when checking cache behavior, but never use the user's default store. A
repository check enforces explicit stores on the suite's planning calls.
Inductor and Triton caches are isolated separately by the CTest launcher.

The suite selects BF16 where supported and FP16 where BF16 compilation is unavailable.
It probes once in a child process before collection, leaving the allocator in the
test process uninitialized, and passes the choice to fresh-process CTest cases.
An explicit pytest --test-dtype choice overrides SHADOWSPILL_TEST_DTYPE; either
overrides automatic selection. CPU-only suite runs retain BF16 storage fixtures.
For example, a gate config can contain {"suite": ["--test-dtype", "float16"]}.

Correctness tests and the numerical gate use fixed-count task profiling:
exact-task warmups and allocation probes still run, followed by the configured
sample count, with no minimum conditioning or measurement duration. This keeps
small test kernels from adding seconds to every fresh-process case. The shared
policy lives in `qualification.profiling`; performance measurements keep
the production profiling policy.

CTest prints each canary's start and result while the suite runs, with a
heartbeat after 30 seconds without output. Each canary retains its CMake
timeout, and a 30-minute ceiling bounds the whole CTest invocation. A timeout
or interruption terminates the process group, including accelerator workers.
Gate logs are flushed as output arrives.

All gate code lives here, using the public `shadowspill` APIs and optional
workload definitions under `workloads/`. There is no second implementation
package or layer of forwarding scripts.

| Location | Purpose |
| --- | --- |
| `gates.py` | Run selected gates in order; stream output and summarize results. |
| `numerical/matrix.py`, `performance/matrix.py` | Choose cases and launch isolated case processes. |
| `numerical/run.py`, `performance/run.py` | Actual command-line implementation for one case. |
| `numerical/`, `performance/` helpers | Case definitions, execution phases, measurements and verdicts. |
| `device_defaults.py`, `precision.py`, `profiling.py` | Shared hardware, dtype and profiling policies. |
| `remote/`, `remote_perf/` | The same matrices using a configured remote spill pool. |
| `tests/qualification/` (repository root) | Tests for gate behavior. |

Generated reference states, compact result summaries, and optional detailed
reports are written beneath `qualification/results/`, which is ignored by Git.
The numerical matrix reuses one identity-checked compiled reference under
`<reference-dir>/<model>/<implementation>/reference.pt`, where `<reference-dir>`
defaults to `qualification/results/references/approximately_1b` on SM80+ and
`qualification/results/references/pre_sm80` on older GPUs.
Its neighboring `inputs.pt` contains the exact input microbatches, while the
reference contains only the final model and optimizer state; repeated matrix
runs do not create duplicate checkpoints.

Run the numerical matrix:

```bash
python -m qualification.numerical.matrix \
  --keep-going
```

Compact correctness evidence is the default. An `--empty-caches` run starts
every reference and planned subprocess with an empty artifact store and empty
Inductor and Triton caches, in a temporary directory removed afterwards, so a
case cannot be served anything an earlier run or another case left behind. Use
`--detailed-artifacts` only for an investigation that needs full PlanReports,
plan records, and per-task runtime traces. Use `--regenerate-reference` only when intentionally
replacing the canonical compiled references and input sidecars.

The numerical gate explicitly uses nearest rounding for AdamW moments, keeping
its existing references valid. Quickstart, Trainer, and performance workloads
select stochastic rounding for BF16 moments by default; FP16 and FP32 moments
keep nearest rounding. MLOps's standalone default remains nearest.

### SSD numerical qualification

`numerical_ssd` uses the **same five cases, model/data/dtype defaults, compiled
reference files, tolerances, checkpoint replay, transfer-pressure checks and
physical-budget checks** as `numerical`. Only the planned arm's spill pool changes
from pinned host to SSD. Its capacity remains 32 GiB. Activation eviction and
fetching remain available to the ordinary planner. This gate runs full training,
not a LoRA-only or read-only test; model initialization and updates write to SSD.

Choose an existing directory on the local SSD with at least 32 GiB free:

```bash
mkdir -p ~/shadowspill_ssd
SHADOWSPILL_SSD_DIRECTORY=~/shadowspill_ssd \
  python -m qualification.gates numerical_ssd --run ssd_check --keep-going
```

The usual `suite numerical performance` default is unchanged. The SSD gate
writes logs to `qualification/results/gates_<run>/numerical_ssd.log` and per-case
results, artifacts and a summary to `qualification/results/numerical_ssd_<run>/`.
Each result records the selected pool configuration. References use the ordinary
numerical gate's canonical directory and need no SSD-specific regeneration.

For explicit settings, put the matrix arguments in the existing gate config:

```json
{
  "numerical_ssd": [
    "--ssd-directory", "/path/on/local/ssd",
    "--ssd-staging-mib", "256",
    "--ssd-chunk-mib", "2",
    "--ssd-queue-depth", "16"
  ]
}
```

Run it with `python -m qualification.gates numerical_ssd --config <config.json>`.
All numerical matrix options, including `--models`, `--implementations`, dtype
overrides and `--detailed-artifacts`, work in this section. The directory flag
takes precedence over `SHADOWSPILL_SSD_DIRECTORY`; a missing directory is an
error. Staging is a separate host-memory cap, not an additional model copy.
SSD calibration uses initialized 16 MiB probes with one warmup and three samples
per measurement, avoiding the normal large write-calibration workload. It uses
the same Runtime calibration and fixed-rate simulator as the host gate; modeling
rate changes between solo and concurrent transfers is a separate improvement.

The matrix also runs directly, without another wrapper module:

```bash
python -m qualification.numerical.matrix --spill-pool ssd \
  --ssd-directory ~/shadowspill_ssd \
  --output-dir qualification/results/numerical_ssd_manual --keep-going
```

Each case creates a temporary direct-I/O file; normal Runtime close or process
exit removes it. Result artifacts and reference checkpoints are separate and
persist. See the [SSD API](../docs/python/api/ssd.md) for pool and staging details.

### Full-model performance qualification

Run the full-model matrix:

```bash
python -m qualification.performance.matrix \
  --output-directory qualification/results/full_model \
  --keep-going
```

Both matrices give each cell its own artifact store under the output
directory, and both take `--build-store-mode` and `--plan-store-mode` to say
what a cell does with each tree of it; the four modes are defined in
[the artifact store guide](../docs/python/artifact-store.md#store-modes).
`contribute` is the default, which is what a gate run wants: it reuses whatever
matches by digest and keeps what it had to build. The
numerical matrix asks for `reuse` on its planning tree unless
`--detailed-artifacts` is given, because there is nothing to keep from a cell
that only has to agree.

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
being traded against. The measured lines that follow are each group's
steps on the device clock: a step is the compute stream's cycle from the
origin it records before its first task to the next step's origin, or to the
marker recorded after the group's last step, so a group's number is the sum
of its steps' cycles and the cell's median step is the median cycle over
every measured step ([timing](../docs/python/api/timing.md)). The host's own
wall clock for the group is reported beside it and decides nothing.

The performance matrix judges throughput against floors measured on one
machine, and the remote performance matrix against a second table measured
there with the spill pool on a peer. `--measure-only` reports the measurement
without those floors,
closing cells as MEASURED rather than PASS, which is the mode to run on a
machine the floors did not come from.

Step inspection reads the saved step diagnostics
(`docs/python/step-diagnostics.md`), and the gap report below summarizes them
across a matrix.

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
python -m qualification.gap_report qualification/results/full_model
```

## Finding where a step stops being reproducible

The numerical gate runs every case with mlops's `deterministic_kernels`
in effect, which asks each operation that offers the choice for the kernel
whose accumulation order is fixed. Without it a kernel that sums with atomics
returns a slightly different answer each run, across most of a step's gradient
tensors, and no comparison against a reference or against a replay can mean
anything. The ordered kernels cost
throughput, so they are not the default outside this gate. The request covers
reference generation as well as the planned run, so a regenerated reference is
itself reproducible.

The request is baked into a compiled graph without a guard, so a graph cached
from a run without it would be reused rather than recompiled. The matrix gives
each case its own compile cache and deletes it afterwards, which is what makes
that safe here; a tool that reuses a cache across the boundary would need to
key it on the setting.

Against the reference, every weight must agree within a relative L2 of
2.5 % (cosine at least 0.999, sign agreement at least 99 %); an optimizer
moment gets 5 %, because it is an accumulator whose reduction order follows the
plan, so the same arithmetic in two orders moves it further than it moves a
weight. A weight that is still nothing but its optimizer steps -- every
element of the reference within `steps` learning rates of zero, as a bias
started at zero is after a few steps -- has no scale of its own for a relative
bound to measure, so it is held to an absolute one instead: no element may be
further from the reference than two learning rates, which is one step whose
gradient sign the two runs' roundings disagreed on. The gate also requires a checkpoint replay to agree with the
uninterrupted run within tolerance, and records whether it agreed bit for bit
besides. When it
did not, the useful question is which stage of the step is not reproducible,
and the nondeterminism probe answers that rather than leaving it at
"somewhere in the backward":

```bash
python -m qualification.nondeterminism llama3 --model-implementation mlops
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


## Hardware defaults and dtype overrides

The numerical and performance launchers inspect the selected device before
constructing a case. SM80 and newer retain the existing BF16 weights,
gradients, and AdamW moments, with no master weights. GPUs below SM80 use
FP16 weights and gradients with FP32 moments and no masters. The hardware
probe runs in a short-lived process, so it cannot initialize the worker's
allocator before ShadowSpill installs it.

Both launchers accept independent overrides:

| Flag | Choices |
|---|---|
| `--model-dtype` | `float16`, `bfloat16`, `float32` |
| `--master-dtype` | `none`, `float16`, `bfloat16`, `float32` |
| `--grad-dtype` | `parameter`, `float16`, `bfloat16`, `float32` |
| `--opt-state-dtype` | `float16`, `bfloat16`, `float32` |

`parameter` means model-weight dtype for gradient accumulation. The optimizer
reads the completed gradient at the dtype of the weight or master it updates.
Every case prints the resolved dtypes and records them in its artifacts.
Nondefault precision is included in numerical reference identity; the original
BF16 identity remains valid. Custom factories own model and optimizer precision
through `--case-option` and can use the master/gradient flags independently.

Below SM80 only, numerical cases default to four layers and vocabulary size
8,192 while retaining their attention and expert widths. Their execution caps
are 3 GiB for each model family. The performance
execution pool defaults to 10 GiB on those devices; it stays at 16 GiB on
SM80+. Its spill pool remains 112 GiB everywhere. Explicit model fields,
per-family numerical `--budget` values, and performance
`--execution-budget-gib` / `--spill-budget-gib` flags override these defaults.


### Behavior on SM80 and newer

The numerical model geometries, execution budgets, reference paths and default
BF16 reference identities remain unchanged. The performance matrix retains its
three mlops cases, 16 GiB execution pool and 112 GiB spill pool. Quickstart and
Trainer keep BF16 model defaults on every architecture; FP16 is an explicit
choice there.

Other fixes apply on every GPU: automatic mlops implementation selection,
checkpoint dtype preservation, external-memory reporting unless
`--reject-overbudget` is selected, and incomplete-trace reporting. A throughput
floor applies only to the hardware and configuration it measured. These
changes preserve the default training precision but can affect kernel
selection, diagnostic output and gate verdicts.

Update both ShadowSpill and mlops, then rerun `scripts/setup.sh` with the desired
Python environment so the installed C libraries match the Python API. The setup
script uses a sibling mlops checkout when available. Existing default BF16
numerical references keep their identities; compiled build and plan artifacts
are separate and may need rebuilding after schema changes.

### Suite precision

The suite's generic low-precision GPU tests accept `--test-dtype float16`
(default: `bfloat16`). The choice travels to fresh CTest processes without a
device probe; BF16-specific CPU and metadata tests keep their declared dtype.
For example, on a pre-SM80 machine use:

```json
{
  "suite": ["--test-dtype", "float16"],
  "numerical": []
}
```

Run `python -m qualification.gates suite numerical --config <config.json>`.
Add `--regenerate-reference` only when intentionally replacing references.
The pre-SM80 reference root is `qualification/results/references/pre_sm80`;
SM80+ keeps `qualification/results/references/approximately_1b`. An explicit
`--reference-dir` takes precedence. No peer configured means the optional
network canary skips.

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

### CPU threads and local references

The gate launcher uses all logical CPUs available to its process by default,
respecting CPU affinity (including scheduler CPU allocations). It sets OpenMP,
MKL, and OpenBLAS thread counts for its children, overriding inherited shell
limits such as `OMP_NUM_THREADS=1`. The selected count appears in each gate log.
Use `python -m qualification.gates --cpu-threads 8` to set a smaller limit.
This controls CPU operations within each process; GPU tests still run sequentially.

Keep the canonical reference directory on local SSD:
`qualification/results/references/approximately_1b` (or `pre_sm80`).
Archive historical result directories separately; do not move the canonical
references onto cold storage. Each `reference.pt` and its `inputs.pt` sidecar
must be kept together. No regeneration is needed when relocating these files.

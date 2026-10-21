# Experiments

An experiment compares the training systems on one model, one geometry and
one precision contract: each engine sweeps its own options, and every case
of every engine shares the rest. This page says how an experiment is laid
out, written, run, resumed, summarized and traced, all through the common
tools of `baselines/`. [`GENERATING.md`](GENERATING.md) describes
`baselines.generate_experiment`, which writes a throughput and memory
experiment's folder from a template; a folder written by hand runs the same.

## The experiments here

An experiment's folder is named `<model>_<sequence length>K_<sequences per update>seqs`
under a folder for its kind and hardware. Full training on one 32 GiB GPU,
`full_rtx5090/`:

| Folder | Scenario |
|---|---|
| [`llama3-8b_4K_16seqs`](full_rtx5090/llama3-8b_4K_16seqs/README.md) | Llama 3 8B |
| [`qwen35-9b_4K_16seqs`](full_rtx5090/qwen35-9b_4K_16seqs/README.md) | Qwen3.5 9B |
| [`olmoe-7b-a1b_4K_16seqs`](full_rtx5090/olmoe-7b-a1b_4K_16seqs/README.md) | OLMoE 7B-A1B |
| [`llama3-1b_4K_8seqs_loss-curves`](full_rtx5090/llama3-1b_4K_8seqs_loss-curves/README.md) | Llama-3.2-1B's loss curves from its checkpoint in three engines |

LoRA from a checkpoint on one 32 GiB GPU, `lora_rtx5090/`:

| Folder | Scenario |
|---|---|
| [`qwen35-27b_4K_16seqs`](lora_rtx5090/qwen35-27b_4K_16seqs/README.md) | Qwen3.6 27B |
| [`qwen35moe-35b-a3b_4K_16seqs`](lora_rtx5090/qwen35moe-35b-a3b_4K_16seqs/README.md) | Qwen3.5 35B-A3B |
| [`qwen3moe-30b-a3b_4K_16seqs`](lora_rtx5090/qwen3moe-30b-a3b_4K_16seqs/README.md) | Qwen3 30B-A3B |

[`demo`](demo/README.md) checks the whole experiment path in minutes: Llama 3
1B with two cases per engine. `templates/` holds the generator's templates,
which are not experiments.

## An experiment's folder

```text
<experiment>/
  README.md          # what defines it, the scenario, each engine's matrix, how to run it
  experiment.json    # the spec: shared settings, engines, variables, summary budgets
  <engine>.json      # one matrix per engine: that engine's own options
  generation.json    # a generated folder's record of how it was made
  results/           # the published summary and its figures, once published
  output/            # everything a run writes; local, ignored by git
```

An experiment is defined by what all its cases share: the model, the
training mode, the sequence length, the sequences per step, the world size
and the hardware.
Its README opens with them in one table, and they title its figures, one
group of figures per distinct combination among its rows.

A folder can sit at any path under `baselines/experiments`. It is the whole
of the experiment's configuration: matrices are copies, never shared with
another experiment, so a change to one experiment changes no other. A run
writes into the folder's `output/`, which git ignores, so an experiment's
runs and logs stay with it; only the summary a run publishes into
`results/` is committed. `output` can be a symlink to another disk.

## Before an experiment

* Environments: `baselines/scripts/setup.sh` builds `baselines` (FSDP2,
  DeepSpeed), the repository's `scripts/setup.sh` builds `shadowspill`, and
  `baselines/scripts/setup_nvidia_training.sh` builds `nvidia-training`
  (Megatron-Core, AutoModel).
* Corpora: `python -m baselines.scripts.prepare_corpus` tokenizes a corpus
  once per tokenizer family (`../README.md`, Corpora).
* Checkpoints: a Hugging Face checkpoint directory per model, read through
  its family's reader.

## The spec, `experiment.json`

| Key | What it is |
|---|---|
| `schema_version` | 1 |
| `gpus` | The world size every case runs at, on one node |
| `python` | Per engine, the interpreter its trainers run in: a path, or the name of an environment beside the running one (`baselines` is `<prefix>/../baselines/bin/python`); by default the engine's own |
| `case_timeout` | Seconds an attempt may take, setup included (3600) |
| `summary_budgets_gib` | The GPU-memory budgets the summary's table is drawn at (optional) |
| `variables` | Each variable's name and what its value is (optional); a string in the settings or a matrix refers to one as `{name}` |
| `settings` | The shared settings, below |
| `sweeps` | One entry per engine: `name`, `engine`, `config` (its matrix, a path from the spec's folder or the repository's root), and optionally `python` and `case_timeout` of its own |
| `output` | Where a run writes (optional): a path from the spec's folder, `output` by default |

The shared settings are what every case of every engine takes, the fields
of `baselines/common/settings.py` that each engine's README lists under
"What a case names":

* the model: `model` (a catalog name), `implementation`, `mode` (`full` or
  `lora`), `lora` (`configure_lora`'s options), `model_config`;
* the data and weights: `data` (a corpus directory), `checkpoint` (a Hugging
  Face checkpoint directory), `seed`;
* the geometry: `sequence_length`, `tokens_per_step`, and
  `tokens_per_microbatch` when every engine runs one microbatch size;
* the precision contract: `model_dtype`, `master_dtype`, `grad_dtype`,
  `opt_state_dtype`, `parameter_rounding`, `opt_state_rounding`,
  `fuse_narrow_gradient_accumulation` (each engine's default when unnamed);
* the run: `warmup_steps`, `steps`, `learning_rate`, `weight_decay`,
  `cpu_threads`.

Any other field of the common settings can be shared too, by naming it
here, and a matrix that names a setting the spec's `settings` name is
refused, so no engine runs another model, geometry or contract than the
rest.

## A matrix, `<engine>.json`

```json
{
  "schema_version": 1,
  "defaults": {"offload_policy": "cpu"},
  "grid": [
    {"tokens_per_microbatch": [4096, 8192], "recompute": [0, 1]},
    {"tokens_per_microbatch": [4096], "forward_prefetch": [1, 2]}
  ]
}
```

`defaults` holds what every case of the engine takes; `grid` is one
Cartesian grid, or a list of grids whose cases are their union; `cases`, in
place of a grid, lists cases one by one. Every key is one of the
engine's settings, under the name its README gives it (`../engines/fsdp2/README.md`,
`../engines/deepspeed/README.md`,
`../engines/shadowspill/README.md`, `../engines/megatron_core/README.md`,
`../engines/automodel/README.md`). Every case is
validated when the matrix is expanded, before anything runs: an unknown
setting, a value out of range or a combination the engine cannot run is
refused there, with the reason. Two grids that name the same case run it
once.

## Running it

An experiment runs from the repository's root, in a tmux window. Its
engines import `baselines/` and the editable installs of `shadowspill` and
`mlops` as they are, so that code stays unchanged until the run ends. Which
code ran is the caller's to track; every row records the sources and commit
it ran from.

```bash
python -u -m baselines.experiment --spec baselines/experiments/<experiment>/experiment.json \
  [--variable NAME=VALUE ...] --dry-run
```

checks every case of every engine and launches nothing; the same command
without `--dry-run` runs the engines in the spec's order.

| Option | What it does |
|---|---|
| `--output DIR` | Write into DIR in place of the folder's `output/` |
| `--variable NAME=VALUE` | A value for a variable the spec declares; a resume takes the recorded values when none is given |
| `--engine NAME`, `--only SWEEP` | Run these engines, by engine or by sweep name, and keep what the others hold; repeatable |
| `--retry-failed` | Attempt the failed, out-of-memory and timed-out cases again |
| `--overwrite` | Discard what the engines this run runs hold, and run them from scratch; the other engines and the records of earlier runs stay |
| `--limit N` | At most N new attempts per engine in this run |
| `--publish DIR` | Also write the summary and its figures into DIR, the experiment's `results/` |
| `--skip-report` | Draw no report |
| `--dry-run` | Validate every case and write its command; launch nothing |

The same command again resumes: an engine keeps its completed cases and
attempts those never attempted or interrupted. An experiment can be
collected one engine at a time, `--engine shadowspill`, then
`--engine fsdp2`, or whole; the summary is the same either way.

An experiment can grow. A resume runs the engines' matrices as they are now:
a case keeps the name and the results it was given, by the digest of its
settings, so a matrix that adds cases, recompute fractions 0.25 to 0.75
beside 0 and 1, say, attempts only the new ones, and the summary and report
take every case the matrices name, old and new. A case a matrix no longer
names stays on disk but counts nowhere, and a changed shared setting
changes every case. What a resume refuses is an engine under another engine,
world size, interpreter or environment; which code ran is the caller's to
track, and every row records its sources. `--overwrite` starts the engines
a run runs from scratch.

## Case names

A case's folder is named for the settings its engine's matrix varies, each
as a short code and its value, then the first eight characters of the
digest of everything the case runs: `mb16384_rc0.25_act-cpu__1a2b3c4d` is
16,384 tokens per microbatch, a quarter recomputed and activations offloaded
to the CPU. A number follows its code; a word or a switch (`on`, `off`)
follows a hyphen; a setting without a code goes by its name. The digest
keeps every name unique, and a case keeps the name it was first given when
a grown matrix varies more settings, so a resume finds it.

| Code | Setting | Code | Setting |
|---|---|---|---|
| `mb` | `tokens_per_microbatch` | `rc` | `recompute` |
| `act` | `activation_offload` | `op` | `offload_policy` |
| `pin` | `pin_memory` | `fpf` | `forward_prefetch` |
| `bpf` | `backward_prefetch` | `raf` | `reshard_after_forward` |
| `rab` | `reshard_after_backward` | `gs` | `gradient_sync` |
| `cb` | `compile_blocks` | `f8` | `float8` |
| `opt` | `optimizer` | `og` | `offload_gradients` |
| `z` | `zero_stage` | `ofp` | `offload_param` |
| `ofo` | `offload_optimizer` | `dc` | `deepcompile` |
| `ex` | `execution_gib` | `sp` | `spill_gib` |
| `res` | `resolution_options` | | |

## What a run writes

```text
<experiment>/output/
  experiment.json            # the latest spec with its variables, each engine's latest run, every run
  runs/<stamp>.json, .log    # each run's record and console
  matrices/<engine>.json     # each engine's matrix with the shared settings joined
  sweeps/<engine>/
    manifest.json            # engine, sources, environment and every case
    README.md, results.json  # every case's latest outcome
    launcher.log             # the engine's console
    cases/<case>/
      config.json, command.txt
      attempt-NNN/
        command.txt          # the launch command
        console.log          # the trainer's stdout and stderr, in order
        case.json            # the resolved case
        rank-NNNNN/          # each rank's request, result or failure
        outcome.json, row.json
  report/                    # baselines.report's figures and tables
  summary.md, summary.json   # baselines.summary's summary
  traces/                    # baselines.trace's traces
```

The console shows the spec's settings and every engine's case count, then
each engine under its heading: a line as each case starts, with the
settings that distinguish it, and a line as it ends, with its outcome and,
when it completed, its tokens per second, update time, GPU peak held and in
use, and host peak. A case's own output stays in its `console.log`.

## Summaries and reports

A run ends with `baselines.report` over every row the experiment holds,
drawn against the GPU memory each engine held, and `baselines.summary`,
which gives each engine's outcomes, the fastest case of each engine within
each budget of `summary_budgets_gib`, and each engine's frontier, with the
report's figures embedded. Either runs again on its own:

```bash
python -m baselines.summary baselines/experiments/<experiment>/output \
  [--budgets GIB ...] [--publish baselines/experiments/<experiment>/results]
python -m baselines.report baselines/experiments/<experiment>/output/sweeps \
  --output baselines/experiments/<experiment>/output/report --memory reserved
```

`--publish` writes the summary into the experiment's `results/` with its
figures beside it and no path of the machine it ran on, to be committed
with the experiment. GPU memory is what an engine held at its peak: the
execution pool's capacity for ShadowSpill, which holds it whole, and the
peak its allocator reserved for any other engine.

## Tracing a case

```bash
python -m baselines.trace baselines/experiments/<experiment>/output <engine> <case> \
  [--steps N] [--devices LIST] [--capture measured|whole] [--output DIR] [--name NAME]
```

runs one case again as its engine ran it, under `nsys profile` with the
GPU's metrics sampled, into `output/traces/<engine>/<case>/<stamp>/`
unless `--output` says otherwise; `python -m baselines.trace --help` lists
every option. A trace is not a throughput result, and no summary or report
reads it.

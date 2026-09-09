# Program collection

This harness **builds programs**. It constructs each model, exports and
compiles it, profiles real kernels, and lowers the result into one reusable
`StepProgram` per case. It never runs the planner, and it takes no planning
arguments: what a planner later does with a program is
[planning evaluation](../planning_eval/README.md)'s business, and its own
store.

A collection needs an execution device and a working provider backend,
because it profiles real kernels. A full matrix takes hours.

## Running it

```bash
python -m benchmarking.program_collection.collect \
  --config benchmarking/program_collection/configs/full_model_program_corpus_v1.json \
  --output-dir benchmarking/datasets/input_programs/full_model_program_corpus_<rev> \
  --artifact-store benchmarking/program_collection/planning_caches/full_model_program_corpus_<rev>
```

Resume, validating and skipping the Programs already collected:

```bash
python -m benchmarking.program_collection.collect \
  --config benchmarking/program_collection/configs/full_model_program_corpus_v1.json \
  --output-dir benchmarking/datasets/input_programs/full_model_program_corpus_<rev> \
  --artifact-store benchmarking/program_collection/planning_caches/full_model_program_corpus_<rev> \
  --resume
```

`--config`, `--output-dir`, and `--artifact-store` are required; the rest
select subsets or override the config for one run:

| Argument | Meaning |
|---|---|
| `--resume` | Validate completed Programs, skip them, and retry the incomplete cases. Without it, an output directory that already holds case state is refused. |
| `--dry-run` | Print the expanded matrix and exit. |
| `--case GLOB` | Collect only matching case IDs; repeatable. |
| `--start-at CASE_ID` | Start at one exact case ID after filtering. |
| `--limit N` | Keep the first N selected cases. |
| `--timeout-seconds`, `--max-attempts` | Override the config's per-case limits. |
| `--quiet-plan` | Silence each build's own phase reporting. |
| `--build-store-mode` | Override `build.build_store_mode` for this run: `contribute`, `reuse`, `require`, or `refresh`. |
| `--revision SHA` | The revision recorded on every case this run produces; defaults to HEAD. It labels the run and checks nothing out. |

`--build-store-mode` says what a worker does about a build artifact the store
does not hold, or holds stale: `contribute` reads what is there and writes
back what is not, `reuse` reads and persists nothing, `require` refuses a
miss, and `refresh` ignores what is there and rebuilds over it. Keep the
store on a local filesystem; it is written throughout a collection.

Every case runs in a fresh subprocess. Python exceptions, timeouts, signals,
and process exits are attributed to one case, recorded, and do not prevent
later cases from running. A successful program is published atomically as
soon as it exists.

## The configuration

One JSON file, validated strictly — an unknown or missing key is an error.
Beside its `schema`, the collection's `name`, `seed`, `expected_programs`,
`case_timeout_seconds`, and `max_attempts`, it has four sections:
`geometry`, `models`, `runtime`, and `build`.

`geometry` gives the three axes — `tokens_per_microbatch`,
`sequence_lengths`, and `accumulation_rounds` — expanded into one program per
divisible combination, per model. In user-facing text the third axis is
**gradient accumulation rounds**; the schema-v1 field name
`accumulation_steps` is still written and accepted, to keep collected digests
stable.

`build` is the `BuildSpec`: `optimizer_ordering`, `allocation_probe_seeds`,
`allocation_probe_repetitions`, `build_store_mode` (`contribute`, `reuse`, or
`require` here; `refresh` is a per-run override only), and
`implementation_revision`. Nothing in it is a planning setting.

`runtime` names the pool capacities and the budgets the build sees. They are
recorded as provenance on each case, not planned against here.

## What it writes

```text
<output>/
├── README.md
├── layout.json
├── cases/<provider-model>/<data-geometry>/<program-digest>/
│   ├── manifest.json      identity and collection provenance
│   └── step_program.json  the ShadowSpillProgram itself, named by its digest
└── _collections/<name>-<config-digest>/
    ├── collection.lock
    ├── config.json
    ├── collection.log
    ├── summary.json
    └── cases/<case-id>/
        ├── request.json
        ├── status.json
        ├── worker-result-NNNN.json
        └── logs/attempt-NNNN.log
```

A case manifest records what produced the program beside it: the collection
name and config, the model, the data geometry, the seed, and the runtime
configuration the costs were measured under. None of it is part of the
program's digest and none of it is read back when the program is planned, so
a field added here cannot invalidate a corpus. What the digest does cover is
in [the artifact store guide](../../docs/python/artifact-store.md#identity).

Journal paths are relative to the dataset root, so moving a complete dataset
does not invalidate resume or integrity validation. When and at what revision
a dataset was collected is recorded in its own collection log under
`_collections/` and on every case manifest, not here.

## Collecting on another machine

The measured task costs are written into each `StepProgram`, so a corpus
describes the machine that collected it. That is what makes a corpus reusable
by `planning_eval` without a GPU, and it is also the reason a corpus is not a
portable description of different hardware. Evaluating planning against
another machine's costs means collecting a corpus there; reusing this one
measures the planner against the costs recorded here, whatever machine the
planner runs on. Name the output directory for the revision that collected
it, so the two cases stay distinguishable.

# Program collection

This package builds reusable pre-PressureFit `StepProgram` inputs. It owns
model construction, Export/AOT/Inductor compilation, structural profiling,
Program lowering, crash isolation, and the collection journal. It does not run
PressureFit.

The obvious launcher is:

```bash
python -m benchmarking.program_collection.collect \
  --config benchmarking/program_collection/configs/full_model_program_corpus_v1.json \
  --output-dir benchmarking/datasets/input_programs/full_model_program_corpus_<rev> \
  --artifact-store benchmarking/program_collection/planning_caches/full_model_program_corpus_<rev>
```

The v1 configuration expands model providers and `DataGeometry` axes into one
Program per point. In user-facing text, the third geometry axis is **gradient
accumulation rounds**. The schema-v1 internal field name `accumulation_steps`
is retained on write to keep digests stable, and accepted on read.

Every Program runs in a fresh subprocess. Python exceptions, timeouts, signals,
and process exits are attributed to one case, recorded, and do not prevent later
cases from running. Successful artifacts are atomically published immediately.

Resume and validate an existing dataset without rebuilding completed Programs:

```bash
python -m benchmarking.program_collection.collect \
  --config benchmarking/program_collection/configs/full_model_program_corpus_v1.json \
  --output-dir benchmarking/datasets/input_programs/full_model_program_corpus_<rev> \
  --artifact-store benchmarking/program_collection/planning_caches/full_model_program_corpus_<rev> \
  --resume
```

`--dry-run` prints the expanded matrix. `--case GLOB`, `--start-at CASE_ID`, and
`--limit N` select development subsets. `--revision`, `--timeout-seconds`,
`--max-attempts`, `--quiet-plan`, and `--force-fresh` control provenance,
per-case limits, and cache reuse. Active compiler/profile caches should
reside on a local filesystem.

## Dataset layout

```text
<output>/
├── README.md
├── layout.json
├── cases/<provider-model>/<data-geometry>/<program-digest>/
│   ├── manifest.json      identity and collection provenance
│   └── step_program.json  the Program itself, named by its digest
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

A case manifest records what produced the Program beside it: the collection
name and config, the model, the data geometry, the seed, and the runtime
configuration the costs were measured under. None of it is part of the
Program's digest and none of it is read back when the Program is planned, so
a field added here cannot invalidate a corpus. What the digest does cover is
in [the artifact store guide](../../docs/python/artifact-store.md#identity).

Journal paths are relative to the dataset root, so moving a complete dataset
does not invalidate resume or integrity validation. When and at what revision
a dataset was collected is recorded in its own collection log under
`_collections/` and on every case manifest, not here.

## Collecting on another machine

Collection needs an execution device and a working provider backend: it
builds each model, compiles it, and profiles real kernels, so a full matrix
takes hours.

The measured task costs are written into each `StepProgram`, so a corpus
describes the machine that collected it. That is what makes a corpus reusable
by `planning_eval` without a GPU, and it is also the reason a corpus is not a
portable description of different hardware. Evaluating planning against
another machine's costs means collecting a corpus there; reusing this one
measures the planner against the costs recorded here, whatever machine the
planner runs on. Name the output directory for the revision that collected
it, so the two cases stay distinguishable.

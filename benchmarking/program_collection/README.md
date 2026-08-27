# Program collection

This package builds reusable pre-PressureFit `StepProgram` inputs. It owns
model construction, Export/AOT/Inductor compilation, structural profiling,
Program lowering, crash isolation, and the collection journal. It does not run
PressureFit.

The obvious launcher is:

```bash
python -m benchmarking.program_collection.collect \
  --config benchmarking/program_collection/configs/full_model_program_corpus_v1.json \
  --output-dir benchmarking/datasets/input_programs/full_model_program_corpus_v1 \
  --artifact-store benchmarking/program_collection/planning_caches/full_model_program_corpus_v1
```

The v1 configuration expands model providers and `DataGeometry` axes into 168
Programs. In user-facing text, the third geometry axis is **gradient
accumulation rounds**. The schema-v1 internal field name `accumulation_steps`
is accepted only to read the existing immutable dataset.

Every Program runs in a fresh subprocess. Python exceptions, timeouts, signals,
and process exits are attributed to one case, recorded, and do not prevent later
cases from running. Successful artifacts are atomically published immediately.

Resume and validate an existing dataset without rebuilding completed Programs:

```bash
python -m benchmarking.program_collection.collect \
  --config benchmarking/program_collection/configs/full_model_program_corpus_v1.json \
  --output-dir benchmarking/datasets/input_programs/full_model_program_corpus_v1 \
  --artifact-store benchmarking/program_collection/planning_caches/full_model_program_corpus_v1 \
  --resume
```

`--dry-run` prints the expanded matrix. `--case GLOB`, `--start-at CASE_ID`, and
`--limit N` select development subsets. Active compiler/profile caches should
reside on a local filesystem.

## Dataset layout

```text
<output>/
├── README.md
├── layout.json
├── cases/<provider-model>/<data-geometry>/<program-digest>/
│   ├── manifest.json
│   └── step_program.json
└── _collections/<name>-<config-digest>/
    ├── config.json
    ├── collection.log
    ├── summary.json
    └── cases/<case-id>/
        ├── request.json
        ├── status.json
        ├── worker-result-NNNN.json
        └── logs/attempt-NNNN.log
```

Journal paths are relative to the dataset root. Moving the complete dataset
does not invalidate resume or integrity validation. The current 168 Programs
and planning cache were moved into this tree; they were not recollected.

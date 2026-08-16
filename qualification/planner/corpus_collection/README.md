# Program corpus collection

This is the input-building half of the planner qualification pipeline. The
sibling `../frontier_collection/` package consumes these saved Programs without
repeating capture, compilation, or profiling.

The collector captures reusable `StepProgram` artifacts and stops before
PressureFit. A strict JSON configuration specifies registered models, packed
data-geometry axes, runtime capacities, and profiling behavior.

Each Program runs in a fresh subprocess so a Python exception, timeout,
signal, or native crash cannot corrupt later cases. The controller records the
failure in that case's `status.json` and attempt log, then always advances to
the next Program. Successful artifacts are written atomically as soon as each
case finishes. Providers are round-robin interleaved so provider-specific
failures surface near the beginning of a multi-hour collection.

Run the complete configured matrix:

```bash
python -m qualification.planner.collect_corpus \
  --config qualification/planner/configs/full_model_program_corpus_v1.json \
  --output-dir /local/storage/shadowspill/program-corpus-v1 \
  --planning-cache /local/storage/shadowspill/planning-cache-v1
```

Resume after interruption, validating and skipping every completed Program:

```bash
python -m qualification.planner.collect_corpus \
  --config qualification/planner/configs/full_model_program_corpus_v1.json \
  --output-dir /local/storage/shadowspill/program-corpus-v1 \
  --planning-cache /local/storage/shadowspill/planning-cache-v1 \
  --resume
```

Use `--dry-run` to validate and print the expanded matrix. `--case GLOB`,
`--start-at CASE_ID`, and `--limit N` select development subsets. A subsequent
full invocation must use `--resume` so those completed cases are reused.

The output contains the human-navigable corpus under `cases/` and controller
state under `_collections/<name>-<config-digest>/`:

- `collection.log`: timestamped collection-level start, stop, skip, and error
  records;
- `summary.json`: atomically refreshed aggregate progress;
- `cases/<case-id>/request.json`: the exact resolved request;
- `cases/<case-id>/status.json`: all attempts and the validated artifact;
- `cases/<case-id>/logs/attempt-NNNN.log`: complete timestamped worker output;
- `cases/<case-id>/worker-result-NNNN.json`: structured success or traceback.

Use a local filesystem for the active planning cache. Network storage is
appropriate as an archive destination after complete artifacts have been
written locally.

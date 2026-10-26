# Program collection

Collect reusable `ProfiledProgram` archives with `build_step` →
`measure_program` → `apply_profiles`. Models are constructed on `meta`;
measurement uses the ordinary PyTorch allocator. Collection does not create
execution/spill pools, run planning, or initialize full model weights.

```bash
python -m benchmarking.program_collection.collect \
  --config benchmarking/program_collection/configs/full_model_program_corpus_v2.json \
  --output-dir benchmarking/datasets/input_programs/my_collection
```

Add `--resume` to validate and skip completed cases. `--dry-run` prints the
matrix; `--case GLOB`, `--start-at CASE_ID`, and `--limit N` select cases.
Timeouts and failures are recorded per case and collection continues.
`--timeout-seconds`, `--max-attempts`, and `--revision` override run controls.
Each case runs in a fresh subprocess.

The version-2 JSON configuration has three sections:

- `models`: catalog names, implementations, optional geometry and head scratch.
- `geometry`: token counts, sequence lengths and accumulation counts. Packed
  sequence lengths are passed as actual model inputs; one shape can describe
  multiple examples without fixing their execution order.
- `profiling`: CUDA device, warmup/measurement seconds and external headroom bytes.

The shipped matrix still contains 168 cases. It requires a compatible GPU and
providers. Old pool budgets, build-store modes and optimizer ordering are removed
from collection configuration. Compiler caches use their normal external paths.

```text
<output>/
├── cases/<provider-model>/<geometry>/<digest>/
│   ├── manifest.json          identity, provenance and checksum
│   └── profiled_program.ssp   Program, task code, timings and memory evidence
└── _collections/<name>-<config-digest>/
    ├── config.json
    ├── collection.log
    ├── summary.json
    └── cases/<case-id>/        attempts, worker results and logs
```

Archives are self-contained and published atomically. They can be loaded with
`ProfiledProgram.load(path)` and passed to the public planning APIs. The
[planning evaluator](../planning_eval/README.md) also reads them directly and
supplies budgets and transfer rates on CPU. Existing JSON corpora remain
readable; they are not rewritten. The collector currently targets one GPU.

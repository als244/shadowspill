# ShadowSpill benchmarking

This tree is the self-contained, reproducible planning benchmark. It depends
on the public ShadowSpill and model APIs, but never imports scripts or files
from `qualification/`.

```text
benchmarking/
├── data_geometry.py
├── datasets/
│   └── input_programs/                 immutable pre-PressureFit datasets
├── program_collection/
│   ├── collect.py                      collection launcher
│   ├── configs/                        versioned collection matrices
│   ├── corpus.py                       Program serialization and validation
│   └── planning_caches/                local Export/compile/profile cache
└── planning_eval/
    ├── evaluate.py                     frontier launcher
    ├── configs/                        versioned budget/bandwidth matrices
    ├── plan_artifacts.py               annotated-plan serialization
    └── results/                        complete measured baselines
```

The workflow has two independent phases:

1. `program_collection` performs model construction, Export/AOT/Inductor
   compilation, structural profiling, and Program lowering. It stops before
   PressureFit and publishes immutable `StepProgram` inputs.
2. `planning_eval` reads those Programs without rebuilding models and evaluates
   PressureFit, simulation, and physical admission across budgets and transfer
   bandwidths.

`DataGeometry` groups sequence length, tokens and sequences per microbatch,
gradient accumulation rounds, and tokens per optimizer step. Logs and result
records use that terminology consistently.

The current full dataset is:

`benchmarking/datasets/input_programs/full_model_program_corpus_v1`

It contains 168 already-collected Programs. Generated datasets, caches, and
results are intentionally git-ignored; their versioned configurations,
launchers, schemas, and documentation are tracked.

See [program_collection/README.md](program_collection/README.md) to build or
validate Program inputs and [planning_eval/README.md](planning_eval/README.md)
to reproduce a PressureFit frontier.

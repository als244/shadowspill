# ShadowSpill benchmarking

This tree is the self-contained, reproducible planning benchmark. It depends
on the public ShadowSpill and model APIs, but never imports scripts or files
from `qualification/`.

```text
benchmarking/
├── data_geometry.py
├── _serialization.py                   shared encode/decode helpers
├── datasets/
│   └── input_programs/                 immutable pre-PressureFit datasets
├── quickstart.py                       plan and run one model, story told
├── quickstart.md                       its guide: flags, phases, terms
├── program_collection/
│   ├── collect.py                      collection launcher
│   ├── configs/                        versioned collection matrices
│   ├── corpus.py                       Program serialization and validation
│   └── planning_caches/                local Export/compile/profile artifact store
└── planning_eval/
    ├── evaluate.py                     frontier launcher
    ├── planning_caches/                the frontier run's own artifact store
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

A dataset lives under `datasets/input_programs/`, named for its collection
configuration and the revision that collected it, and holds one `StepProgram`
per case; a frontier's results live under `planning_eval/results/`, named the
same way for the revision measured. Both, and the artifact stores, are
git-ignored. What is tracked is what reproduces them: the versioned
configurations, the launchers, the schemas, and these guides. Which dataset
is current is a fact about the checkout, read from `input_programs/` rather
than from here.

See [program_collection/README.md](program_collection/README.md) to build or
validate Program inputs and [planning_eval/README.md](planning_eval/README.md)
to reproduce a PressureFit frontier.

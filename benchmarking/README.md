# ShadowSpill benchmarking

This tree is the self-contained, reproducible planning benchmark. It depends
on the public ShadowSpill and model APIs, but never imports scripts or files
from `qualification/`.

```text
benchmarking/
├── data_geometry.py                    the shared geometry vocabulary
├── _serialization.py                   shared encode/decode helpers
├── datasets/
│   └── input_programs/                 immutable StepProgram corpora
├── quickstart.py                       plan and run one model, story told
├── quickstart.md                       its guide: flags, phases, terms
├── quickstart_reports/                 one directory per quickstart run
├── program_collection/
│   ├── collect.py                      the launcher
│   ├── configs/                        versioned collection matrices
│   ├── corpus.py                       StepProgram serialization and validation
│   └── planning_caches/                its build artifact stores
└── planning_eval/
    ├── evaluate.py                     the launcher
    ├── configs/                        versioned budget/bandwidth matrices
    ├── plan_artifacts.py               annotated-plan serialization
    ├── planning_caches/                its planning artifact stores
    └── results/                        complete measured baselines
```

## Three entry points

[**Quickstart**](quickstart.md) takes one model end to end on this machine:
it searches microbatch geometries across execution budgets, optionally
renders figures, and runs the winning plan. It is the tour, and the fastest
way to see the whole system work.

The other two split one long job in half, so the expensive half is paid once:

1. [**Program collection**](program_collection/README.md) builds programs.
   It constructs models, exports and compiles them, profiles real kernels,
   and lowers the result, stopping before the planner. It needs a device, and
   publishes immutable `StepProgram` inputs.
2. [**Planning evaluation**](planning_eval/README.md) plans those programs
   across budgets and transfer bandwidths, without rebuilding a model or
   running a kernel. It needs no device.

The split is enforced by the arguments each one accepts: collection takes no
planning arguments, and evaluation takes no build arguments beyond the path
its corpus lives at.

## Terms and stores

`DataGeometry` groups sequence length, tokens and sequences per microbatch,
gradient accumulation rounds, and tokens per optimizer step. Logs and result
records use that terminology consistently.

An artifact store has two trees, and every surface here names them the same
way: `--artifact-store` roots both, while `--build-store` and `--plan-store`
override either. The **build** tree holds captures, graph pairs, profiles and
compiled artifacts; the **planning** tree holds selection requests, results
and plan manifests. A mode per tree says what a run does with it —
`contribute` reads hits and writes misses, `reuse` reads and persists
nothing, `require` refuses a miss, `refresh` ignores hits and rebuilds over
them — and the flags are `--build-store-mode` and `--plan-store-mode`. The
[artifact store guide](../docs/python/artifact-store.md) is the authority on
what each tree holds and how a key is formed.

## What is tracked

A corpus lives under `datasets/input_programs/`, named for its collection
configuration and the revision that collected it, and holds one `StepProgram`
per case; a frontier's results live under `planning_eval/results/`, named the
same way for the revision measured. Those, the quickstart reports, and the
artifact stores are git-ignored. What is tracked is what reproduces them: the versioned
configurations, the launchers, the schemas, and these guides. Which corpus is
current is a fact about the checkout, read from `input_programs/` rather than
from here.

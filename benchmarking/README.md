# ShadowSpill benchmarking

This tree is the self-contained, reproducible planning benchmark. It depends
on the public ShadowSpill and model APIs, but never imports scripts or files
from `qualification/`.

```text
benchmarking/
├── data_geometry.py                    the shared geometry vocabulary
├── _serialization.py                   shared encode/decode helpers
├── datasets/
│   └── input_programs/                 immutable profiled-program corpora
├── quickstart/                         plan and run one model, story told
├── quickstart.md                       its guide: flags, phases, terms
├── bandwidth_frontier/                 CPU sweeps from ProfiledProgram archives
├── quickstart_reports/                 one directory per quickstart run
├── program_collection/
│   ├── collect.py                      the launcher
│   ├── configs/                        versioned collection matrices
│   ├── artifacts.py                    phase archives and JSON corpus readers
│   └── worker.py                       isolated build/profile worker
└── planning_eval/
    ├── evaluate.py                     the launcher
    ├── configs/                        versioned budget/bandwidth matrices
    ├── plan_artifacts.py               annotated-plan serialization
    ├── planning_caches/                its planning artifact stores
    └── results/                        complete measured baselines
```

## Entry points

[**Quickstart**](quickstart.md) takes one model end to end on this machine:
it searches microbatch geometries across execution budgets, optionally
renders figures, and runs the winning plan. It is the tour, and the fastest
way to see the whole system work.

Quickstart uses `write_report` for saved-Plan summaries, plots and timelines.
The [video tools](../src/tools/pressurefit_trace/README.md#completed-quickstart-winners)
render completed winners directly from their saved Plans.

[**Bandwidth frontier**](bandwidth_frontier/README.md) replans saved
`ProfiledProgram` candidates across memory budgets and communication rates.
It needs no model construction, GPU, runtime pool, or artifact store.

The other two split one long job in half, so the expensive half is paid once:

1. [**Program collection**](program_collection/README.md) builds programs.
   It constructs model structure, exports, compiles and lowers tasks, then
   profiles real kernels. It needs a device and publishes immutable
   `ProfiledProgram` archives without creating execution pools.
2. [**Planning evaluation**](planning_eval/README.md) plans those programs
   across budgets and transfer bandwidths, without rebuilding a model or
   running a kernel. It needs no device.

The split is enforced by the arguments each one accepts: collection takes no
planning arguments, and evaluation takes no build arguments beyond the path
its corpus lives at.

## Terms and artifacts

`DataGeometry` groups sequence length, tokens and sequences per microbatch,
gradient accumulation rounds, and tokens per optimizer step. Logs and result
records use that terminology consistently.

Phase workers exchange explicit saved artifacts. Compiler caches stay separate
from results. The planning evaluator retains its optional planning cache and
can read existing JSON corpora; its own guide documents those options.

## What is tracked

A corpus lives under `datasets/input_programs/`, named for its collection
configuration and the revision that collected it, and holds one `ProfiledProgram`
per case; a frontier's results live under `planning_eval/results/`, named the
same way for the revision measured. Those, the quickstart reports, and the
artifact stores are git-ignored. What is tracked is what reproduces them: the versioned
configurations, the launchers, the schemas, and these guides. Which corpus is
current is a fact about the checkout, read from `input_programs/` rather than
from here.

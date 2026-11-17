# Benchmarks

Tools for measuring ShadowSpill preparation, planning and execution.
Comparisons with other training systems live in the separate `shadowspill_baselines`
project. For ordinary training, start with the
[client demos](../examples/demos/README.md).

```text
benchmarks/
  factories/           Reproducible model and phase configurations
  quickstart/          Build, profile, plan and execute a selected configuration
  plan_runner/         Measure an existing saved Plan
  stage_traces/        Inspect individual compiled stages with Nsight
  program_collection/  Build and profile a corpus of workloads
  planning_eval/       Run CPU planning experiments over that corpus
  bandwidth_frontier/  Sweep memory budgets and hypothetical transfer speeds
  host_memory/         Host-memory checks before creating spill pools
```

`data_geometry.py`, `journals.py`, `spill.py` and `_serialization.py` hold shared
benchmark records and storage helpers. Process launch and phase monitoring come
from the installed [orchestration package](../packages/engine/src/shadowspill/orchestration/README.md).
The engine does not import benchmark programs.

## Choose a workflow

| Goal | Entry point |
| --- | --- |
| Prepare and measure one model configuration | [Phase quickstart](quickstart.md) |
| Measure a saved Plan without rebuilding | [Plan runner](plan_runner/README.md) |
| Explain a stage's kernel time or temporary memory | [Stage traces](stage_traces/README.md) |
| Collect reusable GPU measurements | [Program collection](program_collection/README.md) |
| Compare planner choices across measured workloads | [Planning evaluation](planning_eval/README.md) |
| Sweep budgets and transfer assumptions without a GPU | [Bandwidth frontier](bandwidth_frontier/README.md) |

Collection builds and profiles on a representative GPU. Planning evaluation and
bandwidth sweeps read saved `ProfiledProgram` artifacts on CPU. Saved-Plan runs
measure actual execution; isolated stage traces explain kernels and do not
represent end-to-end throughput.

Source, experiment configurations and result schemas are tracked. Generated
profiles, Plans, reports and compiler caches use ignored output directories.
Keep measurement artifacts and compiler caches separate, and retain source and
configuration identities with results.

Release regression verdicts belong to [qualification](../qualification/README.md).
These benchmark tools support investigation and reproducible measurements.

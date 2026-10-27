# PressureFit capture and analysis

Record planner decisions from saved profiles, inspect every cut, and retain the
winning schedules. These tools run on CPU; model build and GPU measurement happen
beforehand. Generated data belongs in an ignored run directory such as `runs/`.

```sh
python -m tools.pressurefit_trace --help
python -m tools.pressurefit_trace analyze runs/example/trace --output runs/example/analysis
```

The [Python trace API](../../../docs/python/pressurefit-trace.md) covers recording;
the [algorithm guide](../../../docs/architecture/pressurefit.md) defines the policies.
All animation, frame rendering, encoding, galleries, and video-specific documentation
live under [video/](video/README.md).

## Capture new traces

For an arbitrary program, the simplest route is the normal planning API:

```python
from shadowspill.planner import GenericPlanningOptions, SearchOptions, plan_program

plan = plan_program(
    problem,
    search_options=SearchOptions(
        generic=GenericPlanningOptions(deterministic=True),
        trace_directory="runs/example/traces",
    ),
)
```

The API finds the winner, reruns it with tracing, and verifies the result.
Explicit `trace_candidates` pairs record other resolutions/policies. See the
[trace API guide](../../../docs/python/pressurefit-trace.md) for scope, metadata,
and normal frontend semantic-label export. The optional `semantic_labels.json`
sidecar must have the same program digest; the renderer works with the generic
object/task labels when that sidecar is absent.

For an already profiled phase quickstart, all collection commands below run
on CPU. Build/profile/calibrate first with the normal quickstart phases:

```sh
# Read request.json, profiled/*.ssp and communication.ssp; no factory import.
python -m tools.pressurefit_trace dump runs/example/quickstart \
    --output runs/example/dump

python -m tools.pressurefit_trace plan runs/example/dump --output runs/example/plans
python -m tools.pressurefit_trace trace runs/example/plans --output runs/example/traces

# Optional full experiment: every ordering/geometry program, every resolution/policy.
python -m tools.pressurefit_trace trace runs/example/plans \
    --programs all --all-candidates --output runs/example/all_traces

# Optional compressed, verified export of one trace or a whole collection.
python -m tools.pressurefit_trace export runs/example/all_traces \
    --output runs/example/export
```

Dump/plan/trace/export require fresh output directories. The dump enumerates
supported example orderings from saved Programs without rebuilding. It uses the
largest requested physical budget unless `dump --physical-gib` overrides it.
Across geometries, it reserves the largest recorded fixed device overhead,
including communication resources; each candidate's original value is also saved.
This conservative common allowance makes the diagnostic comparisons consistent.

`plan` defaults to the exported capacities. `--execution-gib` specifies usable
planner capacity; `--physical-gib` subtracts the recorded common overhead. These
options are mutually exclusive. A dump bounds the diagnostic searches at its
exported cap; export again with a larger `dump --physical-gib` to raise it, using
the same saved profiles without any GPU work. The artifacts are diagnostic
planner inputs and annotated JSON, not portable Executor plans. Use the public
`plan_program` API or normal quickstart plan phase to produce an executable.

Within the exported cap, one dump can feed several independent budget searches:

```sh
python -m tools.pressurefit_trace plan runs/example/dump \
    --physical-gib 7 --spill-gib 128 --output runs/example/budget_7/plans
python -m tools.pressurefit_trace plan runs/example/dump \
    --physical-gib 16 --spill-gib 128 --output runs/example/budget_16/plans

# Reuse the compute profiles, but search against hypothetical transfer rates.
# Each direction retains 450 decimal GB/s when both lanes copy simultaneously.
python -m tools.pressurefit_trace plan runs/example/dump \
    --physical-gib 16 --spill-gib 128 --transfer-bandwidths 450,450 \
    --output runs/example/budget_16_450gbps/plans
python -m tools.pressurefit_trace trace runs/example/budget_16_450gbps/plans \
    --output runs/example/budget_16_450gbps/traces
```

`--transfer-bandwidths` changes planning and simulation; it does not run a GPU
copy at that speed or alter the captured compute profiles. The two-number form
sets fetch and eviction rates, including concurrent operation. Four numbers
set fetch solo/concurrent and eviction solo/concurrent separately; for example,
`450,225,450,225` models an aggregate 450 GB/s link shared equally while both
lanes copy. Transfer latencies retain the captured values. The saved plan records
the effective calibration, and `trace` reuses it when recording the winner, so
the cut trajectory corresponds to the hypothetical plan rather than the capture's
original transfer rates. This records planner decisions, not measured execution.

Failures already recorded by the profile phase remain in `dump.json` and each
budget's `plans.json`. Unsupported example orderings also remain explicit
rejections. Dump never runs profiling, retries a failed geometry or changes the
source archives. This physical-placement diagnostic requires recorded allocation
evidence. For simulations using estimated memory, use the public
`plan_program`/`write_report` path; estimates cannot become execution evidence.

`--workers` controls CPU planning. `trace` defaults to the winning program and
winning candidate. `--candidates-json pairs.json` supplies a JSON list of
`[selection_id, candidate_id]` pairs and is mutually exclusive with
`--all-candidates`. Full collections can be much larger than winner-only traces.

Collection arguments:

| Command | Argument | Meaning/default |
| --- | --- | --- |
| `dump` | `input`, `--output` | Saved phase quickstart and fresh output directory for diagnostic inputs. |
| `dump` | `--physical-gib` | Largest exported total GPU cap; defaults to the largest requested quickstart budget. |
| `plan` | `input`, `--output` | Capture directory and required fresh planning output directory. |
| `plan` | `--physical-gib` | Physical execution cap; subtract the capture's runtime/headroom carve-out. |
| `plan` | `--execution-gib` | Usable planner execution capacity; mutually exclusive with `--physical-gib`. Without either, retain captured budgets. |
| `plan` | `--spill-gib` | Override the captured spill capacity. |
| `plan` | `--transfer-bandwidths` | Optional decimal GB/s override: `FETCH,EVICT` or `FETCH_SOLO,FETCH_CONCURRENT,EVICT_SOLO,EVICT_CONCURRENT`. Compute profiles and transfer latencies stay as captured. |
| `plan` | `--workers` | CPU planning workers; `0` (default) delegates worker selection to the planner. |
| `trace` | `input`, `--output` | Planning directory and required fresh trace collection directory. |
| `trace` | `--programs winner\|all` | Record the winning program (default) or every feasible program. |
| `trace` | `--all-candidates` | Record every resolution/policy pair instead of only each program's winner. |
| `trace` | `--candidates-json` | JSON list of `[selection_id, candidate_id]` pairs; mutually exclusive with `--all-candidates`. |
| `export` | `input`, `--output` | Raw trace or collection, and required fresh directory for compressed, verified exports. |

The collection's `dump.json`, `plans.json`, and `collection.json` link stages
and retain their source paths; keep the collection tree in place while using
those links. A raw per-search trace directory is self-contained and can be
moved for analysis. The exporter stores large context tables once per resolution.
Collected raw traces also include `collection_context.json`, checked against
the program digest. It supplies the movie's workload, budget, ordering and
microbatch labels. Each movie also names its recompute share and current
reduction, independently of whether it displays all reductions or only the answer.

## Analysis outputs

`analyze` writes:

- `analysis.json`: size bins/quantiles, display filters, per-reduction totals,
  final pressure, and explicit answer references.
- `cuts.csv`: one row per applied cut, selected alias and semantic name, trigger,
  pressure before/after, peak, excess, score, and decisive comparison field.
- `decisions.jsonl`: selected cut and up to three alternatives, complete named
  scores, their first distinguishing field, and top-three boundary measurements.
- `aliases.json`: every alias, all shared members, semantic labels, reference
  positions, sizes, and hide reasons. `tasks.json`: selected task order and
  recorded profiled/plan durations, task workspace, and semantic labels.
- `reduction_NNN_metrics.npy`: columns `cut_count`, `peak_pressure_bytes`,
  `max_excess_bytes`, with excess clipped at zero.
- `pressure_roles.npz`: role names and role-by-boundary byte arrays for each
  reduction's seed and final reducer state, plus `resident_reservations` by role
  to add for the total view, a boundary `workspace` array, and
  `budget_excluding_resident_bytes`. These seed/final arrays retain raw reducer
  values. `analysis.json` reports the total planning budget and whether roles
  could be attributed; all intermediate compositions are verified as well.

## Trace completed quickstart winners with Nsight

Capture separate Nsight Systems reports of the same
already executed winners. Each fresh worker loads the exact portable Plan from
`plans/`, initializes with the factory's bounded loader and runs warmup/captured
updates. It does not build, measure, calibrate or replan. The complete archive
identity must agree with both the planning index and the completed result;
memory admission remains unchanged. Each rank retains an `exact-plan.json`.
Profiler timings are not throughput measurements.

```sh
python -m tools.pressurefit_trace.profile_winners RUN/quickstart \
  --output RUN/nsys --warmup-steps 3 --capture-steps 3 \
  --gpu-metrics-frequency 10000 --nsys /path/to/nsys
```

`--budget-gib` limits capture to one executed winner. Each attempt retains its
command, logs, `.nsys-rep`, SQLite export and validation. Validation requires kernel
records on every GPU, device metrics, semantic task/transfer NVTX and step ranges
from every rank. Resume checks identities and hashes before skipping a capture.
GPU metrics require permission on the allocated node.
Only GPUs selected through `CUDA_VISIBLE_DEVICES` are sampled. NCCL API tracing
is enabled when advertised by the installed Nsight version; otherwise collective
GPU kernels and runtime-transfer ranges remain in the GPU/NVTX trace.
Worker count comes from the saved Program; optional `--world-size` checks it.
The saved request's full logical rank mapping is retained. Captures initialize
the model anew rather than loading the benchmark's final mutable tensors.
Capture boundaries coordinate over a CPU/Gloo group. The command disables
Nsight's synchronous flush inside `cudaProfilerStop`; Nsight still flushes at
collection end. This avoids a documented multi-context profiler-stop hang
without changing the captured schedule or excluding GPU metrics. Validation
also requires every requested captured step from every rank.

For CPU planning with other memory budgets or communication rates, load a
`ProfiledProgram`, supply `MemoryBudgets` and `CommunicationInfo`, then call
`plan_program` or `sweep_programs`. Use `write_report` for simulated timelines.
The saved-profile `dump` / `plan` / `trace` workflow above also supports host-copy
bandwidth overrides for detailed PressureFit analysis.

## Source layout

- `collect/`: saved-profile inputs, CPU planning, decision capture and export.
- `data.py`, `pressure.py`, `explain.py`: replay, memory accounting and cut decisions.
- `profile_winners/`: saved-Plan Nsight execution and evidence checks.
- `video/`: every video renderer, command, storyboard, encoder and gallery.

The binary reader lives in `shadowspill.diagnostics.pressurefit_trace`.
Repository tests cover capture and planner evidence; video rendering is checked
manually when changed.

# Verified symmetric planning

## Interface and behavior

The production opt-in is `Distributed(group, symmetric_planning=True)`.
The default stays false. Quickstart accepts `--symmetric-planning`,
`--no-symmetric-planning`, and the Python `symmetric_planning` override.
The prepared Qwen EP8 experiment enables the mode, with the negative flag
available for comparison.

All ranks capture/profile GPU tasks. The mode verifies equal logical task/object
structure, sizes, lifetimes, mutations, aliases, physical allocation traces,
budgets, alignment, scratch reserve, and starting/final residency. Explicit device
identifiers and compiled ABI digests remain local. Sharing changes CPU planning
only; no executable, tensor, allocator binding, or physical layout is imported.

Equivalent problems use the slowest per-profile duration, slowest calibrated
transfer bandwidths, and highest transfer latencies. Original profile artifacts
remain intact. Every rank admits the resulting schedule using normal local
physical admission. Requirements that differ fall back to independent searches.
Real execution has no added barriers.

A sweep distributes complete ordering/budget chains, preserving the smaller
budget's incumbent. Identical lowered orderings reuse one answer. A direct
training plan distributes resolution candidates; a forward-only plan searches
on one rank. CPU parallelism is bounded by the number of independent problems.

## Implementation layout

The October 8 review split the two long coordination routines into named steps
and removed the circular dependency between the planner and sweep modules.
The public API and collective sequence are unchanged.

| Private production module | Responsibility | Lines | Longest function |
|---|---|---:|---:|
| `_symmetry.py` | Verify requirements, conservative costs, receive against local contract | 247 | 75 |
| `_plan_exchange.py` | Portable problem/plan exchange and retained-plan records | 139 | 39 |
| `_selection.py` | Candidate ownership, outcomes, common choice, result metadata | 385 | 75 |
| `_sweep.py` | Ordering/budget chains, incumbent reuse, local admissions | 232 | 47 |
| `_search.py` | Connect ordinary planner operations to distributed preparation | 190 | 103 |

Counts include signatures, comments, and blank lines. `choose` decreased from
182 to 75 lines; `prepare_geometry` decreased from 168 to 39. The longest
remaining routine is `DistributedPlanner.plan`, whose 103 lines include its
19-line local candidate callback. These are cohesive modules, without new public
interfaces or tiny forwarding files. The ordinary neutral planner continues to
own search algorithms, simulation, and physical admission.

The module dependency chain is acyclic:
`_search -> _sweep -> _selection -> _plan_exchange -> _symmetry`.
Some modules also directly use a later module in that chain. The sweep's private
state holds ownership and results for one geometry; it does not persist runtime
tensors or add synchronization to real execution. Source metrics and dependency
checks are recorded in `evidence/modularity.json`.

## Validation

CPU tests use two real Gloo processes and verify search ownership, conservative
timings, local compiled bindings, physical admission, asymmetric fallback,
failure propagation, duplicate orderings, retained resolutions, and warm-cache
resume. A separate case exercises runtime-shared residency through the existing
admission path. Broader planner, step-search, quickstart, and plotting regression
results are in `logs/symmetric_regression.log`.

Fatnode GPU validation uses device UUIDs:

- `GPU-14013517-e066-7ce0-1d72-fb10083cd905` (device minor 0).
- `GPU-af041483-6cec-9cb4-0453-5d213e479f2e` (device minor 2).

Only their device files plus CUDA control/UVM are exposed in a rootless Podman
container. This excludes the GPUs reporting driver errors from NCCL discovery;
`CUDA_VISIBLE_DEVICES` alone was not relied upon. A bounded NCCL all-reduce
passed before training. Runs are visible in fatnode's `codex:0.0` pane.

| Check | Result |
|---|---|
| FP32 DP2, automatic save/recompute choice, parameter metrics | Passed, three updates against combined-data CPU oracle |
| FP16 DP2, forced recompute, sharded FP32 masters/optimizer, FP32 gradients | Passed, including startup diagnostics |
| Master/compute checkpoint replay, fresh meta-model restore | Passed in both precision cases |
| Forward-only evaluation with symmetric admission | Passed in both precision cases |
| One-stage model, equivalent depth/breadth orderings | Passed; five ordering labels reuse two distinct geometry problems |
| Multi-stage model, two microbatch geometries and five orderings | Passed; rank 0 searched three, rank 1 searched two |
| Trained parameters after searched-plan adoption | Maximum absolute error 5.96e-8 versus CPU oracle |

These checks explicitly assert that planning used symmetric mode, rather than
passing through its fallback. They validate correctness and ownership, not
large-model search speedup. Qwen EP8 testing remains pending allocation.

Remote evidence is under:
`/data/as1669/shadowspill/docs/internal/plans/qwen_moe_ep8_1007/`:

- `logs/fatnode/`: console logs including NCCL and every test stage.
- `evidence/fatnode/{auto,recompute,sweep-whole,sweep-auto}/`: per-rank results,
  search reports, local artifact stores, checkpoints, and startup timelines.
- `scripts/fatnode/`: exact container, health, and search reproductions.

Small result records and console logs are copied to the same plan folder on
Della; compilation caches stay on fatnode. Source hashes are recorded in
`evidence/fatnode_source_manifest.json`. No commits were made.
Repeat direct-training checks use timestamped output directories to avoid
overwriting checkpoints. The final logs identify each exact destination.

## Issues found during validation

1. **Duplicate ordering results:** a one-stage model can lower several requested
   depth/breadth settings to an identical planning problem. Initial work sharing
   rejected the duplicate result. Fixed by assigning each unique problem once,
   retaining all public ordering points, and charging search time once.
2. **Retained alternatives on cache hit:** `PlanStore` previously loaded only the
   winner. This lost alternatives when distributing a resumed result to a rank
   that had not received the earlier artifacts. `keep_resolutions=True` now
   restores the retained alternatives, with identity and schedule validation.
   A completion manifest also makes interrupted alternative writes a cache miss,
   so a resumed search rebuilds them rather than accepting incomplete evidence.
3. **Admission accounting:** received plans now use the ordinary physical
   admission routine instead of duplicating its effective-capacity rules. This
   also covers runtime-shared objects and records actual admission wall time.
4. **Separate staged capture issue:** the tiny model with an integer counter
   mutated before its linear layers failed automatic stage capture with
   `terminal objective loss is not differentiable`. The same model passes when
   captured as a whole stage. The control run reproduced the same error with
   `symmetric_planning=False`; its log is `logs/fatnode/capture_control.log`.
   This occurs before CPU planning, in unchanged capture code. The multi-stage search
   test uses the same linear computation without the unrelated counter mutation.
   This capture issue remains separate from the symmetric search implementation.

Final validation: the broader CPU regression passed 209 tests (one deselected).
After the cache-completion change, all 28 targeted store/symmetry tests passed.
The final GPU reruns passed both precisions, checkpoint policies, and evaluation;
FP16 maximum parameter error was 1.46e-4 under the existing FP16 tolerance.

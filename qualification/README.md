# Qualification

Run the default gates serially:

```bash
python -m qualification.gates
```

| Gate | What it checks | Details |
| --- | --- | --- |
| `suite` | Python tests and C/device canaries, including fresh-worker cases | [Repository validation](../docs/development/repository.md) |
| `numerical` | Five updates against compiled PyTorch, checkpoint replay and memory limits | [Numerical protocol](numerical/README.md) |
| `performance` | Full-model throughput, simulator accuracy and memory limits | [Performance protocol](performance/README.md) |

The numerical and performance runners use **build → profile → CPU plan → fresh
execution worker**. All phases use the public ShadowSpill APIs. Gates run one at
a time so their GPU measurements do not interfere. Each gate streams to the
terminal and its own log.

These numerical and performance gates exercise the public **phase APIs**, including
`Executor`. They retain the pre-refactor measurement protocol and independent
compiled-PyTorch numerical authority. They do not submit `TrainingClient.step()`
or `forward_backward()` requests. Public-client integration tests live with the client, engine and workloads
packages, and are discovered by the suite gate. Tests marked `slow` need explicit
selection as described below.

## Selecting and configuring gates

```bash
python -m qualification.gates suite numerical --run my_check --config gates.json
```

Gate order is fixed regardless of command-line order. `--keep-going` continues
remaining matrix cells after a failure; `--continue-after-failure` continues later
gates. Neither option changes a failing result into a pass.

A JSON configuration forwards each list of arguments to that gate unchanged:

```json
{
  "suite": ["--test-dtype", "bfloat16"],
  "numerical": ["--reference-dir", "/local/references/approximately_1b"],
  "performance": []
}
```

Missing sections use the gate's defaults; unknown sections are errors. Use each
matrix's `--help` for its available options. The default run name includes the
Git revision and time, with `dirty` when tracked files have uncommitted changes.
The launcher rejects stale C builds; rebuild before qualification. The explicit
`--allow-stale-library` override is for deliberately measuring an older build.

## Artifacts and references

Results live under `qualification/results/`:

- `gates_<run>/`: per-gate logs, incremental `suite.events.jsonl` and the terminal summary.
- `numerical_<run>/`, `performance_<run>/`: matrix summaries and per-cell results.
- Each cell's `.phases/`: Program, Profiles, ProfiledProgram, CommunicationInfo,
  Plan, settings and phase timings.

The suite event log records selected/deselected tests, each test's start and
setup/call/teardown results, failure details and session completion. Records are
flushed immediately so a process crash does not discard earlier failures or the
identity of the active test. Sessions have distinct IDs and append to the file;
an absent `session_finish` means the session is incomplete, not a pass. This
does not make records durable against machine or storage failure. Ordinary
pytest runs can opt in with `-p qualification.gates.suite_events --suite-events
PATH`. Numerical and performance workers do not load this plugin.

Every matrix gate starts with fresh phase artifacts and empty Inductor/Triton
caches. Compiler caches are disposable; phase archives remain available for
inspection and replanning. A cell's exact inputs (`inputs.pt`) and streamed
checkpoint (`checkpoint.ssp`) are each the size of its model, so they are
removed once its verdict is recorded; `--retain-execution-files` keeps them for
an execution-only rerun. A failed cell keeps them. Tests use isolated
temporary planning stores.
Fresh-process tests deselected by the shared pytest process are run by CTest;
capability skips are reported separately.

Tests marked `slow` are end-to-end recipe and quickstart runs of minutes each;
the default suite deselects them. `python -m pytest -m slow` runs the tier
alone, and a gate config with `"suite": ["-m", "not fresh_process"]` includes it
in the suite gate.

Numerical references are reused from
`qualification/results/references/approximately_1b` (`pre_sm80` on older GPUs),
or the explicit `--reference-dir`. Keep `reference.pt`, `inputs.pt` and the
`kernel-selection.json` sidecar together on local SSD. Numerical warm-up records
external kernel configurations; both comparison arms use those fixed choices
without autotuning. Missing references are generated on first use. References
from before this selection contract require regeneration. Use
`--regenerate-reference` only for an intentional baseline replacement, then remove
that option from subsequent runs. A new reference checks planned/reference
agreement but cannot establish agreement with the previous baseline.

For a comparison across the trainer refactor, point `--reference-dir` at the
retained pre-refactor reference set, verify all three files exist for every
selected cell, and omit `--regenerate-reference`. Request identities, exact inputs,
kernel choices and target compatibility must still match. Keep the old result
directory and write the new run separately; source moves do not require replacing
the numerical authority.

The performance gate retains its recorded RTX 5090 floors and 5% allowance.
It applies them only to matching hardware, geometry, precision and memory/spill
settings. Archived run JSON also retains per-cycle samples and phase timings for
matched before/after comparisons. Those timings cover saved-Plan execution;
client upload, packing and response publication need separate service measurements.
An unrelated configuration is reported as not applicable, not as a throughput
regression pass.

## Hardware and optional checks

See [hardware and precision](hardware.md) for dtype defaults, budgets and overrides.
The launcher uses all CPUs available by affinity and sets OpenMP/MKL/OpenBLAS
thread counts accordingly. `--cpu-threads 8` overrides that count. Host load and
GPU tenants are recorded at gate start and on failure.

The default matrix uses pinned-host spill. Optional `numerical_ssd`, `remote` and
`remote_perf` use the same saved phases: their pool configuration is passed to
communication calibration and Executor, while model build/task measurement stay
backend-independent. Remote gates skip when no peer is configured. See
[SSD qualification](numerical/ssd.md), [remote correctness](remote/README.md) and
[remote throughput](remote_perf/README.md).

[The nondeterminism probe](nondeterminism/README.md) locates the first divergent
module. For measured-versus-simulated step breakdowns:

```bash
python -m qualification.gap_report qualification/results/performance_my_check
```

## Source layout

| Directory | Responsibility |
| --- | --- |
| `gates/` | CLI/configuration, serial process execution, environment and summaries |
| `pipeline/` | Shared phase launching, saved inputs, bounded loading/export and observations |
| `numerical/`, `performance/` | Case definitions, phase workers, verdicts and matrix launchers |
| `numerical/matrix/`, `performance/matrix/` | Argument parsing, case commands/execution, failure evidence and reports |
| `remote/`, `remote_perf/` | Optional remote matrix selection |
| `nondeterminism/` | The bitwise reproducibility probe |
| `tests/qualification/` | Runner and acceptance-rule tests |

Model definitions live in `packages/workloads/src/shadowspill_workloads/`; runtime and planning behavior remain in
`packages/engine/src/shadowspill/`. There is one implementation per runner, without forwarding
packages or a second preparation path inside execution.

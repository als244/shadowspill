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

- `gates_<run>/`: per-gate logs and the terminal summary.
- `numerical_<run>/`, `performance_<run>/`: matrix summaries and per-cell results.
- Each cell's `.phases/`: Program, Profiles, ProfiledProgram, CommunicationInfo,
  Plan, exact inputs, settings and phase timings.

Every matrix gate starts with fresh phase artifacts and empty Inductor/Triton
caches. Compiler caches are disposable; phase archives remain available for
inspection and execution-only reruns. Tests use isolated temporary planning
stores. Fresh-process tests deselected by the shared pytest process are run by
CTest; capability skips are reported separately.

Numerical references are reused from
`qualification/results/references/approximately_1b` (`pre_sm80` on older GPUs),
or the explicit `--reference-dir`. Keep `reference.pt` and its `inputs.pt` sidecar
together on local SSD. Missing references are generated on first use. Use
`--regenerate-reference` only for an intentional baseline replacement, then remove
that option from subsequent runs. A new reference checks planned/reference
agreement but cannot establish agreement with the previous baseline.

## Hardware and optional checks

See [hardware and precision](hardware.md) for dtype defaults, budgets and overrides.
The launcher uses all CPUs available by affinity and sets OpenMP/MKL/OpenBLAS
thread counts accordingly. `--cpu-threads 8` overrides that count. Host load and
GPU tenants are recorded at gate start and on failure.

The default matrix uses pinned-host spill. Optional `numerical_ssd`, `remote` and
`remote_perf` use the same saved phases: their pool configuration is passed to
communication calibration and Executor, while model build/task measurement stay
backend-independent. Remote gates skip when no peer is configured. Full-size
SSD/remote matrices still require target validation after this migration; core
Executor backend tests and configuration checks are separate evidence. See
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
| `remote/`, `remote_perf/` | Optional remote matrix selection |
| `tests/qualification/` | Runner and acceptance-rule tests |

Model definitions live in `workloads/`; runtime and planning behavior remain in
`src/shadowspill/`. There is one implementation per runner, without forwarding
packages or a second preparation path inside execution.

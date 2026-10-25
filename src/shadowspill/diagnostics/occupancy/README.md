# Occupancy diagnostics

The public imports and `python -m shadowspill.diagnostics.occupancy` command
remain the same. The implementation has these responsibilities:

| Module | Responsibility |
| --- | --- |
| `intervals.py`, `facts.py` | Tensor roles, selected tasks, byte intervals and aggregation |
| `clock.py` | Simulated/device timestamps and interval unions |
| `spill.py`, `execution.py` | Spill residency and physical execution leases |
| `plan.py`, `summary.py` | Assemble a plan view and compute its summary |
| `page.py`, `html.py`, `occupancy_page.html` | Numeric page data, HTML tables and the interactive timeline |
| `stored.py` | Discover old store records and their certificates |
| `run/` | Trace matching and the per-budget report tree |
| `cli.py` | Argument parsing and command orchestration |

Spill residency follows the simulator's transfer/release events. Execution
occupancy follows the admitted layout's leases, including storage handoffs.
Host-copy payload and full execution-object bytes remain separate; collective
time is not counted as host-copy traffic.

These diagnostics still support existing ArtifactStore runs. New phase artifacts
use the public `write_report` API in `shadowspill.reporting`; store discovery must
not become a requirement for reporting a saved Plan.

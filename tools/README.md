# Developer tools

Run these modules from the repository root. They are outside the installed packages.

| Task | Tool |
| --- | --- |
| Collect GPU measurements and compare planner behavior | [Planning](planning/README.md) |
| Measure an existing executable Plan | [Plan measurement](measure_plan/README.md) |
| Inspect a compiled stage with Nsight | [Stage traces](stage_traces/README.md) |
| Analyze planner decisions and render traces | [PressureFit traces](pressurefit_trace/README.md) |
| Prepare tokenized training data | [Dataset tools](datasets/README.md) |

`check_naming.py` checks source conventions; `sanitizers/` supports instrumented builds.
[Qualification](../qualification/README.md) owns correctness and regression gates.
For training, use the [client demos](../examples/demos/README.md). For hypothetical
memory or bandwidth changes, use [offline sweeps](../docs/python/advanced/offline-sweeps.md).

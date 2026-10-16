# Source-tree tools

This package contains repository support files outside ShadowSpill's wheel.

- `check_naming.py` enforces naming and component boundaries.
- `sanitizers/` contains tool-specific support files.
- [`pressurefit_trace/`](pressurefit_trace/README.md) captures planner traces,
  analyzes each decision, and renders residency-cut trajectories as PNG or MP4.

[Qualification](../../qualification/README.md) owns gate orchestration and
acceptance checks. Runtime diagnostics and occupancy rendering live in the
installed `shadowspill.diagnostics` package; see the
[occupancy guide](../../docs/python/occupancy.md).

The development install and test configuration put `src/` on Python's import
path. Product behavior remains in `src/shadowspill/`.

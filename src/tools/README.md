# Source-tree tools

This package contains repository support files outside ShadowSpill's wheel.

- `check_naming.py` enforces naming and component boundaries.
- `sanitizers/` contains tool-specific support files.
- [`pressurefit_trace/`](pressurefit_trace/README.md) contains capture and
  analysis tools and all three video renderers: residency cuts, action emission
  plus placement, and simulated compute/fetch/evict execution. Start with that
  guide for inputs, examples, CLI arguments, and the code map.
- `plan_video/` forwards the earlier plan-video CLI to
  `pressurefit_trace/plan/`; it contains no rendering or analysis implementation.

[Qualification](../../qualification/README.md) owns gate orchestration and
acceptance checks. Runtime diagnostics and occupancy rendering live in the
installed `shadowspill.diagnostics` package; see the
[occupancy guide](../../docs/python/occupancy.md).

The development install and test configuration put `src/` on Python's import
path. Product behavior remains in `src/shadowspill/`.

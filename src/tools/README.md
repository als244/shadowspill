# Source-tree tools

This package contains repository support files outside ShadowSpill's wheel.

- `check_naming.py` enforces naming and component boundaries.
- `sanitizers/` contains tool-specific support files.

[Qualification](../../qualification/README.md) owns gate orchestration and
acceptance checks. Runtime diagnostics and occupancy rendering live in the
installed `shadowspill.diagnostics` package; see the
[occupancy guide](../../docs/python/occupancy.md).

The development install and test configuration put `src/` on Python's import
path. Product behavior remains in `src/shadowspill/`.

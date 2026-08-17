# Source-tree tools

This package contains reusable repository tooling that is intentionally not
part of ShadowSpill's wheel. Qualification launchers, NSYS analysis, naming
checks, and sanitizer support call these modules instead of duplicating product
logic.

- `qualification/` implements reusable acceptance-run orchestration.
- `diagnostics/` inspects serialized step evidence and NSYS exports.
- `check_naming.py` enforces provider and vocabulary boundaries.
- `sanitizers/` contains tool-specific support files.

The source tree is added to Python's import path by the development install and
test configuration. Product behavior remains in `src/shadowspill/`.

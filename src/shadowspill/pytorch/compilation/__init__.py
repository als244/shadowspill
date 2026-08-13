"""Structural task compilation, physical layout, and isolated profiling.

Compilation modules intentionally remain explicit imports.  In particular,
runtime allocation telemetry depends on the lightweight profiling records,
while the compiler itself consumes that telemetry.  Keeping this package
initializer side-effect free makes that dependency direction visible and
prevents import-order cycles.
"""

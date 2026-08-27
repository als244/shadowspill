"""The top-level package exposes one name and nothing else.

Whether importing it drags a framework in is checked in
`test_neutral_imports.py`, which runs each case in a fresh interpreter. The
same check cannot be made here: once any earlier test in this process has
imported torch, a `sys.modules` diff can no longer attribute it.
"""

from __future__ import annotations

import importlib


def test_top_level_public_surface_is_explicit() -> None:
    package = importlib.import_module("shadowspill")
    assert package.__all__ == ["__version__"]
    assert package.__version__

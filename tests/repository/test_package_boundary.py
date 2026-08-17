from __future__ import annotations

import importlib
import sys


def test_top_level_import_is_minimal() -> None:
    before = set(sys.modules)
    package = importlib.import_module("shadowspill")
    imported = set(sys.modules) - before

    assert package.__version__
    assert "torch" not in imported
    assert "mlops" not in imported


def test_top_level_public_surface_is_explicit() -> None:
    package = importlib.import_module("shadowspill")
    assert package.__all__ == ["__version__"]

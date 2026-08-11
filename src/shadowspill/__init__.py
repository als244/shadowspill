"""Framework-neutral memory planning and transparent execution support."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("shadowspill")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.1.0.dev0"

__all__ = ["__version__"]

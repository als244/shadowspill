"""Deterministic discovery of ShadowSpill's installed compiled libraries."""

from __future__ import annotations

import sys
import sysconfig
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent


def library_candidates(
    filename: str,
    *,
    package_root: Path = _PACKAGE_ROOT,
) -> tuple[Path, ...]:
    """Return package and configured editable-build locations in precedence order."""

    packaged = package_root / "lib" / filename
    project = _editable_project_root(package_root)
    if project is None:
        return (packaged,)

    build_root = project / "build"
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return (
        packaged,
        build_root / "dev" / filename,
        build_root / f"{python_tag}-{python_tag}-{platform_tag}" / filename,
    )


def resolve_library(
    filename: str,
) -> Path | None:
    """Resolve a package-owned or configured editable-build artifact."""

    for candidate in library_candidates(filename):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _editable_project_root(package_root: Path) -> Path | None:
    source_root = package_root.parent
    if source_root.name != "src":
        return None
    project = source_root.parent
    if not (project / "pyproject.toml").is_file() or not (
        project / "CMakeLists.txt"
    ).is_file():
        return None
    return project


__all__ = ["library_candidates", "resolve_library"]

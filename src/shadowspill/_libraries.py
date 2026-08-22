"""Deterministic discovery of ShadowSpill's installed compiled libraries."""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path
from typing import Final

_PACKAGE_ROOT = Path(__file__).resolve().parent

#: Names a directory to search before the installed and editable locations.
#: Nothing outside those two is searched unless this says so: a stale build
#: left in a directory the loader happened to know about is silently wrong,
#: and slow or subtly incompatible in ways that look like the code's fault.
LIBRARY_DIRECTORY_ENVIRONMENT: Final = "SHADOWSPILL_LIBRARY_DIRECTORY"


def library_candidates(
    filename: str,
    *,
    package_root: Path = _PACKAGE_ROOT,
) -> tuple[Path, ...]:
    """Return the explicit, installed, and editable locations in precedence order."""

    candidates: list[Path] = []
    override = os.environ.get(LIBRARY_DIRECTORY_ENVIRONMENT)
    if override:
        candidates.append(Path(override).expanduser() / filename)
    candidates.append(package_root / "lib" / filename)
    project = _editable_project_root(package_root)
    if project is not None:
        python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        wheel_tag = f"{python_tag}-{python_tag}-{platform_tag}"
        candidates.append(project / "build" / wheel_tag / filename)
    return tuple(candidates)


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
    if (
        not (project / "pyproject.toml").is_file()
        or not (project / "CMakeLists.txt").is_file()
    ):
        return None
    return project


__all__ = ["LIBRARY_DIRECTORY_ENVIRONMENT", "library_candidates", "resolve_library"]

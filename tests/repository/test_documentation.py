"""Keep public documentation aligned with source-level API boundaries."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
PYTHON_API = DOCS / "python" / "api"

_MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
_C_FUNCTION = re.compile(r"\b(shadowspill_[A-Za-z0-9_]+)\s*\(")

_PUBLIC_PYTHON_MODULES = (
    ROOT / "src" / "shadowspill" / "memory.py",
    ROOT / "src" / "shadowspill" / "ir" / "__init__.py",
    ROOT / "src" / "shadowspill" / "planner" / "__init__.py",
    ROOT / "src" / "shadowspill" / "simulator" / "__init__.py",
    ROOT / "src" / "shadowspill" / "runtime" / "__init__.py",
    ROOT / "src" / "shadowspill" / "pytorch" / "__init__.py",
)

_PUBLIC_C_HEADERS = (
    ROOT / "csrc" / "runtime" / "include" / "shadowspill" / "runtime.h",
    ROOT
    / "csrc"
    / "runtime"
    / "include"
    / "shadowspill"
    / "admission_replay.h",
    ROOT / "csrc" / "planner" / "include" / "shadowspill" / "planner.h",
    ROOT / "csrc" / "simulator" / "include" / "shadowspill" / "simulator.h",
    ROOT
    / "csrc"
    / "pytorch_adapter"
    / "include"
    / "shadowspill"
    / "pytorch_adapter.h",
)


def _markdown_files() -> tuple[Path, ...]:
    ignored = {
        ".cache",
        ".git",
        ".hypothesis",
        ".mypy_cache",
        ".venv",
        "build",
        "datasets",
        "planning_caches",
        "results",
    }
    return tuple(
        path
        for path in ROOT.rglob("*.md")
        if not any(part in ignored for part in path.parts)
        and not path.is_relative_to(DOCS / "internal")
    )


def _local_link_target(document: Path, raw: str) -> Path | None:
    value = raw.strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]
    destination = value.split(maxsplit=1)[0]
    if (
        not destination
        or destination.startswith("#")
        or "://" in destination
        or destination.startswith(("mailto:", "plugin:"))
    ):
        return None
    path = unquote(destination.split("#", 1)[0])
    if not path:
        return None
    return (document.parent / path).resolve()


def _all_exports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise AssertionError(f"{path}: __all__ must be a literal string list")
        return tuple(value)
    raise AssertionError(f"{path}: public module has no literal __all__")


def _documented_symbol(reference: str, name: str) -> bool:
    return re.search(rf"`{re.escape(name)}(?:`|\()", reference) is not None


def test_all_local_markdown_links_resolve() -> None:
    missing: list[str] = []
    for document in _markdown_files():
        for raw in _MARKDOWN_LINK.findall(document.read_text()):
            target = _local_link_target(document, raw)
            if target is not None and not target.exists():
                missing.append(
                    f"{document.relative_to(ROOT)} -> {target.relative_to(ROOT)}"
                )
    assert not missing, "missing local Markdown links:\n" + "\n".join(missing)


def test_documentation_index_exposes_language_boundaries() -> None:
    index = (DOCS / "README.md").read_text()
    for target in (
        "architecture/overview.md",
        "python/README.md",
        "c/README.md",
        "development/README.md",
        "investigations/README.md",
    ):
        assert f"]({target})" in index


def test_every_public_python_export_appears_in_api_reference() -> None:
    reference = "\n".join(path.read_text() for path in PYTHON_API.glob("*.md"))
    missing = {
        path.relative_to(ROOT).as_posix(): [
            name
            for name in _all_exports(path)
            if not _documented_symbol(reference, name)
        ]
        for path in _PUBLIC_PYTHON_MODULES
    }
    missing = {path: names for path, names in missing.items() if names}
    assert not missing, f"undocumented public Python exports: {missing}"


def test_every_public_c_function_appears_in_c_reference() -> None:
    reference = "\n".join(path.read_text() for path in (DOCS / "c").glob("*.md"))
    missing = {
        path.relative_to(ROOT).as_posix(): sorted(
            name
            for name in set(_C_FUNCTION.findall(path.read_text()))
            if not _documented_symbol(reference, name)
        )
        for path in _PUBLIC_C_HEADERS
    }
    missing = {path: names for path, names in missing.items() if names}
    assert not missing, f"undocumented public C functions: {missing}"


def test_superseded_public_documentation_is_removed() -> None:
    for name in (
        "architecture.md",
        "ir.md",
        "lowering_contract.md",
        "memory-budget-semantics.md",
        "planner.md",
        "planning-cache.md",
        "pytorch-allocator.md",
        "pytorch-frontend.md",
        "runtime.md",
        "simulator.md",
    ):
        assert not (DOCS / name).exists()

"""Every declared command-line flag must be read by the module declaring it.

`argparse` turns `--artifact-store-dir` into `arguments.artifact_store_dir`.
Rename one half and the other keeps working: the flag still parses, the
attribute is still produced, and nothing reads it. Nothing fails until the
command actually runs, which for the qualification drivers means a gate --
minutes of GPU work to learn about a typo.

That is not hypothetical: renaming `planning_cachedir` to
`artifact_store_dir` left `add_argument("--planning-cachedir")` behind and
broke three entry points while the whole suite stayed green.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _declared_flags(tree: ast.AST) -> set[str]:
    """The attribute names `argparse` will produce for each long option."""

    names: set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        explicit = next(
            (
                k.value.value
                for k in node.keywords
                if k.arg == "dest" and isinstance(k.value, ast.Constant)
            ),
            None,
        )
        if explicit:
            names.add(str(explicit))
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and str(arg.value).startswith("--"):
                names.add(str(arg.value)[2:].replace("-", "_"))
    return names


def _cli_modules() -> list[Path]:
    """Every module that declares a long option, and so has something to check.

    A module whose parser takes only positional arguments declares no
    attribute that could drift from its flag, so it is not a case here.
    Leaving it in as a case that skips itself reports a skip on every run,
    which reads like something was not checked rather than like there was
    nothing to check.
    """

    tracked = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    # This file names the call it looks for, so it matches its own search.
    here = Path(__file__).resolve()
    candidates = [
        ROOT / f
        for f in tracked
        if (ROOT / f).resolve() != here and "add_argument(" in (ROOT / f).read_text()
    ]
    return [
        path
        for path in candidates
        if _declared_flags(ast.parse(path.read_text(), filename=str(path)))
    ]


@pytest.mark.parametrize("path", _cli_modules(), ids=lambda p: str(p.relative_to(ROOT)))
def test_declared_flags_are_read(path: Path) -> None:
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    declared = _declared_flags(tree)

    # Any attribute access of that name counts: the parsed namespace is passed
    # around under several names (`arguments`, `args`, `options`).
    read = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    unread = sorted(name for name in declared if name not in read)
    assert not unread, (
        f"{path.relative_to(ROOT)} declares options nothing reads: {unread}. "
        "A renamed flag and a renamed attribute have to move together."
    )

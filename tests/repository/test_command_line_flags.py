"""Every declared command-line flag must be read by the module declaring it.

`argparse` turns `--build-store-mode` into `arguments.build_store_mode`.
Rename one half and the other keeps working: the flag still parses, the
attribute is still produced, and nothing reads it. Nothing fails until the
command actually runs, which for the qualification drivers means a gate --
minutes of GPU work to learn about a typo.

That is not hypothetical: renaming `planning_cachedir` to
`artifact_store` left `add_argument("--planning-cachedir")` behind and
broke three entry points while the whole suite stayed green.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.repository.files import repository_files

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


def _repository_python_files() -> list[Path]:
    """The repository's Python files: tracked, and untracked but not ignored.

    Untracked counts because a module moved into a package is untracked until
    the move is committed. Ignored does not: a machine's local scripts are no
    part of the repository, and a check that read them would pass on one
    machine and fail on another.
    """

    return list(repository_files("*.py"))


def _cli_modules() -> list[Path]:
    """Every module that declares a long option, and so has something to check.

    A module whose parser takes only positional arguments declares no
    attribute that could drift from its flag, so it is not a case here.
    Leaving it in as a case that skips itself reports a skip on every run,
    which reads like something was not checked rather than like there was
    nothing to check.
    """

    # This file names the call it looks for, so it matches its own search.
    here = Path(__file__).resolve()
    candidates = [
        path
        for path in _repository_python_files()
        if path.resolve() != here and "add_argument(" in path.read_text()
    ]
    return [
        path
        for path in candidates
        if _declared_flags(ast.parse(path.read_text(), filename=str(path)))
    ]


def _readers(path: Path) -> list[Path]:
    """Where a flag this module declares may be read.

    A one-file command reads its own namespace. A command that is a package
    declares its flags in the entry and hands the namespace to the phases
    beside it, so the package is the unit: a flag nothing in it reads is still
    a flag nothing reads, which is what this checks.
    """

    recipe_packages = {
        ROOT / "workloads/recipes/text/train_llama3.py": ("llama3",),
        ROOT / "workloads/recipes/text/train_qwen3_distributed.py": ("qwen3",),
        ROOT / "workloads/recipes/text/train_qwen35_lora.py": ("qwen35_lora", "phases"),
        ROOT / "workloads/recipes/text/glm53_flash_lora/arguments.py": (
            ".",
            "../phases",
        ),
        ROOT / "workloads/recipes/distributed_training.py": ("regression",),
    }
    if path in recipe_packages:
        return [
            path,
            *(
                reader
                for package in recipe_packages[path]
                for reader in sorted((path.parent / package).glob("*.py"))
            ),
        ]
    if path.parent == ROOT / "baselines/moe_layer":
        return list(path.parent.rglob("*.py"))
    package_commands = {
        ROOT / "qualification/numerical/run.py",
        ROOT / "qualification/performance/run.py",
    }
    if (
        path.name not in ("__init__.py", "__main__.py", "cli.py")
        and path not in package_commands
    ):
        return [path]
    return sorted(item for item in path.parent.glob("*.py"))


def _names_read(paths: list[Path]) -> set[str]:
    """Any attribute access or string constant, over every reader.

    The parsed namespace is passed around under several names (`arguments`,
    `args`, `options`), so the attribute name is what can be matched.
    """

    names: set[str] = set()
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        names |= {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        names |= {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
    return names


@pytest.mark.parametrize("path", _cli_modules(), ids=lambda p: str(p.relative_to(ROOT)))
def test_declared_flags_are_read(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    declared = _declared_flags(tree)
    read = _names_read(_readers(path))
    unread = sorted(name for name in declared if name not in read)
    assert not unread, (
        f"{path.relative_to(ROOT)} declares options nothing reads: {unread}. "
        "A renamed flag and a renamed attribute have to move together."
    )


def test_no_caller_passes_a_keyword_a_public_entry_point_does_not_accept() -> None:
    """Check imported public phase calls, not similarly named internal helpers."""
    import inspect

    import shadowspill

    names = {
        "build_forward",
        "build_forward_backward",
        "build_step",
        "measure_program",
        "apply_profiles",
        "measure_communication",
        "plan_program",
        "sweep_programs",
        "write_report",
    }
    entry_points = {name: getattr(shadowspill, name) for name in names}

    offences: list[str] = []
    roots = ("src", "tests", "benchmarking", "reference", "workloads", "training")
    for path in _repository_python_files():
        if path.relative_to(ROOT).parts[0] not in roots:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        imported = {}
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "shadowspill":
                imported.update({a.asname or a.name: a.name for a in node.names})
            elif isinstance(node, ast.Import):
                modules.update(
                    a.asname or a.name for a in node.names if a.name == "shadowspill"
                )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = (
                imported.get(node.func.id) if isinstance(node.func, ast.Name) else None
            )
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in modules
            ):
                called = node.func.attr
            function = entry_points.get(called)
            if function is None:
                continue
            accepted = set(inspect.signature(function).parameters)
            for keyword in node.keywords:
                if keyword.arg is not None and keyword.arg not in accepted:
                    offences.append(
                        f"{path.relative_to(ROOT)}:{node.lineno} "
                        f"{called}({keyword.arg}=...)"
                    )
    assert not offences, "calls passing an argument that does not exist:\n" + "\n".join(
        offences
    )


@pytest.mark.parametrize(
    "module",
    (
        "qualification.gates",
        "qualification.numerical.matrix",
        "qualification.performance.matrix",
    ),
)
def test_qualification_entrypoints_start_without_a_device(module: str) -> None:
    """Import the actual CLI package, including wrappers around moved tooling."""
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout

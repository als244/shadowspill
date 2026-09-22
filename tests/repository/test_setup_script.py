"""What `scripts/setup.sh` imports must exist.

The script's last step opens a fresh interpreter and imports ShadowSpill to
prove an install works, which makes it the one caller no test exercises and
no refactor follows: a module it names can be moved, every test stay green,
and the break appear only on a machine installing from scratch. It happened
once, when the adapter's list of required storage operations moved out of a
module the script still imported.

So the names the script imports are parsed out of it and imported here. The
parse is deliberately literal: it reads `from shadowspill... import ...`
lines rather than running the script, because running it would install.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

SETUP = Path(__file__).resolve().parents[2] / "scripts" / "setup.sh"

#: `from <module> import <names>` for any ShadowSpill module, across the
#: heredocs the script feeds to Python. A continuation in parentheses is
#: folded first, so a multi-line import reads as one.
_IMPORT = re.compile(
    r"^from\s+(shadowspill[\w.]*)\s+import\s+(\(?[^\n)]*\)?)", re.MULTILINE
)


def _imports() -> tuple[tuple[str, tuple[str, ...]], ...]:
    text = SETUP.read_text()
    folded = re.sub(r"\(\s*([^)]*?)\s*,?\s*\)", lambda m: m.group(1), text, flags=re.S)
    found = []
    for module, names in _IMPORT.findall(folded):
        parsed = tuple(
            name.strip()
            for name in names.strip("()").split(",")
            if name.strip() and name.strip() != "\\"
        )
        found.append((module, parsed))
    return tuple(found)


def test_the_script_imports_something() -> None:
    """A parse that silently found nothing would pass every case below."""

    assert _imports(), f"no ShadowSpill imports parsed out of {SETUP}"


@pytest.mark.parametrize("module, names", _imports())
def test_every_name_the_script_imports_exists(module: str, names: tuple[str, ...]) -> None:
    imported = importlib.import_module(module)
    missing = [name for name in names if not hasattr(imported, name)]
    assert not missing, f"{SETUP.name} imports {missing} from {module}, which no longer has them"


def test_the_python_the_script_runs_parses() -> None:
    """Each heredoc it feeds to Python must at least be syntactically whole."""

    text = SETUP.read_text()
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY\n", text, flags=re.S)
    assert blocks, "no Python heredocs found in the setup script"
    for index, block in enumerate(blocks):
        ast.parse(block)

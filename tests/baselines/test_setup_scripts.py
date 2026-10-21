"""The setup scripts' embedded Python: every quoted heredoc and every
program the environment's Python runs with ``-c`` compiles, which
``bash -n`` does not check."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = sorted((ROOT / "baselines/scripts").glob("*.sh"))


def programs(path):
    """Each embedded program with where it starts: <<'PY' ... PY heredocs
    and the single-quoted argument of ``"${python_executable}" -c``."""

    text = path.read_text()
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if re.search(r"<<-?'PY'", line):
            end = lines.index("PY", index + 1)
            yield f"{path.name}:{index + 2}", "\n".join(lines[index + 1 : end])
    for match in re.finditer(r"""python_executable}" -c '([^']*)'""", text):
        yield f"{path.name}:{text.count(chr(10), 0, match.start()) + 1}", match.group(1)


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda path: path.name)
def test_embedded_python_compiles(path):
    found = list(programs(path))
    assert found, f"{path.name} embeds no Python"
    for where, source in found:
        compile(source, where, "exec")

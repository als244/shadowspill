"""The API reference builds without a warning and covers each package's exports."""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sphinx", reason="the reference needs the dev extra's Sphinx")
pytest.importorskip("numpydoc", reason="the reference needs the dev extra's numpydoc")

import shadowspill
import shadowspill_client

ROOT = Path(__file__).resolve().parents[2]


def _undocumented(built: Path, section: str, package: str, names) -> list[str]:
    pages = "".join(
        path.read_text(encoding="utf-8")
        for path in (built / "reference" / section).glob("*.html")
    )
    return [name for name in names if f'id="{package}.{name}"' not in pages]


def test_the_reference_builds_and_documents_every_export(tmp_path):
    # numpydoc's validation runs during the build, so a public object that
    # lacks its summary, a parameter or its return value fails here.
    command = [sys.executable, "-m", "sphinx", "-W", "--keep-going", "-q", "-b"]
    built = subprocess.run(
        [*command, "html", str(ROOT / "docs"), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    client = [name for name in shadowspill_client.__all__ if not name.startswith("_")]
    missing = _undocumented(tmp_path, "client", "shadowspill_client", client)
    assert not missing, f"client exports absent from the reference: {missing}"
    missing = _undocumented(tmp_path, "engine", "shadowspill", shadowspill._PUBLIC)
    assert not missing, f"engine exports absent from the reference: {missing}"

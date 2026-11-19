"""The API reference builds without a warning and covers the public client."""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sphinx", reason="the reference needs the dev extra's Sphinx")
pytest.importorskip("numpydoc", reason="the reference needs the dev extra's numpydoc")

import shadowspill_client

ROOT = Path(__file__).resolve().parents[2]


def test_the_reference_builds_and_documents_every_client_export(tmp_path):
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
    pages = "".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "reference" / "client").glob("*.html")
    )
    exports = [name for name in shadowspill_client.__all__ if not name.startswith("_")]
    missing = [
        name for name in exports if f'id="shadowspill_client.{name}"' not in pages
    ]
    assert not missing, f"client exports absent from the reference: {missing}"

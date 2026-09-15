"""A step program filed under its pre-capture identity comes back as it was."""

from __future__ import annotations

from pathlib import Path

import pytest

from shadowspill.step import StepArchive
from shadowspill.store import StorePolicy
from tests.benchmarking._fixtures import _fixture

KEY = "ab" * 32
OTHER = "cd" * 32


def test_a_filed_program_is_read_back_by_its_key(tmp_path: Path) -> None:
    program = _fixture()
    archive = StepArchive(tmp_path)
    assert archive.read(KEY) is None
    archive.write(KEY, program, {"export_bypass_key": "k", "inputs": ["x"]})
    found = archive.read(KEY)
    assert found is not None
    assert found.digest == program.digest
    assert found.to_json() == program.to_json()
    assert archive.read(OTHER) is None


def test_the_manifest_names_the_key_and_the_program(tmp_path: Path) -> None:
    program = _fixture()
    archive = StepArchive(tmp_path)
    archive.write(KEY, program, {"export_bypass_key": "k"})
    manifest = archive.path(KEY) / "manifest.json"
    manifest.write_text(manifest.read_text().replace(KEY, OTHER))
    with pytest.raises(ValueError, match="wrong identity"):
        archive.read(KEY)


def test_policy_decides_reads_and_writes(tmp_path: Path) -> None:
    program = _fixture()
    silent = StepArchive(tmp_path, policy=StorePolicy(write_enabled=False))
    silent.write(KEY, program, {})
    assert not (silent.path(KEY) / "step_program.json").exists()
    StepArchive(tmp_path).write(KEY, program, {})
    blind = StepArchive(tmp_path, policy=StorePolicy(read_enabled=False))
    assert blind.read(KEY) is None
    assert StepArchive(tmp_path).read(KEY) is not None

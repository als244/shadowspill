"""Phase corpora retain measured evidence and are consumable without PyTorch."""

import os
import subprocess
import sys
from dataclasses import replace

import pytest

from benchmarking.planning_eval.source import discover_program_cases
from benchmarking.program_collection.artifacts import (
    load_program,
    save_profiled_program,
)
from benchmarking.program_collection.corpus import ProgramCaseIdentity
from tests.benchmarking._profiled import measured


def test_archive_is_atomic_immutable_and_relocatable(tmp_path):
    program = measured()
    identity = ProgramCaseIdentity("tiny", "pytorch", 64, 32, 2)
    metadata = {"seed": 17}
    first = save_profiled_program(
        tmp_path, identity=identity, program=program, metadata=metadata
    )
    assert (
        save_profiled_program(
            tmp_path, identity=identity, program=program, metadata=metadata
        )
        == first
    )
    with pytest.raises(FileExistsError, match="different evidence"):
        save_profiled_program(
            tmp_path, identity=identity, program=program, metadata={"seed": 18}
        )
    assert not list(first.directory.parent.glob(".collect-*"))
    assert len(discover_program_cases(tmp_path)) == 1
    moved = first.directory.with_name("relocated")
    first.directory.rename(moved)
    saved, restored = load_program(moved)
    assert restored.digest == program.digest
    assert saved.program_path.name == "profiled_program.ssp"
    with saved.program_path.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="artifact digest"):
        load_program(moved)


def test_saved_corpus_plans_without_build_torch_or_runtime(tmp_path):
    saved = save_profiled_program(
        tmp_path / "corpus",
        identity=ProgramCaseIdentity("tiny", "pytorch", 64, 32, 2),
        program=measured(),
        metadata={},
    )
    child = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "tests.benchmarking._plan_corpus",
            str(tmp_path),
            str(saved.directory),
            "4096",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stdout + child.stderr


def test_estimates_are_not_promoted_to_physical_allocation_evidence(tmp_path):
    from benchmarking.planning_eval.config import load_frontier_config
    from benchmarking.planning_eval.inputs import planning_problem
    from tests.benchmarking.test_planning_eval import _CONFIG

    source = measured()
    source = replace(source, metadata={"memory_kind": "estimated"})
    with pytest.raises(ValueError, match="measured allocation evidence"):
        planning_problem(source, load_frontier_config(_CONFIG))

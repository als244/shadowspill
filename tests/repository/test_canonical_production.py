"""Guard the single supported production implementation.

These checks deliberately inspect source rather than behavior: compatibility paths can
otherwise remain dormant for years while still increasing the runtime's surface area.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "src" / "shadowspill"
C_ROOT = ROOT / "csrc"


def _production_sources() -> tuple[Path, ...]:
    suffixes = {".c", ".cc", ".cpp", ".h", ".hpp", ".py"}
    return tuple(
        sorted(
            path
            for root in (PYTHON_ROOT, C_ROOT)
            for path in root.rglob("*")
            if path.is_file() and path.suffix in suffixes
        )
    )


def test_production_does_not_import_reference_implementations() -> None:
    offenders = [
        path.relative_to(ROOT)
        for path in PYTHON_ROOT.rglob("*.py")
        if re.search(
            r"(?:from|import)\s+(?:reference|shadowspill\.reference)(?:\.|\s|$)",
            path.read_text(encoding="utf-8"),
        )
    ]
    assert offenders == []


def test_removed_compatibility_names_do_not_return() -> None:
    forbidden = (
        "shadowspill_before_task(",
        "shadowspill_after_task(",
        "shadowspill_abort_task(",
        "shadowspill_pytorch_before_task(",
        "shadowspill_pytorch_after_task(",
        "shadowspill_pytorch_abort_task(",
        "resize_spill_pool",
        "relocate_model_state",
        "externalize_model_state",
        "progress_thread",
        "dense_id",
    )
    offenders: list[str] = []
    for path in _production_sources():
        source = path.read_text(encoding="utf-8")
        offenders.extend(
            f"{path.relative_to(ROOT)}: {token}"
            for token in forbidden
            if token in source
        )
    assert offenders == []


def test_worker_hot_loop_has_no_sleeping_wait_primitive() -> None:
    worker = (C_ROOT / "src" / "runtime" / "worker.c").read_text(encoding="utf-8")
    forbidden = ("pthread_cond", "futex", "nanosleep", "sched_yield", "usleep")
    assert [token for token in forbidden if token in worker] == []


def test_only_shadowspill_is_installed_from_the_source_tree() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'wheel.packages = ["src/shadowspill"]' in project

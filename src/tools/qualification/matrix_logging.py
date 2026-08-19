"""Timestamped matrix console logging shared by qualification launchers.

The format follows the planning-evaluation runner: every unit of work opens
with a labeled START block describing exactly what is about to run, streamed
child output is visible live under a ``[cell/N]`` prefix, and every unit
closes with a status block carrying UTC START, STOP, and DURATION records.
Console lines are duplicated into one persistent matrix log with UTC
timestamps; each cell log additionally receives the child's exact lines with
timestamps prepended.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""

    return datetime.now(UTC).isoformat(timespec="seconds")


def format_bytes(count: int) -> str:
    """Render a byte count as GiB with the exact count alongside."""

    return f"{count / (1 << 30):.3f} GiB ({count} bytes)"


class MatrixConsole:
    """Duplicate matrix progress lines to stdout and one persistent log."""

    def __init__(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open("a", buffering=1)
        self.path = log_path

    def __enter__(self) -> MatrixConsole:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._log.close()

    def emit(self, line: str = "", *, prefix: str | None = None) -> None:
        """Write one line to the console and the timestamped matrix log."""

        rendered = line if prefix is None else f"{prefix} | {line}"
        print(rendered, flush=True)
        self._log.write(f"[{utc_now()}] {rendered}\n" if rendered else "\n")

    def block(
        self,
        header: str,
        details: list[str],
        *,
        prefix: str | None = None,
    ) -> None:
        """Emit one labeled block: a header line and indented detail lines."""

        self.emit(header, prefix=prefix)
        for detail in details:
            self.emit(f"  {detail}", prefix=prefix)

    def stream(
        self,
        command: list[str],
        *,
        cell_log_path: Path,
        prefix: str,
        environment: dict[str, str] | None = None,
    ) -> int:
        """Run a child, teeing each line to console, matrix log, and cell log.

        The cell log receives the child's exact lines with a UTC timestamp
        prepended; the console and matrix log carry the ``prefix``.
        """

        cell_log_path.parent.mkdir(parents=True, exist_ok=True)
        with cell_log_path.open("a", buffering=1) as cell_log:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=environment,
            )
            assert process.stdout is not None
            for line in process.stdout:
                text = line.rstrip("\n")
                cell_log.write(f"[{utc_now()}] {text}\n")
                sys.stdout.write(f"{prefix} | {text}\n")
                sys.stdout.flush()
                self._log.write(f"[{utc_now()}] {prefix} | {text}\n")
            process.stdout.close()
            return process.wait()


__all__ = ["MatrixConsole", "format_bytes", "utc_now"]

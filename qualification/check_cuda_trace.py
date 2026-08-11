"""Validate CUDA/NVTX invariants in an Nsight Systems SQLite export."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

FORBIDDEN_APIS = (
    "cuCtxSynchronize",
    "cuMemAddressFree",
    "cuMemAddressReserve",
    "cuMemCreate",
    "cuMemMap",
    "cuMemRelease",
    "cuMemSetAccess",
    "cuMemUnmap",
    "cudaDeviceSynchronize",
)
REQUIRED_RANGES = (
    "shadowspill.pytorch.storage_rebind",
    "shadowspill.pytorch.task.",
    "shadowspill.runtime.allocate",
    "shadowspill.runtime.transfer.d2h",
    "shadowspill.runtime.transfer.h2d",
    "shadowspill.runtime.wait_event",
)


def _scalar(
    connection: sqlite3.Connection, query: str, values: tuple[object, ...]
) -> int:
    row = connection.execute(query, values).fetchone()
    if row is None:
        raise AssertionError("trace query unexpectedly returned no row")
    return int(row[0])


def check_trace(path: Path) -> None:
    """Raise when a Phase-5 CUDA trace violates the runtime contract."""

    with sqlite3.connect(path) as connection:
        api_count = _scalar(
            connection,
            """
            SELECT count(*)
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
            JOIN StringIds AS names ON names.id = runtime.nameId
            WHERE names.value = ?
            """,
            ("cuMemAlloc_v2",),
        )
        if api_count != 1:
            raise AssertionError(
                f"expected one CUDA slab allocation, observed {api_count}"
            )
        pinned_count = _scalar(
            connection,
            """
            SELECT count(*)
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
            JOIN StringIds AS names ON names.id = runtime.nameId
            WHERE names.value = ?
            """,
            ("cuMemHostAlloc",),
        )
        if pinned_count != 1:
            raise AssertionError(
                f"expected one pinned-host allocation, observed {pinned_count}"
            )
        forbidden = connection.execute(
            """
            SELECT DISTINCT names.value
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
            JOIN StringIds AS names ON names.id = runtime.nameId
            WHERE names.value IN ({})
            ORDER BY names.value
            """.format(",".join("?" for _ in FORBIDDEN_APIS)),
            FORBIDDEN_APIS,
        ).fetchall()
        if forbidden:
            raise AssertionError(
                "forbidden CUDA APIs appeared: "
                + ", ".join(str(row[0]) for row in forbidden)
            )
        for prefix in REQUIRED_RANGES:
            count = _scalar(
                connection,
                "SELECT count(*) FROM NVTX_EVENTS WHERE text LIKE ?",
                (prefix + "%",),
            )
            if count == 0:
                raise AssertionError(f"missing NVTX range prefix {prefix!r}")
        overlap = connection.execute(
            """
            SELECT copy_names.label, count(*)
            FROM CUPTI_ACTIVITY_KIND_MEMCPY AS copy
            JOIN ENUM_CUDA_MEMCPY_OPER AS copy_names ON copy_names.id = copy.copyKind
            JOIN CUPTI_ACTIVITY_KIND_KERNEL AS kernel
              ON copy.start < kernel.end AND kernel.start < copy.end
            JOIN StringIds AS kernel_names ON kernel_names.id = kernel.shortName
            WHERE kernel_names.value LIKE '%spin_kernel%'
            GROUP BY copy_names.label
            """
        ).fetchall()
        overlap_counts = {str(name): int(count) for name, count in overlap}
        if overlap_counts.get("Host-to-Device", 0) < 1:
            raise AssertionError("no H2D transfer overlapped the compute kernel")
        if overlap_counts.get("Device-to-Host", 0) < 1:
            raise AssertionError("no D2H transfer overlapped the compute kernel")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    arguments = parser.parse_args()
    check_trace(arguments.sqlite.resolve())
    print(f"CUDA trace gate passed: {arguments.sqlite.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

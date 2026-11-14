"""Html for pool-occupancy diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape


def _row(cells: Sequence[str]) -> str:
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


_STYLE = (
    "<style>body{font:14px/1.4 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;"
    "margin:24px;max-width:1200px}table{border-collapse:collapse;margin:8px 0 20px}"
    "td,th{border-bottom:1px solid #d2d2d7;padding:4px 10px;text-align:left;"
    "font-variant-numeric:tabular-nums;vertical-align:top}th{font-weight:600}"
    "pre{background:#f4f4f6;padding:10px;border-radius:8px;overflow-x:auto;"
    "font-size:12px}nav a{margin-right:14px}</style>"
)


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{escape(title)}</title>{_STYLE}</head><body>{body}</body></html>"
    )


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    return (
        "<table><tr>"
        + "".join(f"<th>{cell}</th>" for cell in head)
        + "</tr>"
        + "".join(_row(cells) for cells in rows)
        + "</table>"
    )

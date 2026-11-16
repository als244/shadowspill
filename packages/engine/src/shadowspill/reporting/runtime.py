"""Bounded standalone HTML comparing an invocation with its saved Plan."""

import json
from collections.abc import Mapping
from typing import Any

from shadowspill.artifacts.plan import Plan
from shadowspill.diagnostics.occupancy.page import html_parts

from .occupancy import occupancy_data


def runtime_html(
    plan: Plan,
    diagnostics: Mapping[str, Any],
    *,
    invocation: Mapping[str, Any],
    max_bytes: int,
) -> bytearray:
    """Render inside an admitted buffer; never replan or substitute predictions."""
    payload = bytearray()

    def append(text: str) -> None:
        data = text.encode()
        if len(payload) + len(data) > max_bytes:
            raise ValueError("runtime HTML exceeds its admitted report bytes")
        payload.extend(data)

    def write_json(value: Any) -> None:
        for part in json.JSONEncoder(allow_nan=False, separators=(",", ":")).iterencode(
            value
        ):
            append(part.replace("<", "\\u003c"))

    prefix, suffix = html_parts("Runtime trace and saved Plan")
    append(prefix)
    append('{"kind":"runtime_trace","rows":')
    write_json([{"kind": "invocation", "values": invocation}])
    append(',"views":[')
    for index, (mode, evidence) in enumerate(
        (("Measured runtime", diagnostics), ("Predicted Plan", None))
    ):
        if index:
            append(",")
        write_json(
            {
                "rank": invocation["rank"],
                "id": invocation["plan"],
                "mode": mode,
                "data": occupancy_data(
                    plan,
                    description=(
                        f"{invocation['operation']} · rank {invocation['rank']}"
                        f" · Plan {invocation['plan']}"
                    ),
                    diagnostics=evidence,
                ),
            }
        )
    append("]}")
    append(suffix)
    return payload

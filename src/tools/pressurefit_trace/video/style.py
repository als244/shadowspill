"""Shared visual palette and duration labels for all PressureFit movies."""

BG = "#0b111c"
PANEL = "#101c2b"
WHITE = "#e9f0fa"
MUTED = "#8fa2b9"
RED = "#ff697b"
GREEN = "#64dec1"
RECOMPUTE = "#98d20b"


def duration_label(nanoseconds):
    for scale, unit in ((1e9, "s"), (1e6, "ms"), (1e3, "µs")):
        if nanoseconds >= scale:
            digits = 2 if unit == "s" else 1
            return f"{nanoseconds / scale:.{digits}f} {unit}"
    return f"{nanoseconds} ns"

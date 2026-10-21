"""What every baseline engine shares.

``geometry`` is the shape of one optimizer update; ``precision`` the dtype
contract every engine exposes and the refusal of a combination it cannot
honor; ``models`` the catalog model an engine builds, in full or LoRA mode,
with its objective and useful work; ``results`` the row an engine writes and
the measurement contract; ``provenance`` what a row records about the code,
the packages and the hardware; ``data`` the synthetic tokens each rank draws
and the pre-tokenized corpus a real-data run reads.

Nothing here imports a training system, and only ``models`` and ``data``
import PyTorch, inside the functions that need it, so a sweep launcher can
import this package without initializing a device.
"""

from .geometry import accumulation_steps, recompute_layers
from .results import summarize_cycles, write_json

__all__ = ["accumulation_steps", "recompute_layers", "summarize_cycles", "write_json"]

"""What every baseline harness shares.

``geometry`` is the shape of one optimizer update; ``precision`` the dtype
contract every harness exposes and the refusal of a combination it cannot
honor; ``models`` the catalog model a harness builds, in full or LoRA mode,
with its objective and useful work; ``results`` the row a harness writes and
the measurement contract; ``provenance`` what a row records about the code,
the packages, the hardware and the kernels that ran; ``data`` the synthetic
tokens each rank draws and the pre-tokenized corpus a real-data run reads.

Nothing here imports a training system, and only ``models`` and ``data``
import PyTorch, inside the functions that need it, so a sweep launcher can
import this package without initializing a device.
"""

from .geometry import accumulation_steps, recompute_layers
from .results import summarize_cycles, write_json

__all__ = ["accumulation_steps", "recompute_layers", "summarize_cycles", "write_json"]

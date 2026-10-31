"""What every baseline engine shares.

``geometry`` is the shape of one optimizer update; ``precision`` the dtype
contract every engine exposes and the refusal of a combination it cannot
honor; ``models`` the catalog model an engine builds, in full or LoRA mode,
with its objective and useful work; ``results`` the row an engine writes and
the measurement contract; ``provenance`` what a row records about the code,
the packages and the hardware; ``data`` the synthetic tokens each rank draws
and the pre-tokenized corpus a real-data run reads; ``settings`` what every
engine's settings share and the forms a case takes; ``launch`` the start of
one case of any engine; ``preparation`` the preparation artifacts cases
share; ``runner`` what every engine's trainer process shares; ``training``
the catalog model every engine trains and the helpers around its
construction; ``reference`` the tiny-model check of an engine's
accumulation, reduction and optimizer ownership; ``annotations`` the
optional NVTX scopes around phases and recomputation.

Nothing here imports a training system. ``reference`` and ``training``
import PyTorch as they load; every other module that uses it imports it
inside the functions that need it, so a sweep launcher can import this
package, its settings and its launcher without initializing a device.
"""

from .geometry import accumulation_steps, recompute_layers
from .results import summarize_cycles, write_json

__all__ = ["accumulation_steps", "recompute_layers", "summarize_cycles", "write_json"]

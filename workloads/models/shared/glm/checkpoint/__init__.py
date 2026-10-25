"""Stream a local HF checkpoint into model state already allocated in a pool.

This module knows the GLM layouts; a family's subclass of `GLMCheckpoint`
supplies its names. It imports no ShadowSpill runtime code: ``load_into`` is an
ordinary in-place initializer usable with any destination.

The routed experts are stacked banks, ``w13 [E, 2I, H]`` and ``w2 [E, H, I]``,
held as ``expert_storage`` says: ``"bf16"``, ``"fp8_block"``
(`BlockFP8Weight`, E4M3 once with a scale per 128x128 block) or ``"fp8"``
(`QuackFP8Weight`, two E4M3 orientations with per-row descales) -- by
default bf16 for a dense checkpoint and ``"fp8_block"`` for a compressed one.
A dense checkpoint's experts copy into bf16, an FP8 checkpoint's into
``"fp8_block"``, byte for byte; every other pairing decodes each expert and
quantizes it as the storage does. Either way a bank is assembled in host
memory, an expert at a time, and written to its destination once."""

from .reader import GLMCheckpoint
from .tensors import physical_state

__all__ = ["GLMCheckpoint", "physical_state"]

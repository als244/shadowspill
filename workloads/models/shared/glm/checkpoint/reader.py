"""Metadata-first, streaming GLM checkpoint reader shared by both families."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from mlops.sequential_moe.precision.checkpoint import HFWeights
from torch import nn

from ..moe import MoE
from .experts import plan_experts
from .loading import initialize_lora, load_conversions, load_copies
from .model import configure, map_tensors, replace_projections
from .tensors import physical_state
from .validation import apply_dtypes, inventory, used_keys, validate_copies

EXPERT_STORAGES = ("bf16", "fp8", "fp8_block")


class GLMCheckpoint:
    """Validate checkpoint metadata, declare a meta model, and load incrementally.

    No converted checkpoint is written. Keep this reader open through import;
    it may be closed immediately afterwards. Vision is optional; MTP is excluded.
    """

    #: A family's subclass supplies these four: the prefix its text weights
    #: carry in the checkpoint, the reader that turns the checkpoint's
    #: metadata into its ``Config``, the translation of a model state name
    #: into the checkpoint's, and its ``Model`` type.
    source_prefix: str
    config_from_metadata: Callable[..., Any]
    source_name: Callable[[str], str]
    model_type: type[nn.Module]

    def __init__(self, directory, *, gemm_precision="bf16", expert_storage=None):
        if expert_storage not in (None, *EXPERT_STORAGES):
            raise ValueError(f"expert_storage must be one of {EXPERT_STORAGES}")
        if gemm_precision != "bf16":
            raise ValueError(
                "The composed GLM checkpoint path currently supports BF16 GEMMs "
                "only. Low-precision GEMM recipes, including checkpoint activation "
                "scales, are not integrated yet; compressed weight storage remains "
                "FP8 or NVFP4."
            )
        self.directory = Path(directory)
        self.weights = HFWeights(directory)
        self.metadata = json.loads((self.directory / "config.json").read_text())
        self.gemm_precision = gemm_precision
        self.expert_storage = expert_storage
        self.copies = []
        self.conversions = []
        self.lora_initializers = {}
        self.inventory = None
        self._keys = None
        self._expert_keys = set()
        self._expert_shapes = None

    def vision_config(self, include_vision):
        """The vision encoder's configuration when one is asked for; a
        language-only family refuses."""
        if include_vision:
            raise ValueError("this checkpoint's model is text-only")
        return None

    def build_model(
        self,
        *,
        layer_limit=None,
        lora_rank=0,
        lora_alpha=32.0,
        lora_factor_dtype=torch.float32,
        lora_head=False,
        include_vision=False,
        router_weight_dtype=None,
    ):
        """Declare checkpoint storage on meta and validate every copy up front.

        Router weights retain checkpoint dtype unless router_weight_dtype
        explicitly requests BF16 or FP32. Routing logits remain FP32.
        """
        if self.inventory is not None:
            raise RuntimeError("Use a fresh GLMCheckpoint to build another model")
        if router_weight_dtype not in (None, torch.bfloat16, torch.float32):
            raise ValueError("router_weight_dtype must be None, BF16 or FP32")
        c = configure(
            self,
            layer_limit=layer_limit,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_factor_dtype=lora_factor_dtype,
            lora_head=lora_head,
            router_weight_dtype=router_weight_dtype,
        )
        vision = self.vision_config(include_vision)
        model = self.model_type(c, device="meta", vision_config=vision)
        replace_projections(self, model)
        for name, module in model.named_modules():
            if isinstance(module, MoE):
                plan_experts(self, name, c)
        map_tensors(self, model, c)
        targets = apply_dtypes(self, model, router_weight_dtype)
        validate_copies(self, targets, router_weight_dtype)
        used = used_keys(self, c, include_vision)
        if not c.lora_rank:
            model.requires_grad_(False)
        self.inventory = inventory(self, model, c, vision, targets, used)
        return model

    @torch.no_grad()
    def load_into(
        self,
        model,
        *,
        progress=None,
        adopted=frozenset(),
        write_tensor=None,
        on_tensor=None,
    ):
        """Fill supplied state in place; retain no tensor payload between copies.

        ``adopted`` names state a pool image already holds; those copies are
        skipped and the rest, LoRA factors included, is written as usual.
        ``write_tensor(destination, fill)`` may execute a physical checkpoint
        read only on a storage's owner and distribute it. The default writes
        ordinary tensors without requiring a runtime. LoRA initialization keeps
        its existing RNG sequence; representation conversions use the ordinary
        per-expert path. ``on_tensor(name)`` reports each destination after
        its bytes have been published, including distributed shard copies.
        """
        if self.inventory is None:
            raise RuntimeError("Call build_model() before loading")
        targets = physical_state(model)
        initialize_lora(self, targets, adopted, on_tensor)
        done, total = load_copies(
            self, targets, adopted, progress, write_tensor, on_tensor
        )
        load_conversions(self, targets, adopted, progress, on_tensor, done, total)

    def close(self):
        self.weights.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

"""Stream a local HF checkpoint into model state already allocated in a pool.

This module knows GLM names and layouts. It imports no ShadowSpill runtime code:
``load_into`` is an ordinary in-place initializer usable with any destination.
"""

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from mlops.lora import LoRALinear
from mlops.sequential_moe.experts.layer import SequentialExpert
from mlops.sequential_moe.precision.checkpoint import HFWeights, join_gate_up
from mlops.sequential_moe.precision.linear import FrozenLinear, FrozenLoRALinear
from mlops.sequential_moe.precision.storage import DENSE
from torch import nn

from .hf_config import from_huggingface
from .model import LanguageModel
from .vision import VisionConfig


def physical_state(model):
    """Enumerate ordinary tensors, including public tensor-wrapper components."""
    result = {}

    def add(name, value):
        flatten = getattr(value, "__tensor_flatten__", None)
        if flatten is None:
            result[name] = value
        else:
            for child in flatten()[0]:
                add(name + "." + child, getattr(value, child))

    for name, value in (*model.named_parameters(), *model.named_buffers()):
        add(name, value)
    return result


def source_name(name):
    if name.startswith("visual."):
        return "model." + name
    for short, original in (("attn_hc", "hc_attn"), ("ffn_hc", "hc_ffn")):
        for field in ("fn", "base", "scale"):
            name = name.replace(f".{short}.{field}", f".{original}_{field}")
    name = name.replace(".forget_gate.", ".")
    name = name.replace(".backend.router", ".gate.weight")
    name = name.replace(
        ".mlp.e_score_correction_bias", ".mlp.gate.e_score_correction_bias"
    )
    return name if name.startswith("lm_head.") else "model.language_model." + name


@dataclass(frozen=True)
class Copy:
    destination: str
    source: str
    start: int | None = None
    stop: int | None = None
    reciprocal: bool = False

    def view(self, tensors):
        value = tensors[self.destination]
        return value if self.start is None else value[self.start : self.stop]


class GLMCheckpoint:
    """Validate checkpoint metadata, declare a meta model, and load incrementally.

    No converted checkpoint is written. Keep this reader open through import;
    it may be closed immediately afterwards. Vision is optional; MTP is excluded.
    """

    def __init__(self, directory, *, gemm_precision="bf16"):
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
        self.copies = []
        self.lora_initializers = {}
        self.inventory = None

    def _projection_copies(self, destinations, prefix, spec, *, starts=None):
        keys = (
            ["weight"]
            if spec.format in DENSE
            else (
                ["weight", "weight_scale_inv"]
                if spec.format == "fp8"
                else ["weight_packed", "weight_scale", "weight_global_scale"]
            )
        )
        for i, (destination, key) in enumerate(zip(destinations, keys, strict=True)):
            src = prefix + "." + key
            shape = self.weights.source.tensor(src, meta=True).shape
            start = starts[i] if starts is not None else None
            self.copies.append(
                Copy(
                    destination,
                    src,
                    start,
                    start + shape[0] if start is not None else None,
                    key == "weight_global_scale",
                )
            )

    def build_model(
        self,
        *,
        layer_limit=None,
        lora_rank=0,
        lora_alpha=32.0,
        lora_factor_dtype=torch.float32,
        lora_head=False,
        include_vision=False,
    ):
        """Declare final shapes/dtypes on meta and validate every copy up front."""
        if self.inventory is not None:
            raise RuntimeError("Use a fresh GLMCheckpoint to build another model")
        c = from_huggingface(self.metadata, gemm_precision=self.gemm_precision)
        # Choose the actual routed format before constructing wrapper parameters.
        first = c.first_k_dense_replace
        prefix = f"model.language_model.layers.{first}.mlp.experts.0"
        _, spec = self.weights.projection(
            prefix + ".gate_proj", (c.moe_intermediate_size, c.hidden_size), meta=True
        )
        c = from_huggingface(
            self.metadata,
            expert_storage=spec.format,
            gemm_precision=self.gemm_precision,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_factor_dtype=lora_factor_dtype,
            lora_head=lora_head,
        )
        if layer_limit is not None:
            if not 1 <= layer_limit <= c.num_hidden_layers:
                raise ValueError("layer_limit must select a nonempty decoder prefix")
            c = replace(c, num_hidden_layers=layer_limit)
        vision = (
            VisionConfig.from_huggingface(self.metadata) if include_vision else None
        )
        model = LanguageModel(c, device="meta", vision_config=vision)

        # Replace quantized dense projections with component-bearing modules.
        for name, module in list(model.named_modules()):
            if not isinstance(module, (nn.Linear, LoRALinear)):
                continue
            prefix = source_name(name + ".weight").removesuffix(".weight")
            parts, spec = self.weights.projection(
                prefix, tuple(module.weight.shape), meta=True
            )
            if spec.format in DENSE:
                continue
            if name == "lm_head":
                raise ValueError(
                    "GLM checkpoint output heads currently require dense storage"
                )
            parent, _, leaf = name.rpartition(".")
            replacement = FrozenLinear(parts, spec, gemm_precision=self.gemm_precision)
            if isinstance(module, LoRALinear):
                replacement = FrozenLoRALinear(replacement, module.lora_config)
            model.get_submodule(parent).set_submodule(leaf, replacement)
            self._projection_copies(
                [name + f".components.{i}" for i in range(spec.count)], prefix, spec
            )

        for name, expert in model.named_modules():
            if not isinstance(expert, SequentialExpert):
                continue
            prefix = "model.language_model." + name.replace(".backend", "")
            h, d = c.moe_intermediate_size, c.hidden_size
            gate = self.weights.projection(prefix + ".gate_proj", (h, d), meta=True)
            up = self.weights.projection(prefix + ".up_proj", (h, d), meta=True)
            down = self.weights.projection(prefix + ".down_proj", (d, h), meta=True)
            joined = join_gate_up(gate, up)
            expert.load_weights_(gate_up=joined, down=down)
            for kind, (_parts, spec) in (("gate_up", joined), ("down", down)):
                parameter = getattr(expert, kind)
                flatten = getattr(parameter, "__tensor_flatten__", None)
                destinations = (
                    [name + "." + kind + "." + key for key in flatten()[0]]
                    if flatten
                    else [
                        name + "." + kind + suffix
                        for suffix in ("", "_block_scale", "_global_scale")[
                            : spec.count
                        ]
                    ]
                )
                if kind == "down":
                    self._projection_copies(destinations, prefix + ".down_proj", spec)
                else:
                    for index, source in enumerate(("gate_proj", "up_proj")):
                        starts = [index * h]
                        if spec.count > 1:
                            starts.append(index * (h // spec.block_rows))
                        if spec.count > 2:
                            starts.append(index)
                        self._projection_copies(
                            destinations, prefix + "." + source, gate[1], starts=starts
                        )
            # Preserve activation quantization metadata for the later GEMM recipe.
            for projection in ("gate", "up", "down"):
                key = prefix + f".{projection}_proj.input_global_scale"
                if key in self.weights.files:
                    attr = projection + "_input_global_scale"
                    expert.register_buffer(
                        attr, self.weights.source.tensor(key, meta=True)
                    )
                    self.copies.append(Copy(name + "." + attr, key))

        targets = physical_state(model)
        bound = {copy.destination for copy in self.copies}
        for name, value in targets.items():
            if name in bound:
                continue
            leaf = name.removesuffix(".compute").rsplit(".", 1)[-1]
            if c.lora_rank and leaf in {
                "lora_a",
                "lora_b",
                "gate_up_a",
                "gate_up_b",
                "down_a",
                "down_b",
            }:
                self.lora_initializers[name] = leaf.endswith("_a")
                continue
            if name.endswith(".self_attn.conv1d.weight"):
                width = value.shape[0] // 3
                prefix = source_name(name).removesuffix("conv1d.weight")
                for i, letter in enumerate(("q", "k", "v")):
                    self.copies.append(
                        Copy(
                            name,
                            prefix + letter + "_conv1d.weight",
                            i * width,
                            (i + 1) * width,
                        )
                    )
            else:
                source = source_name(name.removesuffix(".compute"))
                self.copies.append(Copy(name, source))
        for copy in self.copies:
            source = self.weights.source.tensor(copy.source, meta=True)
            target = copy.view(targets)
            if source.shape != target.shape:
                raise ValueError(
                    f"{copy.source}: {tuple(source.shape)} does not match "
                    f"{copy.destination}: {tuple(target.shape)}"
                )
            if source.dtype != target.dtype and not (
                source.is_floating_point() and target.dtype == torch.float32
            ):
                raise ValueError(
                    f"Unsupported dtype conversion {copy.source}: "
                    f"{source.dtype} -> {target.dtype}"
                )
        used = {copy.source for copy in self.copies}
        required = {
            key
            for key in self.weights.files
            if key.startswith(
                (
                    "model.language_model.embed_tokens.",
                    "model.language_model.norm.",
                    "lm_head.",
                )
            )
        }
        for i in range(c.num_hidden_layers):
            required.update(
                key
                for key in self.weights.files
                if key.startswith(f"model.language_model.layers.{i}.")
            )
        if include_vision:
            required.update(
                key for key in self.weights.files if key.startswith("model.visual.")
            )
        if required - used:
            raise ValueError(f"Unmapped model tensors: {sorted(required - used)[:10]}")
        if not c.lora_rank:
            model.requires_grad_(False)
        self.inventory = {
            "layers": c.num_hidden_layers,
            "vision_layers": 0 if vision is None else vision.depth,
            "source_tensors": len(used),
            "physical_tensors": len(targets),
            "model_bytes": sum(t.numel() * t.element_size() for t in targets.values()),
            "largest_tensor_bytes": max(
                t.numel() * t.element_size() for t in targets.values()
            ),
            "expert_storage": c.expert_storage,
            "gemm_precision": c.gemm_precision,
            "lora_rank": c.lora_rank,
            "lora_alpha": c.lora_alpha,
            "lora_factor_dtype": str(c.lora_factor_dtype),
            "lora_head": c.lora_head,
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "ignored_source_tensors": len(self.weights.files.keys() - used),
        }
        return model

    @torch.no_grad()
    def load_into(self, model, *, progress=None):
        """Fill supplied state in place; retain no tensor payload between copies."""
        if self.inventory is None:
            raise RuntimeError("Call build_model() before loading")
        targets = physical_state(model)
        # Construct only one factor at a time; imported meta parameters have no
        # initialized bytes. copy_ also supports non-addressable spill pools.
        for name, is_a in self.lora_initializers.items():
            target = targets[name]
            value = torch.empty(tuple(target.shape), dtype=target.dtype, device="cpu")
            if is_a:
                nn.init.kaiming_uniform_(value, a=math.sqrt(5))
            else:
                nn.init.zeros_(value)
            target.copy_(value)
        grouped = {}
        for copy in self.copies:
            grouped.setdefault(copy.destination, []).append(copy)
        done = 0
        for destination, copies in grouped.items():
            target = targets[destination]
            if len(copies) == 1 and copies[0].start is None:
                staging = target
            else:
                # Assemble joined gate/up and Q/K/V rows before publishing.
                # This writes each destination once even for an SSD pool.
                # Scratch is bounded by one joined projection, never a layer.
                expected_start = 0
                for copy in copies:
                    if copy.start != expected_start or copy.stop is None:
                        raise ValueError(
                            f"Incomplete/overlapping rows for {destination}"
                        )
                    expected_start = copy.stop
                if expected_start != target.shape[0]:
                    raise ValueError(f"Incomplete rows for {destination}")
                staging = torch.empty(
                    tuple(target.shape), dtype=target.dtype, device="cpu"
                )
            for copy in copies:
                source = self.weights.source.tensor(copy.source)
                if copy.reciprocal:
                    if not bool(torch.isfinite(source).all() & (source > 0).all()):
                        raise ValueError(f"Invalid NVFP4 global scale: {copy.source}")
                    source = source.reciprocal()
                view = (
                    staging if copy.start is None else staging[copy.start : copy.stop]
                )
                view.copy_(source)
                del source, view
                self.weights.source.release_pages(copy.source)
                done += 1
                if progress is not None:
                    progress(done, len(self.copies), copy.source)
            if staging is not target:
                target.copy_(staging)
            del staging

    def close(self):
        self.weights.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

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
memory, an expert at a time, and written to its destination once.
"""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from mlops.lora import LoRALinear
from mlops.sequential_moe.precision.checkpoint import HFWeights
from mlops.sequential_moe.precision.linear import FrozenLinear, FrozenLoRALinear
from mlops.sequential_moe.precision.storage import DENSE, materialize
from torch import nn

from .moe import MoE

EXPERT_STORAGES = ("bf16", "fp8", "fp8_block")
LORA_FACTORS = {
    "lora_a",
    "lora_b",
    "lora_gate_up_a",
    "lora_gate_up_b",
    "lora_down_a",
    "lora_down_b",
}


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


@dataclass(frozen=True)
class Copy:
    """``source`` into rows ``start:stop`` of ``destination`` (of its expert
    ``index``, for a stacked bank)."""

    destination: str
    source: str
    start: int | None = None
    stop: int | None = None
    reciprocal: bool = False
    index: int | None = None

    def view(self, tensors):
        value = tensors[self.destination]
        if self.index is not None:
            value = value[self.index]
        return value if self.start is None else value[self.start : self.stop]


@dataclass(frozen=True)
class Conversion:
    """One layer's ``w13`` or ``w2`` bank decoded from the checkpoint, an
    expert at a time, and stored as ``storage`` holds it."""

    module: str
    kind: str
    storage: str
    experts: tuple

    @property
    def destinations(self):
        name = f"{self.module}.{self.kind}"
        if self.storage == "bf16":
            return (name,)
        components = (
            ("_data", "_scale")
            if self.storage == "fp8_block"
            else ("_data", "_transposed", "_scale", "_transposed_scale")
        )
        return tuple(f"{name}.{component}" for component in components)


def _projections(kind):
    return ("gate_proj", "up_proj") if kind == "w13" else ("down_proj",)


def _check_rows(destination, copies, shape):
    """The copies into ``destination`` cover each row (of each expert) once,
    in order."""
    stacked = copies[0].index is not None
    rows = shape[1] if stacked else shape[0]
    spans = {}
    for copy in copies:
        spans.setdefault(copy.index, []).append(copy)
    if stacked and sorted(spans) != list(range(shape[0])):
        raise ValueError(f"Incomplete experts for {destination}")
    for group in spans.values():
        position = 0
        for copy in group:
            start = 0 if copy.start is None else copy.start
            stop = rows if copy.stop is None else copy.stop
            if start != position:
                raise ValueError(f"Incomplete/overlapping rows for {destination}")
            position = stop
        if position != rows:
            raise ValueError(f"Incomplete rows for {destination}")


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

    def _projection_keys(self, prefix):
        """Every checkpoint tensor of one projection (``prefix.weight``, its
        scales, any activation-scale metadata)."""
        if self._keys is None:
            self._keys = {}
            for key in self.weights.files:
                self._keys.setdefault(key.rsplit(".", 1)[0], []).append(key)
        return tuple(self._keys.get(prefix, ()))

    def _plan_experts(self, name, c):
        """The copies, or else the conversions, that fill one layer's banks."""
        h, d = c.moe_intermediate_size, c.hidden_size
        self._expert_shapes = {
            "gate_proj": (h, d),
            "up_proj": (h, d),
            "down_proj": (d, h),
        }
        source = self.source_prefix + name + ".experts"
        experts = [(index, f"{source}.{index}") for index in range(c.num_local_experts)]
        for kind in ("w13", "w2"):
            specs = [
                self.weights.projection(
                    f"{prefix}.{projection}", self._expert_shapes[projection], meta=True
                )[1]
                for _index, prefix in experts
                for projection in _projections(kind)
            ]
            formats = {spec.format for spec in specs}
            exact = len(formats) == 1 and (
                (c.expert_storage == "bf16" and formats == {"bf16"})
                or (
                    c.expert_storage == "fp8_block"
                    and formats == {"fp8"}
                    and all(s.block_rows == s.block_cols == 128 for s in specs)
                    and (kind == "w2" or h % 128 == 0)
                )
            )
            if not exact:
                self.conversions.append(
                    Conversion(name, kind, c.expert_storage, tuple(experts))
                )
                continue
            for index, prefix in experts:
                self._copy_expert(name, kind, index, prefix, c.expert_storage, h)

    def _copy_expert(self, name, kind, index, prefix, storage, rows):
        """One expert's tensors as stored: a dense weight's rows into the bf16
        bank, or an FP8 weight's E4M3 rows and 128x128 block scales into a
        block-scaled bank's components; ``w13`` gate rows, then up rows."""
        scale_key = (
            "weight_scale" if self.weights.encoding == "mixed" else "weight_scale_inv"
        )
        parts = (
            (("", "weight", rows),)
            if storage == "bf16"
            else (("._data", "weight", rows), ("._scale", scale_key, rows // 128))
        )
        for position, projection in enumerate(_projections(kind)):
            self._expert_keys.update(self._projection_keys(f"{prefix}.{projection}"))
            for component, key, count in parts:
                start, stop = (
                    (position * count, (position + 1) * count)
                    if kind == "w13"
                    else (None, None)
                )
                self.copies.append(
                    Copy(
                        f"{name}.{kind}{component}",
                        f"{prefix}.{projection}.{key}",
                        start,
                        stop,
                        index=index,
                    )
                )

    def _converted(self, prefix, conversion, device):
        """One expert's ``w13`` or ``w2`` decoded from the checkpoint and taken
        to ``conversion.storage``'s components, in its destinations' order."""
        from mlops.expert_parallel.quack.parameters.fp8 import QuackFP8Weight
        from mlops.kernels.block_fp8 import quantize_blocks

        values = []
        for projection in _projections(conversion.kind):
            name = f"{prefix}.{projection}"
            parts, spec = self.weights.projection(name, self._expert_shapes[projection])
            values.append(
                materialize([part.to(device) for part in parts], spec, torch.float32)
            )
            del parts
            for key in self._projection_keys(name):
                self.weights.source.release_pages(key)
        value = torch.cat(values) if len(values) > 1 else values[0]
        if conversion.storage == "bf16":
            return (value.to(torch.bfloat16),)
        if conversion.storage == "fp8_block":
            data, scale = quantize_blocks(value[None])
            return data[0], scale[0]
        container = QuackFP8Weight.quantize(
            value[None], interleaved=conversion.kind == "w13"
        )
        return tuple(component[0] for component in container.components())

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
        c = self.config_from_metadata(self.metadata, gemm_precision=self.gemm_precision)
        # The routed experts' checkpoint format chooses their default storage:
        # dense stays bf16, compressed (FP8, NVFP4) becomes one-copy block FP8.
        first = c.first_k_dense_replace
        prefix = f"{self.source_prefix}layers.{first}.mlp.experts.0"
        _, spec = self.weights.projection(
            prefix + ".gate_proj", (c.moe_intermediate_size, c.hidden_size), meta=True
        )
        self.expert_format = spec.format
        storage = self.expert_storage or (
            "bf16" if spec.format in DENSE else "fp8_block"
        )
        c = self.config_from_metadata(
            self.metadata,
            expert_storage=storage,
            gemm_precision=self.gemm_precision,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_factor_dtype=lora_factor_dtype,
            lora_head=lora_head,
        )
        if router_weight_dtype is not None:
            c = replace(c, router_weight_dtype=router_weight_dtype)
        if layer_limit is not None:
            if not 1 <= layer_limit <= c.num_hidden_layers:
                raise ValueError("layer_limit must select a nonempty decoder prefix")
            c = replace(c, num_hidden_layers=layer_limit)
        vision = self.vision_config(include_vision)
        model = self.model_type(c, device="meta", vision_config=vision)

        # Replace quantized dense projections with component-bearing modules.
        for name, module in list(model.named_modules()):
            if not isinstance(module, (nn.Linear, LoRALinear)):
                continue
            prefix = self.source_name(name + ".weight").removesuffix(".weight")
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

        for name, module in model.named_modules():
            if isinstance(module, MoE):
                self._plan_experts(name, c)

        targets = physical_state(model)
        bound = {copy.destination for copy in self.copies}
        bound.update(
            destination
            for conversion in self.conversions
            for destination in conversion.destinations
        )
        for name, value in targets.items():
            if name in bound:
                continue
            leaf = name.removesuffix(".compute").rsplit(".", 1)[-1]
            if c.lora_rank and leaf in LORA_FACTORS:
                self.lora_initializers[name] = leaf.endswith("_a")
                continue
            if name.endswith(".self_attn.conv1d.weight"):
                width = value.shape[0] // 3
                prefix = self.source_name(name).removesuffix("conv1d.weight")
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
                source = self.source_name(name.removesuffix(".compute"))
                self.copies.append(Copy(name, source))
        # The checkpoint, not a randomly initialized model's training defaults,
        # owns frozen storage precision. In particular, HF stores mHC projections
        # and KDA convolutions in BF16 even though their arithmetic may use FP32.
        # Resolve joined tensors as a unit so mixed source dtypes cannot silently
        # round one component. Quantized containers already declare their exact
        # physical component dtypes above.
        source_dtypes = {}
        for copy in self.copies:
            dtype = self.weights.source.tensor(copy.source, meta=True).dtype
            source_dtypes.setdefault(copy.destination, set()).add(dtype)
        parameters = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        for name, dtypes in source_dtypes.items():
            if len(dtypes) != 1:
                raise ValueError(f"Mixed checkpoint dtypes for joined tensor {name}")
            dtype = next(iter(dtypes))
            if router_weight_dtype is not None and name.endswith(".mlp.router"):
                dtype = router_weight_dtype
            value = targets[name]
            if dtype == value.dtype:
                continue
            if name not in parameters and name not in buffers:
                raise ValueError(
                    f"Checkpoint dtype differs for physical component {name}"
                )
            if not value.is_floating_point() or not dtype.is_floating_point:
                raise ValueError(
                    f"Unsupported checkpoint storage dtype for {name}: {dtype}"
                )
            parent, _, leaf = name.rpartition(".")
            owner = model.get_submodule(parent)
            value = torch.empty_like(value, dtype=dtype)
            if name in parameters:
                value = nn.Parameter(
                    value, requires_grad=parameters[name].requires_grad
                )
            setattr(owner, leaf, value)
        targets = physical_state(model)
        for copy in self.copies:
            source = self.weights.source.tensor(copy.source, meta=True)
            target = copy.view(targets)
            if source.shape != target.shape:
                raise ValueError(
                    f"{copy.source}: {tuple(source.shape)} does not match "
                    f"{copy.destination}: {tuple(target.shape)}"
                )
            router_override = (
                router_weight_dtype is not None
                and copy.destination.endswith(".mlp.router")
                and source.dtype in (torch.bfloat16, torch.float32)
                and target.dtype == router_weight_dtype
            )
            if source.dtype != target.dtype and not router_override:
                raise ValueError(
                    f"Unsupported dtype conversion {copy.source}: "
                    f"{source.dtype} -> {target.dtype}"
                )
        used = {copy.source for copy in self.copies}
        # Every tensor of an expert projection is accounted for, the
        # activation scales of an NVFP4 recipe included: the FP8 forms
        # quantize their activations per row per call.
        used.update(
            key
            for conversion in self.conversions
            for _index, source in conversion.experts
            for projection in _projections(conversion.kind)
            for key in self._projection_keys(f"{source}.{projection}")
        )
        used.update(self._expert_keys)
        required = {
            key
            for key in self.weights.files
            if key.startswith(
                (
                    self.source_prefix + "embed_tokens.",
                    self.source_prefix + "norm.",
                    "lm_head.",
                )
            )
        }
        for i in range(c.num_hidden_layers):
            required.update(
                key
                for key in self.weights.files
                if key.startswith(f"{self.source_prefix}layers.{i}.")
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
            "checkpoint_expert_format": self.expert_format,
            "converted_expert_banks": len(self.conversions),
            "storage_dtype_bytes": {
                str(dtype): sum(
                    t.numel() * t.element_size()
                    for t in targets.values()
                    if t.dtype == dtype
                )
                for dtype in sorted({t.dtype for t in targets.values()}, key=str)
            },
            "checkpoint_storage_preserved": not self.conversions,
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
        # Construct only one factor at a time; imported meta parameters have no
        # initialized bytes. copy_ also supports non-addressable spill pools.
        for name, is_a in self.lora_initializers.items():
            if name in adopted:
                continue
            target = targets[name]
            value = torch.empty(tuple(target.shape), dtype=target.dtype, device="cpu")
            if is_a:
                # Per expert for a stacked [E, rank, in] factor: fan-in ``in``.
                nn.init.kaiming_uniform_(
                    value.view(-1, value.shape[-1]), a=math.sqrt(5)
                )
            else:
                nn.init.zeros_(value)
            target.copy_(value)
            if on_tensor is not None:
                on_tensor(name)
        grouped = {}
        for copy in self.copies:
            if copy.destination in adopted:
                continue
            grouped.setdefault(copy.destination, []).append(copy)
        total = len(self.copies) + sum(len(c.experts) for c in self.conversions)
        done = 0
        for destination, copies in grouped.items():
            target = targets[destination]
            joined = not (
                len(copies) == 1 and copies[0].start is None and copies[0].index is None
            )
            if joined:
                # Assemble joined gate/up and Q/K/V rows, and a layer's expert
                # bank, before publishing: each destination is written once,
                # even to an SSD pool. Scratch is one joined projection or one
                # component of one layer's bank.
                _check_rows(destination, copies, tuple(target.shape))

            filled = False

            def fill(staging, selected=tuple(copies), name=destination, before=done):
                nonlocal filled
                filled = True
                for index, copy in enumerate(selected, start=1):
                    source = self.weights.source.tensor(copy.source)
                    if copy.reciprocal:
                        if not bool(torch.isfinite(source).all() & (source > 0).all()):
                            raise ValueError(
                                f"Invalid NVFP4 global scale: {copy.source}"
                            )
                        source = source.reciprocal()
                    view = copy.view({name: staging})
                    view.copy_(source)
                    del source, view
                    self.weights.source.release_pages(copy.source)
                    if progress is not None:
                        progress(before + index, total, copy.source)

            if write_tensor is None:
                staging = (
                    torch.empty(tuple(target.shape), dtype=target.dtype, device="cpu")
                    if joined
                    else target
                )
                fill(staging)
                if staging is not target:
                    target.copy_(staging)
                del staging
            else:
                write_tensor(target, fill)
            done += len(copies)
            if on_tensor is not None:
                on_tensor(destination)
            if progress is not None and not filled:
                progress(done, total, copies[-1].source)
        # Banks that change form: decoded and quantized an expert at a time,
        # on the GPU when there is one, into host staging written once.
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        for conversion in self.conversions:
            stagings = {
                destination: torch.empty(
                    tuple(targets[destination].shape),
                    dtype=targets[destination].dtype,
                    device="cpu",
                )
                for destination in conversion.destinations
                if destination not in adopted
            }
            for index, prefix in conversion.experts:
                if stagings:
                    values = self._converted(prefix, conversion, device)
                    for destination, value in zip(
                        conversion.destinations, values, strict=True
                    ):
                        if destination in stagings:
                            stagings[destination][index].copy_(value)
                    del values
                done += 1
                if progress is not None:
                    progress(done, total, prefix)
            for destination, staging in stagings.items():
                targets[destination].copy_(staging)
                if on_tensor is not None:
                    on_tensor(destination)
            del stagings

    def close(self):
        self.weights.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

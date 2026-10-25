"""Family checkpoint reader and legacy module initialization."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from mlops.sequential_moe.precision.safetensors import SafeTensors

from .copy import copy_into
from .sources import Source


class Checkpoint:
    """A local Hugging Face checkpoint directory of one family."""

    #: The checkpoint's metadata (``config.json``) as the family's ``Config``.
    config_from_metadata: Callable[..., Any]
    #: Key prefixes the family's text decoder does not use.
    ignored_prefixes: tuple[str, ...] = ()

    def __init__(self, directory: str | Path, **overrides: Any) -> None:
        self.directory = Path(directory)
        metadata_path = self.directory / "config.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"{self.directory} has no config.json")
        self.metadata = json.loads(metadata_path.read_text())
        self.config = type(self).config_from_metadata(self.metadata, **overrides)
        self.reader = SafeTensors(self.directory)

    def sources(self, name: str, config: Any) -> Source:
        """The checkpoint tensors parameter ``name`` is made of."""
        raise NotImplementedError

    def keys(self) -> list[str]:
        return sorted(self.reader.info)

    def copy_chunk(self, name, shape, offset, destination) -> None:
        """Load a parameter's flat range into supplied storage, with bounded scratch."""
        source = self.sources(name, self.config)
        if tuple(source.read(self.reader, meta=True).shape) != tuple(shape):
            raise ValueError(f"checkpoint shape differs for {name}")
        source.copy_into(self.reader, offset, destination)

    def load_into(
        self,
        model: torch.nn.Module,
        *,
        copy: Callable[[torch.Tensor, torch.Tensor], None] = copy_into,
        rename: Callable[[str], str] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        """Fill every parameter of ``model`` from the checkpoint; returns what
        was written and the checkpoint keys nothing used. ``rename`` turns a
        parameter's name in the model given into the model's own (a harness
        wraps blocks in modules of their own before it loads). Existing LoRA
        factors are retained; ``initialize`` also resets those factors."""

        return self._load_into(
            model, reader=self.reader, copy=copy, rename=rename, progress=progress
        )

    def _load_into(
        self,
        model: torch.nn.Module,
        *,
        reader: SafeTensors,
        copy: Callable[[torch.Tensor, torch.Tensor], None] = copy_into,
        rename: Callable[[str], str] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:

        from workloads.models.shared.routed_lora import FACTORS

        used: set[str] = set()
        written: list[str] = []
        seen: dict[int, str] = {}
        parameters = list(model.named_parameters())
        for index, (name, parameter) in enumerate(parameters, start=1):
            parent, _, leaf = name.rpartition(".")
            if leaf in {*FACTORS, "lora_a", "lora_b"} and hasattr(
                model.get_submodule(parent), "lora_config"
            ):
                continue
            if id(parameter) in seen:
                continue
            seen[id(parameter)] = name
            if rename is not None:
                name = rename(name)
            source = self.sources(name, self.config)
            value = source.read(reader)
            copy(parameter, value)
            used.update(source.keys)
            for key in source.keys:
                reader.release_pages(key)
            del value
            written.append(name)
            if progress is not None:
                progress(index, len(parameters), name)
        unused = [
            key
            for key in sorted(reader.info)
            if key not in used and not key.startswith(self.ignored_prefixes)
        ]
        return {"written": written, "unused": unused}

    def initialize(self, model: torch.nn.Module) -> dict[str, Any]:
        """Fill pool-owned base weights, derived buffers and fresh LoRA factors.

        Suitable for ``Trainer.prepare(initialize=checkpoint.initialize)``.
        Base tensors are loaded directly; no random full-model initialization
        or temporary copy of the model is needed. Adapter-only state is not
        looked up in the base checkpoint. The temporary file mappings are closed
        before returning; this initializer can be called again without reopening
        the checkpoint object or requiring a caller's close/context manager.
        """
        from workloads.models.shared.routed_lora import FACTORS, reset_factors

        skipped = []
        for name, module in model.named_modules():
            if tuple(module.buffers(recurse=False)):
                reset = getattr(module, "reset_parameters", None)
                if not callable(reset):
                    raise ValueError(f"no initializer for derived buffers on {name!r}")
                reset()
            if not hasattr(module, "lora_config"):
                continue
            reset_lora = getattr(module, "reset_lora_parameters", None)
            if callable(reset_lora):
                reset_lora()
            elif all(hasattr(module, factor) for factor in FACTORS):
                reset_factors(module)
            else:
                raise ValueError(f"no adapter initializer on {name!r}")
            for leaf, _ in module.named_parameters(recurse=False):
                if leaf in {*FACTORS, "lora_a", "lora_b"}:
                    skipped.append(f"{name}.{leaf}" if name else leaf)
        with SafeTensors(self.directory) as reader:
            report = self._load_into(model, reader=reader)
        return {**report, "initialized_adapters": skipped}

    def close(self) -> None:
        self.reader.close()

    def __enter__(self) -> Checkpoint:
        return self

    def __exit__(self, *arguments: object) -> None:
        self.close()

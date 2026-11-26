"""Read packed expert payloads or decode bounded ranges into floating-point banks."""

from dataclasses import replace

import torch
from mlops.groupwise_experts import decode_groups

from shadowspill_workloads.models.shared.checkpoint import Checkpoint, Source, tensor

from .config import from_huggingface, storage_formats, validate_recipe

ROOT = "model.llm."


def source_key(name):
    if name == "embed.embed_norm.weight":
        return ROOT + "embed_norm.weight"
    if name == "lm_head.weight":
        return ROOT + "unembed.weight"
    if name == "output_query":
        return ROOT + "attn_res.output_query"
    if not name.startswith("layers."):
        return ROOT + name
    _, index, leaf = name.split(".", 2)
    if leaf == "attn_query":
        return ROOT + f"attn_res.attn_queries.{int(index) - 1}"
    if leaf == "mlp_query":
        return ROOT + f"attn_res.mlp_queries.{index}"
    if leaf.startswith(("mlp.w13", "mlp.w2")):
        bank, suffix = leaf[4:].split("_", 1) if "_" in leaf else (leaf[4:], "")
        leaf = "mlp.experts." + bank + "_weight" + ("_" + suffix if suffix else "")
    leaf = {
        "attn.q_scale": "attn.q_l2_clip.scale",
        "attn.rel_scale": "attn.rel_l2_clip.scale",
        "attn.proj": "attn.rel_logits_proj.proj",
    }.get(leaf, leaf)
    return ROOT + f"layers.{index}.{leaf}"


def _unembed(key, rows):
    return Source(
        (key,),
        lambda values: values[0][:rows],
        lambda reader, offset, dst: reader.copy_into(key, offset, dst),
    )


def _decoded(key, fmt, dtype=torch.float32):
    fp4 = fmt == "nvfp4"
    group = 16 if fp4 else 32
    keys = (key, key + "_scale")
    if fp4:
        keys += (key + "_amax", key + "_outer_scale_denominator")

    def geometry(values):
        experts, rows, groups = values[1].shape
        return experts, rows, groups * group

    def copy_values(values, start, destination):
        payload, scales = values[:2]
        _, rows, columns = geometry(values)
        elements_per_expert = rows * columns
        packed = payload.reshape(-1, group // 2 if fp4 else group)
        scales = scales.reshape(-1)
        written = 0
        while written < destination.numel():
            pos = start + written
            expert, inner = divmod(pos, elements_per_expert)
            count = min(
                destination.numel() - written, elements_per_expert - inner, 1 << 18
            )
            first, left = divmod(pos, group)
            stop = (pos + count + group - 1) // group
            factor = values[2][expert] / values[3] if fp4 else None
            decoded = decode_groups(
                packed[first:stop], scales[first:stop], factor, dtype=destination.dtype
            )
            destination[written : written + count].copy_(
                decoded.reshape(-1)[left : left + count]
            )
            written += count

    def assemble(values):
        out = values[0].new_empty(geometry(values), dtype=dtype)
        if not out.is_meta:
            copy_values(values, 0, out.view(-1))
        return out

    def copy_range(reader, start, destination):
        copy_values([reader.tensor(k) for k in keys], start, destination)

    return Source(keys, assemble, copy_range)


class InklingCheckpoint(Checkpoint):
    config_from_metadata = staticmethod(from_huggingface)
    ignored_prefixes = ("model.visual.",)

    def __init__(self, directory, **overrides):
        super().__init__(directory, **overrides)
        try:
            from shadowspill_workloads.initialization.hub import read_config

            recipe_path = self.directory / "hf_quant_config.json"
            self.config = replace(
                self.config,
                expert_formats=storage_formats(self.reader.info, self.config),
            )
            validate_recipe(
                read_config(recipe_path) if recipe_path.is_file() else None,
                self.config.expert_formats,
                self.reader.info,
            )
        except Exception:
            self.close()
            raise

    def model_config(self, *, lora: bool):
        return self.config.resolve_experts(lora=lora)

    def sources(self, name, config, *, destination_dtype=None):
        key = source_key(name)
        if name == "lm_head.weight":
            return _unembed(key, config.vocab_size)
        if name.endswith((".mlp.w13", ".mlp.w2")):
            packed = (
                config.expert_storage == "packed"
                if destination_dtype is None
                else destination_dtype in (torch.uint8, torch.float8_e4m3fn)
            )
            fmt = config.expert_formats[int(name.split(".")[1])]
            if fmt != "dense" and not packed:
                dtype = destination_dtype or (
                    torch.bfloat16 if config.expert_storage == "bf16" else torch.float32
                )
                return _decoded(key, fmt, dtype)
        return tensor(key)

    def _source_for_destination(self, name, dtype):
        return self.sources(name, self.config, destination_dtype=dtype)

"""Plan expert-bank copies and convert one expert at a time."""

from dataclasses import dataclass

import torch
from mlops.sequential_moe.precision.storage import materialize

from .tensors import Copy


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


def projection_keys(reader, prefix):
    """Every checkpoint tensor of one projection (``prefix.weight``, its
    scales, any activation-scale metadata)."""
    if reader._keys is None:
        reader._keys = {}
        for key in reader.weights.files:
            reader._keys.setdefault(key.rsplit(".", 1)[0], []).append(key)
    return tuple(reader._keys.get(prefix, ()))


def plan_experts(reader, name, c):
    """The copies, or else the conversions, that fill one layer's banks."""
    h, d = c.moe_intermediate_size, c.hidden_size
    reader._expert_shapes = {
        "gate_proj": (h, d),
        "up_proj": (h, d),
        "down_proj": (d, h),
    }
    source = reader.source_prefix + name + ".experts"
    experts = [(index, f"{source}.{index}") for index in range(c.num_local_experts)]
    for kind in ("w13", "w2"):
        specs = [
            reader.weights.projection(
                f"{prefix}.{projection}", reader._expert_shapes[projection], meta=True
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
            reader.conversions.append(
                Conversion(name, kind, c.expert_storage, tuple(experts))
            )
            continue
        for index, prefix in experts:
            copy_expert(reader, name, kind, index, prefix, c.expert_storage, h)


def copy_expert(reader, name, kind, index, prefix, storage, rows):
    """One expert's tensors as stored: a dense weight's rows into the bf16
    bank, or an FP8 weight's E4M3 rows and 128x128 block scales into a
    block-scaled bank's components; ``w13`` gate rows, then up rows."""
    scale_key = (
        "weight_scale" if reader.weights.encoding == "mixed" else "weight_scale_inv"
    )
    parts = (
        (("", "weight", rows),)
        if storage == "bf16"
        else (("._data", "weight", rows), ("._scale", scale_key, rows // 128))
    )
    for position, projection in enumerate(_projections(kind)):
        reader._expert_keys.update(projection_keys(reader, f"{prefix}.{projection}"))
        for component, key, count in parts:
            start, stop = (
                (position * count, (position + 1) * count)
                if kind == "w13"
                else (None, None)
            )
            reader.copies.append(
                Copy(
                    f"{name}.{kind}{component}",
                    f"{prefix}.{projection}.{key}",
                    start,
                    stop,
                    index=index,
                )
            )


def convert_expert(reader, prefix, conversion, device):
    """One expert's ``w13`` or ``w2`` decoded from the checkpoint and taken
    to ``conversion.storage``'s components, in its destinations' order."""
    from mlops.expert_parallel.quack.parameters.fp8 import QuackFP8Weight
    from mlops.kernels.block_fp8 import quantize_blocks

    values = []
    for projection in _projections(conversion.kind):
        name = f"{prefix}.{projection}"
        parts, spec = reader.weights.projection(name, reader._expert_shapes[projection])
        values.append(
            materialize([part.to(device) for part in parts], spec, torch.float32)
        )
        del parts
        for key in projection_keys(reader, name):
            reader.weights.source.release_pages(key)
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

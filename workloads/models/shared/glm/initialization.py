"""Physical tensor roles and bounded stage-trace initialization."""

import math

import torch
from mlops.sequential_moe.precision.linear import FrozenLinear

from .moe import MoE


def _roles(model, tensors):
    roles = {}
    for name, module in model.named_modules():
        if isinstance(module, FrozenLinear):
            if module.spec.format != "fp8":
                raise ValueError(
                    "random checkpoint-shaped initialization currently supports "
                    f"dense and block-FP8 storage, not {module.spec.format}"
                )
            roles[name + ".components.0"] = "fp8"
            roles[name + ".components.1"] = "fp8_scale"
        if isinstance(module, MoE) and module.config.expert_storage != "bf16":
            if module.config.expert_storage != "fp8_block":
                raise ValueError(
                    "random checkpoint-shaped experts require bf16 or fp8_block"
                )
            for bank in ("w13", "w2"):
                roles[f"{name}.{bank}._data"] = "fp8"
                roles[f"{name}.{bank}._scale"] = "fp8_scale"
    for name, tensor in tensors.items():
        if name in roles:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if "lora" in leaf:
            roles[name] = "lora_a" if leaf.endswith("_a") else "zero"
        elif leaf in (
            "base",
            "bias",
            "A_log",
            "e_score_correction_bias",
            "index_kpool_compress_ape",
        ):
            roles[name] = "zero"
        elif leaf in ("scale", "index_kpool_compress_gate") or (
            leaf == "weight" and tensor.ndim == 1
        ):
            roles[name] = "one"
        elif leaf == "dt_bias":
            roles[name] = "dt_bias"
        else:
            roles[name] = "normal"
        if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(
                f"unsupported random initializer for {name}: {tensor.dtype}"
            )
    return roles


@torch.no_grad()
def fill_random_tensor(value, *, role, tensor_seed, std, device):
    """Fill one physical tensor using the checkpoint-shaped random policy."""
    device = torch.device(device)
    if role in ("zero", "one", "fp8_scale"):
        value.fill_({"zero": 0.0, "one": 1.0, "fp8_scale": std / 64}[role])
        return
    generator = torch.Generator(device=device).manual_seed(tensor_seed)
    flat = value.view(-1)
    # Bound FP32 generation scratch independently of a multi-GiB bank.
    for offset in range(0, flat.numel(), 1 << 20):
        count = min(1 << 20, flat.numel() - offset)
        sample = torch.empty(count, dtype=torch.float32, device=device)
        if role == "lora_a":
            bound = value.shape[-1] ** -0.5
            sample.uniform_(-bound, bound, generator=generator)
        elif role == "dt_bias":
            sample.uniform_(math.log(1e-3), math.log(1e-1), generator=generator)
            dt = sample.exp()
            sample = dt + torch.log(-torch.expm1(-dt))
        else:
            sample.normal_(std=64 if role == "fp8" else std, generator=generator)
            if role == "fp8":
                sample.clamp_(-448, 448)
        # Cast on the generation device before transferring a bounded
        # chunk. The destination remains checkpoint-precision CPU spill.
        flat[offset : offset + count].copy_(sample.to(dtype=value.dtype))

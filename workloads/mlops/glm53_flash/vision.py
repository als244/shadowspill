"""GLM vision encoder: BF16 projections, FP32 normalization and axial RoPE.

Image geometry is prepared on the CPU before capture. Attention uses PyTorch
SDPA on each frame; it never reads device metadata back to Python.
"""

from dataclasses import dataclass, fields
from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F

from .common import RMSNorm


@dataclass(frozen=True)
class VisionConfig:
    depth: int = 24
    hidden_size: int = 1024
    num_heads: int = 16
    intermediate_size: int = 4096
    out_hidden_size: int = 4096
    projection_intermediate_size: int = 10240
    in_channels: int = 3
    patch_size: int = 14
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    attention_bias: bool = True
    rms_norm_eps: float = 1e-5
    swiglu_limit: float = 10.0
    rope_theta: float = 10000.0
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if (
            min(
                self.depth,
                self.hidden_size,
                self.intermediate_size,
                self.out_hidden_size,
                self.projection_intermediate_size,
                self.in_channels,
                self.num_heads,
                self.patch_size,
                self.temporal_patch_size,
                self.spatial_merge_size,
            )
            < 1
        ):
            raise ValueError("Vision dimensions must be positive")
        if self.hidden_size % (4 * self.num_heads):
            raise ValueError(
                "Axial rotary embedding requires head width divisible by 4"
            )
        if self.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Vision currently supports FP32/BF16 computation")

    @classmethod
    def from_huggingface(cls, metadata, **overrides):
        values = metadata.get("vision_config", metadata)
        if values.get("hidden_act", "silu") != "silu":
            raise ValueError("GLM vision requires SiLU")
        if values.get("attention_dropout", 0.0) != 0.0:
            raise ValueError("This checkpoint path requires zero attention dropout")
        rope = values.get("rope_parameters") or {}
        if rope.get("rope_type", "axial") != "axial":
            raise ValueError("GLM vision requires axial RoPE")
        names = {field.name for field in fields(cls)} - {"dtype"}
        config = {key: value for key, value in values.items() if key in names}
        config["rope_theta"] = rope.get("rope_theta", 10000.0)
        config.update(overrides)
        return cls(**config)


def prepare_geometry(grid_thw, *, merge_size=2):
    """Build CPU metadata once from processor geometry, before capture.

    Patches are ordered by spatial merge block. Each temporal frame is a
    separate noncausal attention segment, as in the published vision model.
    """
    if merge_size < 1:
        raise ValueError("merge_size must be positive")
    grid = torch.as_tensor(grid_thw, dtype=torch.int64)
    if grid.device.type != "cpu":
        raise ValueError("Prepare vision geometry from CPU metadata")
    if grid.ndim != 2 or grid.shape[1] != 3 or not len(grid):
        raise ValueError("grid_thw must contain [time, height, width] rows")
    positions, boundaries = [], [0]
    for time, height, width in grid.tolist():
        if min(time, height, width) < 1 or height % merge_size or width % merge_size:
            raise ValueError(
                "Vision grids must be positive and divisible by merge_size"
            )
        row, column = torch.meshgrid(
            torch.arange(height), torch.arange(width), indexing="ij"
        )
        shape = (height // merge_size, merge_size, width // merge_size, merge_size)
        positions.append(
            torch.stack(
                [
                    axis.reshape(shape).transpose(1, 2).flatten()
                    for axis in (row, column)
                ],
                dim=-1,
            ).repeat(time, 1)
        )
        for _ in range(time):
            boundaries.append(boundaries[-1] + height * width)
    return torch.cat(positions), tuple(boundaries)


def rotary_frequencies(head_width, theta):
    """Match HF's FP32 CPU initialization once, before capture.

    Computing pow/reciprocal again on the GPU can change the last bit. Deep BF16
    vision blocks amplify those changes, so keep the initialized constants.
    """
    spatial_width = head_width // 2
    inverse = 1.0 / (
        theta
        ** (
            torch.arange(0, spatial_width, 2, dtype=torch.float32, device="cpu")
            / spatial_width
        )
    )
    return tuple(inverse.tolist())


def rotary_embeddings(positions, inverse_frequencies):
    # Keep the literal in host storage; the captured graph owns its device copy.
    inverse = torch.tensor(inverse_frequencies, dtype=torch.float32, device="cpu").to(
        positions.device
    )
    angles = positions.float().unsqueeze(-1) * inverse
    cos, sin = angles.cos().flatten(1), angles.sin().flatten(1)
    return torch.cat((cos, cos), dim=-1), torch.cat((sin, sin), dim=-1)


def rotate(value, cos, sin):
    left, right = value.float().chunk(2, dim=-1)
    rotated = torch.cat((-right, left), dim=-1)
    return (value.float() * cos[:, None] + rotated * sin[:, None]).to(value.dtype)


def projection(inputs, outputs, config, device, *, bias):
    return nn.Linear(inputs, outputs, bias=bias, dtype=config.dtype, device=device)


class VisionMLP(nn.Module):
    def __init__(self, config, device, *, width=None, intermediate=None, bias=None):
        super().__init__()
        width = config.hidden_size if width is None else width
        intermediate = (
            config.intermediate_size if intermediate is None else intermediate
        )
        bias = config.attention_bias if bias is None else bias
        self.gate_proj = projection(width, intermediate, config, device, bias=bias)
        self.up_proj = projection(width, intermediate, config, device, bias=bias)
        self.down_proj = projection(intermediate, width, config, device, bias=bias)
        self.limit = config.swiglu_limit

    def forward(self, x):
        gate = self.gate_proj(x).clamp(max=self.limit)
        up = self.up_proj(x).clamp(min=-self.limit, max=self.limit)
        return self.down_proj(F.silu(gate) * up)


class VisionAttention(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        d = config.hidden_size
        self.heads = config.num_heads
        self.qkv = projection(d, 3 * d, config, device, bias=config.attention_bias)
        self.proj = projection(d, d, config, device, bias=config.attention_bias)
        self.q_norm = RMSNorm(d // self.heads, config, device)
        self.k_norm = RMSNorm(d // self.heads, config, device)

    def forward(self, x, boundaries, cos, sin):
        q, k, v = self.qkv(x).reshape(len(x), 3, self.heads, -1).unbind(1)
        q, k = rotate(self.q_norm(q), cos, sin), rotate(self.k_norm(k), cos, sin)
        outputs = []
        for start, stop in pairwise(boundaries):
            pieces = [t[start:stop].transpose(0, 1).unsqueeze(0) for t in (q, k, v)]
            out = F.scaled_dot_product_attention(
                *pieces, dropout_p=0.0, is_causal=False
            )
            outputs.append(out.squeeze(0).transpose(0, 1).reshape(stop - start, -1))
        return self.proj(torch.cat(outputs))


class VisionBlock(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, config, device)
        self.norm2 = RMSNorm(config.hidden_size, config, device)
        self.attn = VisionAttention(config, device)
        self.mlp = VisionMLP(config, device)

    def forward(self, x, boundaries, cos, sin):
        x = x + self.attn(self.norm1(x), boundaries, cos, sin)
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.patch_shape = (
            config.in_channels,
            config.temporal_patch_size,
            config.patch_size,
            config.patch_size,
        )
        self.proj = nn.Conv3d(
            config.in_channels,
            config.hidden_size,
            self.patch_shape[1:],
            stride=self.patch_shape[1:],
            device=device,
            dtype=config.dtype,
        )

    def forward(self, pixels):
        return self.proj(
            pixels.reshape(-1, *self.patch_shape).to(self.proj.weight.dtype)
        ).flatten(1)


class PatchMerger(VisionMLP):
    def __init__(self, config, device):
        super().__init__(
            config,
            device,
            width=config.out_hidden_size,
            intermediate=config.projection_intermediate_size,
            bias=False,
        )
        self.proj = projection(
            config.out_hidden_size, config.out_hidden_size, config, device, bias=False
        )
        self.post_projection_norm = nn.LayerNorm(
            config.out_hidden_size, device=device, dtype=config.dtype
        )

    def forward(self, x):
        x = F.gelu(self.post_projection_norm(self.proj(x)))
        return super().forward(x)


class VisionModel(nn.Module):
    def __init__(self, config, *, device="cuda"):
        super().__init__()
        self.config = config
        self.inverse_frequencies = rotary_frequencies(
            config.hidden_size // config.num_heads, config.rope_theta
        )
        self.patch_embed = PatchEmbed(config, device)
        self.blocks = nn.ModuleList(
            VisionBlock(config, device) for _ in range(config.depth)
        )
        self.post_layernorm = RMSNorm(config.hidden_size, config, device)
        self.downsample = nn.Conv2d(
            config.hidden_size,
            config.out_hidden_size,
            config.spatial_merge_size,
            stride=config.spatial_merge_size,
            device=device,
            dtype=config.dtype,
        )
        self.merger = PatchMerger(config, device)

    def forward(self, pixels, positions, boundaries):
        if positions.shape != (pixels.shape[0], 2):
            raise ValueError("One 2D position is required per input patch")
        if boundaries[0] != 0 or boundaries[-1] != pixels.shape[0]:
            raise ValueError("Vision boundaries must cover every patch")
        hidden = self.patch_embed(pixels)
        c = self.config
        cos, sin = rotary_embeddings(positions, self.inverse_frequencies)
        for block in self.blocks:
            hidden = block(hidden, boundaries, cos, sin)
        hidden = self.post_layernorm(hidden)
        hidden = hidden.reshape(
            -1, c.spatial_merge_size, c.spatial_merge_size, c.hidden_size
        )
        hidden = self.downsample(hidden.permute(0, 3, 1, 2)).flatten(1)
        return self.merger(hidden)


@dataclass(frozen=True)
class ImageInputs:
    """Preprocessed patch values and their locations in a flattened text input."""

    pixels: torch.Tensor
    positions: torch.Tensor
    boundaries: tuple[int, ...]
    token_indices: torch.Tensor


torch.export.register_dataclass(
    ImageInputs, serialized_type_name="workloads.mlops.glm53_flash.ImageInputs"
)


def prepare_images(pixels, grid_thw, token_ids, *, image_token_id=154854, merge_size=2):
    """Bind HF-processed still images to their expanded placeholder tokens.

    Inputs are CPU tensors from preprocessing. The returned tensor leaves are
    ordinary model inputs, so the caller/planner controls their device placement.
    No image resizing, token expansion or device synchronization happens here.
    """
    if any(t.device.type != "cpu" for t in (pixels, token_ids)):
        raise ValueError("Prepare image inputs from CPU preprocessing outputs")
    positions, boundaries = prepare_geometry(grid_thw, merge_size=merge_size)
    if pixels.shape[0] != positions.shape[0]:
        raise ValueError("Patch count does not match image grid geometry")
    indices = (token_ids.reshape(-1) == image_token_id).nonzero().flatten()
    expected = positions.shape[0] // merge_size**2
    if indices.numel() != expected:
        raise ValueError(
            f"Image placeholders ({indices.numel()}) do not match features ({expected})"
        )
    return ImageInputs(pixels, positions, boundaries, indices)

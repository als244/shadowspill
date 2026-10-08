"""CPU design probe. This is not the production MLOps LoRA API.

Exercise post-construction conversion, shared module/parameter identity and
meta initialization without changing the checkouts used by qualification.
"""
from dataclasses import dataclass
from fnmatch import fnmatchcase
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 32
    alpha: float = 32.0
    dropout: float = 0.0
    factor_dtype: torch.dtype | None = None

    def __post_init__(self):
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.factor_dtype not in (None, torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("factor_dtype must be a floating compute dtype")

    @property
    def scale(self):
        return self.alpha / self.rank


class LoRALinear(nn.Module):
    """Keep the original module and its initialization; add two small factors."""
    def __init__(self, base: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.base = base
        self.lora_config = config
        self.in_features = base.in_features
        self.out_features = base.out_features
        options = dict(device=base.weight.device, dtype=config.factor_dtype or base.weight.dtype)
        self.lora_a = nn.Parameter(torch.empty(config.rank, base.in_features, **options))
        self.lora_b = nn.Parameter(torch.empty(base.out_features, config.rank, **options))
        self.reset_parameters()
        self.train(base.training)

    def reset_parameters(self):
        # A normal module traversal also visits base and initializes its own
        # state. Conversion of an initialized model never resets base weights.
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, inputs):
        hidden = F.dropout(inputs, self.lora_config.dropout, self.training)
        low_rank = F.linear(hidden, self.lora_a.to(inputs.dtype))
        update = F.linear(low_rank, self.lora_b.to(inputs.dtype))
        return self.base(inputs) + update * self.lora_config.scale


def apply_lora(model, config, *, targets, trainable_base=(), converters=None):
    """Convert once, outside execution; extension functions own expert details."""
    targets, trainable_base = tuple(targets), tuple(trainable_base)
    if not targets:
        raise ValueError("targets must not be empty")
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("model already contains LoRA modules")
    modules = dict(model.named_modules(remove_duplicate=False))
    parameters = dict(model.named_parameters(remove_duplicate=False))
    matched = {}
    for pattern in targets:
        names = [name for name in modules if fnmatchcase(name, pattern)]
        if not names:
            raise ValueError(f"target {pattern!r} matched no modules")
        matched.update((name, modules[name]) for name in names)
    for name in matched:
        if any(name.startswith(parent + '.') for parent in matched if parent):
            raise ValueError("select a module or its children, not both")
    trainable_ids = set()
    for pattern in trainable_base:
        names = [name for name in parameters if fnmatchcase(name, pattern)]
        if not names:
            raise ValueError(f"trainable_base {pattern!r} matched no parameters")
        trainable_ids.update(id(parameters[name]) for name in names)
    converters = {nn.Linear: LoRALinear, **(converters or {})}
    selected = {}
    for name, module in matched.items():
        converter = next((converters[t] for t in type(module).__mro__ if t in converters), None)
        if converter is None:
            raise TypeError(f"no LoRA conversion for {name!r}: {type(module).__name__}")
        selected[id(module)] = (module, converter)
    # Validate all selections before allocating factors or changing the model.
    replacements = {identity: factory(module, config) for identity, (module, factory) in selected.items()}
    for parameter in parameters.values():
        parameter.requires_grad_(id(parameter) in trainable_ids)
    # All aliases of a selected module point to the same replacement.
    for name, module in modules.items():
        if id(module) not in replacements:
            continue
        replacement = replacements[id(module)]
        if not name:
            model = replacement
        else:
            parent, _, leaf = name.rpartition('.')
            setattr(modules[parent], leaf, replacement)
    return model


class ExpertLoRA(nn.Module):
    """Independent PyTorch oracle for a packed SwiGLU expert bank.

    Routing is supplied so all variants use identical assignments. Production
    will execute grouped kernels, not this all-rows/per-expert reference loop.
    """
    def __init__(self, gate_up, down, config):
        super().__init__()
        self.gate_up = gate_up
        self.down = down
        self.lora_config = config
        experts, width, packed = gate_up.shape
        options = dict(device=gate_up.device, dtype=config.factor_dtype or gate_up.dtype)
        self.lora_gate_up_a = nn.Parameter(torch.empty(experts, width, config.rank, **options))
        self.lora_gate_up_b = nn.Parameter(torch.empty(experts, config.rank, packed, **options))
        self.lora_down_a = nn.Parameter(torch.empty(experts, packed // 2, config.rank, **options))
        self.lora_down_b = nn.Parameter(torch.empty(experts, config.rank, width, **options))
        self.reset_parameters()
        gate_up.requires_grad_(False)
        down.requires_grad_(False)

    def reset_parameters(self):
        for prefix in ('gate_up', 'down'):
            a, b = getattr(self, f'lora_{prefix}_a'), getattr(self, f'lora_{prefix}_b')
            nn.init.normal_(a, std=a.shape[1] ** -0.5)
            nn.init.zeros_(b)

    def forward(self, x, ids, routing_weights):
        result = torch.zeros_like(x)
        scale = self.lora_config.scale
        for expert in range(self.gate_up.shape[0]):
            h13 = x @ self.gate_up[expert]
            h13 = h13 + scale * (x @ self.lora_gate_up_a[expert]) @ self.lora_gate_up_b[expert]
            gate, up = h13.chunk(2, -1)
            activated = F.silu(gate) * up
            y = activated @ self.down[expert]
            y = y + scale * (activated @ self.lora_down_a[expert]) @ self.lora_down_b[expert]
            coefficient = (routing_weights * (ids == expert)).sum(-1)
            result = result + coefficient[:, None] * y
        return result

"""Independent eager weight VJPs at FP32, before autograd's BF16 leaf cast.

The reference model supplies ordinary PyTorch activations and their cotangents.
These elementary equations retain weight reductions at the precision requested
by the service. No engine capture, gradient rewrite, provider kernel or optimizer
implementation is used. Ordinary autograd still computes activation derivatives.
"""

import torch

from shadowspill_workloads.models.shared.decoder import RMSNorm


class WeightGradients:
    def __init__(self, model):
        self.values = {}
        self.handles = [
            module.register_forward_hook(self._forward(name))
            for name, module in model.named_modules()
            if any(p.requires_grad for p in module.parameters(recurse=False))
        ]

    def _forward(self, name):
        def forward(module, args, output):
            if not output.requires_grad:
                return
            inputs = args[0].detach()

            def backward(gradient):
                x, dy = inputs, gradient.detach()
                if isinstance(module, torch.nn.Embedding):
                    values = {"weight": torch.zeros_like(module.weight).float()}
                    values["weight"].index_add_(
                        0, x.reshape(-1), dy.reshape(-1, dy.shape[-1]).float()
                    )
                elif isinstance(module, RMSNorm):
                    assert not module.zero_centered
                    source = x.float()
                    normalized = (
                        source
                        * torch.rsqrt(
                            source.square().mean(-1, keepdim=True) + module.epsilon
                        )
                    ).to(x.dtype)
                    # The multiply is in activation precision; its reduction
                    # remains FP32 instead of rounding once more at the leaf.
                    values = {
                        "weight": (dy * normalized)
                        .float()
                        .reshape(-1, x.shape[-1])
                        .sum(0)
                    }
                else:
                    x, dy = x.reshape(-1, x.shape[-1]), dy.reshape(-1, dy.shape[-1])
                    assert module.bias is None
                    values = {"weight": dy.float().T @ x.float()}
                    if hasattr(module, "lora_a"):
                        inner = torch.nn.functional.linear(
                            x, module.lora_a.detach().to(x.dtype)
                        )
                        scaled = dy * module.lora_config.scale
                        hidden = scaled @ module.lora_b.detach().to(x.dtype)
                        values.update(
                            lora_a=hidden.float().T @ x.float(),
                            lora_b=scaled.float().T @ inner.float(),
                        )
                    else:
                        assert isinstance(module, torch.nn.Linear)
                for local, value in values.items():
                    if not getattr(module, local).requires_grad:
                        continue
                    key = f"{name}.{local}" if name else local
                    self.values[key] = self.values.get(key, 0) + value

            output.register_hook(backward)

        return forward

    def close(self):
        for handle in self.handles:
            handle.remove()

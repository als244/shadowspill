"""Shared BF16 rank-32 factors, empty experts and an independent CPU oracle.

Calls the Sonic provider directly, including for CPU fake capture; this does
not broaden its automatic device dispatch or qualify a full model family.
"""

import torch


class Network(torch.nn.Module):
    def __init__(self, *, provider="sonicmoe", device="cpu"):
        super().__init__()
        self.provider = provider
        generator = torch.Generator().manual_seed(62841)
        shapes = {
            "router": (4, 64),
            "up": (4, 128, 64),
            "down": (4, 64, 64),
            "up_a": (4, 32, 64),
            "up_b": (4, 128, 32),
            "down_a": (4, 32, 64),
            "down_b": (4, 64, 32),
        }
        for name, shape in shapes.items():
            dtype = torch.float32 if name == "router" else torch.bfloat16
            value = (torch.randn(shape, generator=generator) * 0.04).to(dtype)
            # A wide routing margin and two empty experts isolate accumulation
            # from the separately open real-model routing/numerical question.
            if name == "router":
                value[0].fill_(0.06)
                value[1].fill_(0.03)
                value[2].fill_(-0.03)
                value[3].fill_(-0.06)
            self.register_parameter(
                name, torch.nn.Parameter(value.to(device), requires_grad="_" in name)
            )

    def experts(self, hidden):
        if self.provider == "sonicmoe":
            from mlops.providers.sonicmoe.lora import expert_lora

            return expert_lora(
                hidden,
                hidden,
                self.router,
                self.up,
                self.down,
                self.up_a,
                self.up_b,
                self.down_a,
                self.down_b,
                top_k=2,
                routing_mode="topk_then_softmax",
                scale=1.75,
            )[0]
        assert self.provider == "independent"
        flat = hidden.reshape(-1, hidden.shape[-1])
        logits = flat.float() @ self.router.to(flat.dtype).float().T
        chosen, indices = torch.topk(logits, 2, dim=-1)
        weights = torch.softmax(chosen, dim=-1)
        contributions = []
        for expert in range(4):
            low = flat @ self.up_a[expert].T
            packed = flat @ self.up[expert].T + low @ (1.75 * self.up_b[expert]).T
            gate, up = packed.chunk(2, dim=-1)
            activated = (torch.nn.functional.silu(gate.float()) * up.float()).to(
                flat.dtype
            )
            low = activated @ self.down_a[expert].T
            output = (
                activated @ self.down[expert].T + low @ (1.75 * self.down_b[expert]).T
            )
            selected = ((indices == expert) * weights).sum(dim=-1)
            contributions.append(output.float() * selected[:, None])
        return (sum(contributions) + flat.float()).to(hidden.dtype).reshape_as(hidden)

    def forward(self, hidden):
        hidden = self.experts(hidden).sin()
        return self.experts(hidden).sin()


def examples(step=0, batch=1):
    generator = torch.Generator().manual_seed(2903 + step)
    return [
        (
            (
                (torch.rand((batch, length, 64), generator=generator) * 0.5 + 0.2).to(
                    torch.bfloat16
                ),
            ),
            {"labels": torch.randn((batch, length, 64), generator=generator) * 0.25},
        )
        for length in (65, 48)
    ]


def objective(model, hidden, *, labels):
    result = model(hidden)
    return (result.float() - labels).square().mean() / 2, {
        "mean": result.float().mean()
    }

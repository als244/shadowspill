"""Independent FP32 expert/router weight reductions for the Qwen3 CPU oracle.

Re-evaluate ordinary per-expert equations with the observed output cotangent.
Linear output hooks retain FP32 parameter products before BF16 autograd leaf
rounding. Activation derivatives still use ordinary eager autograd. No compiler,
provider backward, destination mapper or optimizer code computes these products.
"""

import torch
from torch.nn import functional as F

from shadowspill_workloads.models.shared.qwen_moe.experts import MoE
from shadowspill_workloads.models.shared.qwen_moe.model import QwenMoE
from workload_tests._llama_oracle import WeightGradients as DenseGradients


def experts(module, hidden, residual, *, observe=None):
    """Explicit normalized top-k, per-expert projections and residual combine."""
    assert module.experts is None and module.shared is None
    x = hidden.reshape(-1, hidden.shape[-1])

    def linear(x, weight, name, expert=None, *, wide=False):
        output = F.linear(x.float(), weight.float()) if wide else F.linear(x, weight)
        if observe is not None and output.requires_grad:
            observe(name, expert, x, output)
        return output

    logits = linear(x, module.router.weight.detach(), "router.weight", wide=True)
    chosen, indices = logits.topk(module.config.top_k, dim=-1)
    coefficients = chosen.softmax(-1)
    contributions = []
    lora = getattr(module, "lora_config", None)

    def project(x, name, prefix, expert):
        result = linear(x, getattr(module, name)[expert].detach(), name, expert)
        if lora is not None:
            a, b = f"lora_{prefix}_a", f"lora_{prefix}_b"
            inner = linear(
                x, getattr(module, a)[expert].detach().to(x.dtype), a, expert
            )
            result = result + lora.scale * linear(
                inner, getattr(module, b)[expert].detach().to(x.dtype), b, expert
            )
        return result

    for expert in range(module.config.n_experts):
        gate, up = project(x, "w13", "gate_up", expert).chunk(2, dim=-1)
        activated = F.silu(gate.float()).to(gate.dtype) * up
        output = project(activated, "w2", "down", expert)
        selected = (coefficients * (indices == expert)).sum(-1)
        contributions.append(output.float() * selected[:, None])
    return (
        (sum(contributions) + residual.reshape_as(x).float())
        .to(hidden.dtype)
        .reshape_as(hidden)
    )


class WeightGradients(DenseGradients):
    def __init__(self, model):
        self.names = {id(module): name for name, module in model.named_modules()}
        self.parameters = dict(model.named_parameters())
        super().__init__(model)
        if isinstance(model, QwenMoE) and model.embed.weight.requires_grad:
            # This model calls functional embedding, bypassing embed.__call__.
            # Observe its output at the first block, before any computation.
            self.handles.extend(
                (
                    model.register_forward_pre_hook(self._tokens, with_kwargs=True),
                    model.blocks[0].register_forward_pre_hook(self._embedding),
                )
            )

    def _tokens(self, module, args, kwargs):
        self.tokens = (args[0] if args else kwargs["tokens"]).detach()

    def _embedding(self, module, args):
        output, tokens = args[0], self.tokens
        if not output.requires_grad:
            return

        def backward(gradient):
            value = self.values.setdefault(
                "embed.weight",
                torch.zeros_like(self.parameters["embed.weight"], dtype=torch.float32),
            )
            value.index_add_(
                0,
                tokens.reshape(-1),
                gradient.detach().reshape(-1, gradient.shape[-1]).float(),
            )

        output.register_hook(backward)

    def _observe(self, name, expert, inputs, output):
        parameter = self.parameters[name]
        if not parameter.requires_grad or not output.requires_grad:
            return
        x = inputs.detach().reshape(-1, inputs.shape[-1]).float()

        def backward(gradient):
            product = gradient.detach().reshape(-1, gradient.shape[-1]).float().T @ x
            value = self.values.setdefault(
                name, torch.zeros_like(parameter, dtype=torch.float32)
            )
            if expert is None:
                value.add_(product)
            else:
                value[expert].add_(product)

        output.register_hook(backward)

    def router(self, module, hidden, logits):
        """Observe the independent auxiliary formula's separate router product."""
        self._observe(f"{self.names[id(module)]}.router.weight", None, hidden, logits)

    def _forward(self, name):
        ordinary = super()._forward(name)

        def forward(module, args, output):
            if not isinstance(module, MoE):
                return ordinary(module, args, output)
            if not output[0].requires_grad:
                return
            hidden, residual = (value.detach() for value in args)

            def backward(gradient):
                with torch.enable_grad():
                    reference = experts(
                        module,
                        hidden.requires_grad_(True),
                        residual,
                        observe=lambda local, expert, x, y: self._observe(
                            f"{name}.{local}" if name else local, expert, x, y
                        ),
                    )
                    reference.backward(gradient.detach())

            output[0].register_hook(backward)

        return forward

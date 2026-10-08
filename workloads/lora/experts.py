"""Dedicated expert replacements with each architecture's existing call signature."""

from mlops.lora.experts import expert_lora, initialize_factors, reset_factors
from torch import nn

from workloads.mlops._qwen_moe.experts import MoE as QwenMoE
from workloads.mlops.olmoe import MoE as OLMoE
from workloads.pytorch.olmoe import MoE as PyTorchOLMoE

from .reference import expert_lora as reference_expert_lora


class OLMoELoRA(OLMoE):
    def __init__(self, base, config):
        nn.Module.__init__(self)
        self.config, self.lora_config = base.config, config
        self.experts = None
        self.router = base.router
        self.w13_experts, self.w2_experts = base.w13_experts, base.w2_experts
        initialize_factors(self, self.w13_experts, self.w2_experts, config)
        self.train(base.training)

    def reset_parameters(self):
        OLMoE.reset_parameters(self)
        reset_factors(self)

    def _run(self, hidden, residual, function):
        return function(
            hidden,
            residual,
            self.router.weight.T,
            self.w13_experts,
            self.w2_experts,
            self.lora_gate_up_a,
            self.lora_gate_up_b,
            self.lora_down_a,
            self.lora_down_b,
            top_k=self.config.top_k,
            routing_mode="softmax_then_topk",
            scale=self.lora_config.scale,
        )

    def forward(self, hidden, residual, *, return_metrics=False):
        values = self._run(hidden, residual, expert_lora)
        return values if return_metrics else values[:2]


class PyTorchOLMoELoRA(OLMoELoRA):
    def reset_parameters(self):
        PyTorchOLMoE.reset_parameters(self)
        reset_factors(self)

    def forward(self, hidden, residual):
        return self._run(hidden, residual, reference_expert_lora)[:2]


class QwenMoELoRA(QwenMoE):
    def __init__(self, base, config):
        nn.Module.__init__(self)
        self.config, self.lora_config = base.config, config
        self.experts = None
        self.shared, self.router = base.shared, base.router
        self.w13, self.w2 = base.w13, base.w2
        initialize_factors(self, self.w13, self.w2, config)
        self.train(base.training)

    def reset_parameters(self):
        QwenMoE.reset_parameters(self)
        reset_factors(self)

    def forward(self, hidden, residual):
        output, _, counts, probabilities = expert_lora(
            hidden,
            residual,
            self.router.weight.T,
            self.w13,
            self.w2,
            self.lora_gate_up_a,
            self.lora_gate_up_b,
            self.lora_down_a,
            self.lora_down_b,
            top_k=self.config.top_k,
            routing_mode="topk_then_softmax",
            scale=self.lora_config.scale,
        )
        if self.shared is not None:
            output = output + self.shared(hidden)
        return output, counts, probabilities

"""A non-elementwise optimizer used to check the generic update boundary.

This is a test algorithm, not Muon: matrix-wide normalization and column-shaped
state deliberately violate the assumptions of flat elementwise sharding.
"""

import torch


class MatrixNormalizedMomentum(torch.optim.Optimizer):
    def __init__(self, parameters, lr=0.005, **ignored):
        super().__init__(parameters, dict(lr=lr, momentum=0.8))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["velocity"] = torch.zeros_like(parameter)
                    state["column_energy"] = parameter.new_zeros(parameter.shape[-1:])
                gradient = parameter.grad
                energy = gradient.square()
                if parameter.ndim > 1:
                    energy = energy.mean(dim=tuple(range(parameter.ndim - 1)))
                velocity, columns = state["velocity"], state["column_energy"]
                beta = group["momentum"]
                velocity.mul_(beta).add_(gradient * (1 - beta))
                columns.mul_(beta).add_(energy * (1 - beta))
                direction = velocity / (columns.sum() + 0.01).sqrt()
                parameter.add_(direction * (-group["lr"]))
        return loss

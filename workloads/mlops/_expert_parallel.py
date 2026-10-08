"""Construction and routing shared by the example MoE architectures.

MLOps owns expert kernels. These helpers only compose them into workload models;
they have no dependency on ShadowSpill's trainer, planner, or runtime.
"""

from contextlib import ExitStack, contextmanager, nullcontext

import torch
from torch import nn
from torch.nn import functional as F


class _ExpertFactory:
    """Temporary construction state; never attach this object to a model."""

    def __init__(self, options, group, buffer, device, parameter_device, cleanup):
        self.options = options
        self.group = group
        self.buffer = buffer
        self.device = device
        self.parameter_device = parameter_device
        self.cleanup = cleanup

    def __call__(self):
        if self.options is None:
            return None
        from mlops.expert_parallel import QuackMoE

        layer = QuackMoE(
            self.options, self.group, buffer=self.buffer, device=self.device
        )
        self.cleanup.callback(layer.close)
        if self.parameter_device != self.device:
            # Copy logical parameters, not their larger communication-bank
            # storage. Publication materializes these values before compute.
            for name, parameter in tuple(layer.named_parameters()):
                value = (
                    parameter.detach().to(self.parameter_device, copy=True).contiguous()
                )
                layer.register_parameter(
                    name, nn.Parameter(value, requires_grad=parameter.requires_grad)
                )
        return layer


@contextmanager
def expert_construction(
    config,
    *,
    renormalize_topk,
    ep_group=None,
    token_capacity=None,
    buffer=None,
    device=None,
    parameter_device=None,
    dtype=None,
    router_dtype=None,
    compute_precision="bf16",
    weight_grad_dtype=torch.bfloat16,
    activation_transport="bf16",
):
    """Set construction defaults and optionally share one EP buffer across layers.

    The yielded factory returns an expert module for EP, or None for local
    experts. On success the model owns its layers and any newly created buffer;
    on failure all resources created here are released, including incomplete
    blocks. A supplied buffer always remains caller-owned.
    """
    parallel = ep_group is not None
    options = None
    if parallel:
        if config.n_layers < 1:
            raise ValueError("an EP model must contain at least one layer")
        if (token_capacity is None) == (buffer is None):
            raise ValueError("EP requires exactly one of token_capacity or buffer")
        if token_capacity is not None and (
            type(token_capacity) is not int or token_capacity <= 0
        ):
            raise ValueError("token_capacity must be a positive integer")
        device = torch.device(device or f"cuda:{torch.cuda.current_device()}")
        if device.type != "cuda":
            raise ValueError("expert parallelism requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        parameter_device = torch.device(parameter_device or device)
        if parameter_device != device and parameter_device.type != "cpu":
            raise ValueError("Parameters must start on the compute device or CPU")
        dtype = dtype or torch.bfloat16
        if dtype != torch.bfloat16:
            raise ValueError("QuackMoE's hidden activation dtype must be BF16")
        # Optional accelerator dependencies stay out of ordinary CPU/meta
        # construction and workload discovery.
        from mlops.expert_parallel import QuackMoEConfig

        options = QuackMoEConfig(
            ep_size=torch.distributed.get_world_size(ep_group),
            num_experts=config.n_experts,
            top_k=config.top_k,
            model_dim=config.d_model,
            expert_hidden_dim=config.d_ff_expert,
            router_dtype=router_dtype or dtype,
            router_weight_grad_dtype=router_dtype or dtype,
            weight_grad_dtype=weight_grad_dtype,
            compute_precision=compute_precision,
            activation_transport=activation_transport,
            renormalize_topk=renormalize_topk,
            share_expert_banks=True,
            init_std=getattr(config, "initializer_range", 0.02),
        )
    elif buffer is not None or token_capacity is not None:
        raise ValueError("a communication buffer requires an EP process group")

    dtype = dtype or torch.get_default_dtype()
    parameter_device = parameter_device or device or torch.get_default_device()
    previous_dtype = torch.get_default_dtype()
    try:
        compute_device = torch.cuda.device(device) if parallel else nullcontext()
        with compute_device, ExitStack() as cleanup:
            if parallel and buffer is None:
                from mlops.expert_parallel import create_buffer

                buffer = create_buffer(options, token_capacity, ep_group)
                cleanup.callback(buffer.destroy)
            factory = _ExpertFactory(
                options, ep_group, buffer, device, parameter_device, cleanup
            )
            torch.set_default_dtype(dtype)
            with torch.device(parameter_device):
                yield factory
            # Ownership passes to the model only after construction succeeds.
            cleanup.pop_all()
    finally:
        torch.set_default_dtype(previous_dtype)


def close_experts(layers, *, destroy_buffer):
    """Close layer runtimes first, then a model-owned communication buffer."""
    layers = [layer for layer in layers if layer is not None]
    with ExitStack() as cleanup:
        if destroy_buffer and layers:
            cleanup.callback(layers[0].communication_buffer.destroy)
        for layer in layers:
            cleanup.callback(layer.close)


def parallel_forward(experts, hidden):
    """Routed output, assignment counts, and probability sums without host reads."""
    from mlops.expert_parallel.quack.router import route_op

    c = experts.config
    flat = hidden.reshape(-1, c.model_dim)
    router = experts.router_weight
    logits = F.linear(flat.to(router.dtype), router).float()
    weights, ids, counts = route_op(logits, c.top_k, c.renormalize_topk)
    routed = experts(flat, expert_ids=ids, routing_weights=weights)
    return routed.reshape_as(hidden), counts, logits.softmax(-1).sum(0)

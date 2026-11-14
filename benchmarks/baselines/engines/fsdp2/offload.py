"""FSDP lifetime management around the unchanged torchtune offloader."""

import torch

from .offload_overlapped import blocks_context as torchtune_blocks_context


def blocks_context(blocks, **options):
    """Install before ``fully_shard`` registers its post-forward hooks.

    Torchtune assumes compute uses the default stream and holds saved tensors
    until their copies finish. FSDP can resize a gathered parameter's storage
    to zero despite those references. The harness must tell the allocator
    about the offloader's use before FSDP releases that storage.
    """

    def check_compute_stream(module, inputs):
        if torch.cuda.current_stream() != torch.cuda.default_stream():
            raise RuntimeError(
                "torchtune activation offload requires the default compute stream"
            )

    # Check before the first block enters the offloader's context.
    for block in blocks:
        block.register_forward_pre_hook(check_compute_stream)
    context = torchtune_blocks_context(blocks, **options)

    def protect_gathered_storage(module, inputs, output):
        if context.use_streams:
            for parameter in module.parameters():
                value = (
                    parameter.to_local()
                    if hasattr(parameter, "to_local")
                    else parameter
                )
                if value.is_cuda:
                    value.record_stream(context.s1)

    # FSDP's subsequently registered post-forward hook may free this storage.
    for block in blocks:
        block.register_forward_hook(protect_gathered_storage, always_call=True)
    return context

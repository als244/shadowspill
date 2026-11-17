"""Shared GLM integration with the ordinary installed decoder definitions."""

from shadowspill_workloads.definitions.decoder import register as register_decoder

from .configuration import build, capture_context, create
from .initialization import describe


def register(registry, settings):
    register_decoder(
        registry,
        settings,
        create=create,
        build=build,
        initialization=describe,
        capture_context=capture_context,
        capabilities=("target_logprobs",),
        sources=("models/shared/glm/training",),
        providers=("tilelang", "cudnn", "flash_mla", "deep_gemm", "tile_kernels"),
    )

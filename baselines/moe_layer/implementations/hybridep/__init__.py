"""NVIDIA MoE with the hybridep dispatcher."""


def build(**configuration):
    from ..nvidia import build as build_nvidia

    return build_nvidia(dispatcher="hybridep", **configuration)

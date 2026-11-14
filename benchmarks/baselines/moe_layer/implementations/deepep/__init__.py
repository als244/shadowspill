"""NVIDIA MoE with the deepep dispatcher."""


def build(**configuration):
    from ..nvidia import build as build_nvidia

    return build_nvidia(dispatcher="deepep", **configuration)

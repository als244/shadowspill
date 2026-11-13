"""Workload opt-in to the operation library's saved implementation contracts."""


def operator_preparations(implementation, dtype):
    if implementation != "mlops":
        return ()
    import torch

    from shadowspill.integrations.mlops import operator_preparations as descriptions

    return descriptions(dtype=getattr(torch, dtype))

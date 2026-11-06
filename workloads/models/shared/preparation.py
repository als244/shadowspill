"""Workload opt-in to the operation library's saved implementation contracts."""


def operator_preparations(implementation, dtype):
    if implementation != "mlops":
        return ()
    import torch
    from mlops.integrations.shadowspill import operator_preparations as descriptions

    return descriptions(dtype=getattr(torch, dtype))

"""The optional independent-element capability required by flat shard updates."""

import torch


def validate_flat_shards(optimizer: torch.optim.Optimizer) -> None:
    known = type(optimizer) in {torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD}
    declared = getattr(optimizer, "supports_flat_parameter_shards", False) is True
    if not known and not declared:
        raise TypeError(
            "flat optimizer shards require independent elementwise updates; "
            "this optimizer has not declared supports_flat_parameter_shards. "
            "Use replicated optimizer storage to retain complete tensor shapes"
        )

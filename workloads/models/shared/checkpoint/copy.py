"""Copy checkpoint tensors into ordinary parameters or FSDP2 row shards."""

from __future__ import annotations

import torch


def copy_into(parameter: torch.Tensor, value: torch.Tensor) -> None:
    """Write a full tensor into a parameter: as it is, or this rank's rows
    of an FSDP2 shard, which shards dimension zero in ``torch.chunk``'s
    pieces with the last ranks possibly empty."""

    if value.shape != tuple(parameter.shape):
        raise ValueError(
            f"shape {tuple(value.shape)} does not fit {tuple(parameter.shape)}"
        )
    with torch.no_grad():
        if hasattr(parameter, "to_local") and hasattr(parameter, "device_mesh"):
            placements = getattr(parameter, "placements", ())
            if (
                len(placements) != 1
                or not getattr(placements[0], "is_shard", lambda: False)()
                or placements[0].dim != 0
            ):
                raise ValueError("only dimension-zero sharded parameters are loaded")
            mesh = parameter.device_mesh
            rank, world = mesh.get_local_rank(), mesh.size()
            pieces = value.chunk(world, dim=0)
            local = parameter.to_local()
            piece = pieces[rank] if rank < len(pieces) else value[:0]
            if tuple(piece.shape) != tuple(local.shape):
                raise ValueError(
                    f"rank {rank}'s shard holds {tuple(local.shape)},"
                    f" not {tuple(piece.shape)}"
                )
            local.copy_(piece.to(dtype=local.dtype, device=local.device))
        else:
            parameter.copy_(value.to(dtype=parameter.dtype, device=parameter.device))

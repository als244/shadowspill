"""Parallelism declarations a run takes as its ``parallelism`` setting."""

from collections.abc import Mapping, Sequence
from typing import Any

#: Default shard floor, in bytes (2 MiB). A parameter whose weight storage is
#: smaller stays whole on every rank: its shard would save little memory and
#: cost a collective at every fetch.
MIN_SHARD_BYTES = 2 << 20


def data_parallel(
    ranks: int,
    *,
    fields: Sequence[Sequence[str]] | None = None,
    outputs: Mapping[str, Any] | None = None,
    min_shard_bytes: int = MIN_SHARD_BYTES,
) -> dict[str, Any]:
    """Declare data parallelism over ``ranks`` workers.

    Each worker trains on its own part of every batch. Weights, gradients and
    optimizer state are sharded over the workers, except parameters whose
    weight storage is under ``min_shard_bytes``, which every worker keeps whole.

    Parameters
    ----------
    ranks : int
        Number of workers.
    fields : sequence of sequence of str, optional
        Input fields each worker receives, each as a path of keys such as
        ``("inputs", "tokens")``. By default they come from the model and
        objective during setup; custom definitions may list them.
    outputs : mapping of str to mapping, optional
        How each named output is assembled across workers, for example
        ``{"prediction": {"mode": "leader", "axis": None}}``. None by default.
    min_shard_bytes : int, default MIN_SHARD_BYTES
        Weight storage size, in bytes, below which a parameter is kept whole
        on every worker. Zero shards every parameter.

    Returns
    -------
    dict
        The declaration, passed as a run's ``parallelism`` setting.
    """
    return {
        "ranks_per_replica": ranks,
        "num_replicas": 1,
        "groups": {"all_ranks": [list(range(ranks))]},
        "default_parameter_placement": {"kind": "Replicated", "group": "all_ranks"},
        "parameter_placements": {},
        "default_parameter_storage": {
            "group": "all_ranks",
            "weights": "sharded",
            "gradients": "sharded",
            "optimizer": "sharded",
            "min_shard_bytes": min_shard_bytes,
        },
        "parameter_storage": {},
        "inputs": {
            "version": 1,
            "partitions": {"name": "inputs", "groups": [[r] for r in range(ranks)]},
            "fields": [
                {"path": list(path), "mode": "replicate", "axis": None}
                for path in (() if fields is None else fields)
            ],
            "loss": "leader",
            "outputs": {} if outputs is None else outputs,
        },
    }


def expert_parallel(
    ranks: int,
    *,
    fields: Sequence[Sequence[str]] | None = None,
    experts: Sequence[str],
    outputs: Mapping[str, Any] | None = None,
    min_shard_bytes: int = MIN_SHARD_BYTES,
) -> dict[str, Any]:
    """Declare expert parallelism over ``ranks`` workers.

    Each worker trains on its own part of every batch and stores its own
    share of the experts. Every other parameter keeps its weights and
    gradients whole on every worker and shards its optimizer state, unless its
    weight storage is under ``min_shard_bytes``.

    Parameters
    ----------
    ranks : int
        Number of workers.
    fields : sequence of sequence of str, optional
        Input fields each worker receives, as in `data_parallel`.
    experts : sequence of str
        Name patterns of the expert parameters, such as
        ``"layers.*.mlp.experts.*"``. Each is split across the workers along
        its first dimension.
    outputs : mapping of str to mapping, optional
        How each named output is assembled across workers, as in
        `data_parallel`.
    min_shard_bytes : int, default MIN_SHARD_BYTES
        Weight storage size, in bytes, below which a non-expert parameter
        keeps its optimizer state whole on every worker. Zero shards every
        one. Different experts are never merged.

    Returns
    -------
    dict
        The declaration, passed as a run's ``parallelism`` setting.
    """
    result = data_parallel(
        ranks, fields=fields, outputs=outputs, min_shard_bytes=min_shard_bytes
    )
    result["groups"]["local"] = [[r] for r in range(ranks)]
    storage = result["default_parameter_storage"]
    storage.update(weights="replicated", gradients="replicated")
    result["parameter_placements"] = {
        name: {"kind": "Sharded", "group": "all_ranks", "dim": 0} for name in experts
    }
    result["parameter_storage"] = {name: {"group": "local"} for name in experts}
    return result

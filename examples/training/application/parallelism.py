"""Portable declarations for the existing generic DP/EP ownership contracts."""


def data_parallel(ranks, *, fields, outputs=None):
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
            "min_shard_bytes": 0,
        },
        "parameter_storage": {},
        "inputs": {
            "version": 1,
            "partitions": {"name": "inputs", "groups": [[r] for r in range(ranks)]},
            "fields": [
                {"path": list(path), "mode": "replicate", "axis": None}
                for path in fields
            ],
            "loss": "leader",
            "outputs": {} if outputs is None else outputs,
        },
    }


def expert_parallel(ranks, *, fields, experts, outputs=None):
    """Shard the caller-declared expert axis; keep independent input owners."""
    result = data_parallel(ranks, fields=fields, outputs=outputs)
    result["groups"]["local"] = [[r] for r in range(ranks)]
    storage = result["default_parameter_storage"]
    del storage["min_shard_bytes"]
    storage.update(weights="replicated", gradients="replicated")
    result["parameter_placements"] = {
        name: {"kind": "Sharded", "group": "all_ranks", "dim": 0} for name in experts
    }
    result["parameter_storage"] = {name: {"group": "local"} for name in experts}
    return result

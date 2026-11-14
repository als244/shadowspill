"""Ranks agree on what they exchange, not on what each compiled."""

from dataclasses import replace

from shadowspill.parallelism.agreement import agreement_digest, shared_structure

from ._example import program


def test_agreement_ignores_each_ranks_own_build_and_keys_by_name():
    first = program()
    second = replace(
        first,
        setup={"gpu": "another card"},
        executables={"k": b"other kernels"},
        fixed_device_bytes=12345,
        bindings={**first.bindings, "num_examples": 7, "task_names": ["x"]},
    )
    assert agreement_digest(first) == agreement_digest(second)
    structure = shared_structure(first)
    assert "num_examples" not in str(structure)
    sharded = replace(
        first,
        bindings={
            **first.bindings,
            "weight_shards": {
                first.alias_groups[0].alias_group_id: {
                    "groups": [[0, 1]],
                    "shard_bytes": 64,
                }
            },
        },
    )
    assert agreement_digest(sharded) != agreement_digest(first)
    names = shared_structure(sharded)["weight_shards"]
    assert all(" " not in key or key.split() for key in names)

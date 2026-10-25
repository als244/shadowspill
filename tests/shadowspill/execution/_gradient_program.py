"""Persistent padded gradients for native spill/restore qualification."""

from dataclasses import replace


def gradient_program(shard_ranks, world_size, dtype="torch.bfloat16"):
    from shadowspill.artifacts._values import plain
    from tests.shadowspill.execution._sharded_storage import program

    source = program(shard_ranks, world_size)
    bindings = plain(source.bindings)
    description = bindings.pop("weight_shards")["alias_000001"]
    groups = description["groups"]
    width = 2 if dtype == "torch.bfloat16" else 4
    padded_bytes = (16 if shard_ranks == 2 else 15) * width
    description["padded_bytes"] = padded_bytes
    description["shard_bytes"] = padded_bytes // shard_ranks
    description["logical_bytes"] = 15 * width
    description["dtype"] = dtype
    description["replica_groups"] = (
        tuple(zip(*groups, strict=True)) if len(groups) > 1 else ()
    )
    bindings.update(
        kind="forward_backward",
        registrations=[],
        root_inputs=[[]],
        gradients=[],
        persistent_gradients=[
            dict(
                parameter_name="weight",
                gradient_object_id="object_000001",
                shape=(15,),
                dtype=dtype,
            )
        ],
        parameter_ownership=[
            dict(
                names=("weight",),
                local_shape=(15,),
                gradients=dict(groups=groups, elements=15),
            )
        ],
        gradient_shards={"alias_000001": description},
    )
    bindings["model_tensor_geometry"]["object_000001"]["dtype"] = dtype
    return replace(
        source,
        alias_groups=(replace(source.alias_groups[0], size_bytes=padded_bytes),),
        objects=(replace(source.objects[0], size_bytes=15 * width),),
        bindings=bindings,
    )

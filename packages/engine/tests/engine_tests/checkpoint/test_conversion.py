"""Converted state uses the actual rank codec and atomic generation publisher."""

import hashlib
from dataclasses import replace

import pytest
import torch

from engine_tests.checkpoint.test_logical import model_spec, shaped
from engine_tests.checkpoint.test_state import plan
from engine_tests.server.test_artifacts import store
from engine_tests.server.test_checkpoints import boundary, stream_rank
from shadowspill.checkpoint import checkpoint_size, load_checkpoint, save_checkpoint
from shadowspill.checkpoint.compatibility import describe_family, require_compatible
from shadowspill.checkpoint.conversion import convert_state
from shadowspill.checkpoint.identity import state_identity
from shadowspill.checkpoint.state import PlanState
from shadowspill.ir import AliasGroupSpec, ObjectSpec
from shadowspill.parallelism import Parallelism
from shadowspill.parallelism.resolve import ParameterGeometry, resolve_parameters
from shadowspill.server.checkpoints import publish_checkpoint
from shadowspill.training.worker.checkpoint_records import RankSave
from shadowspill_client.protocol.codec import decode

WEIGHT = torch.linspace(-1.3, 2.7, 48).reshape(4, 12)
FIRST = torch.linspace(-0.2, 0.3, 48).reshape(4, 12)
SECOND = torch.linspace(0.01, 0.4, 48).reshape(4, 12)
GRADIENT = torch.sin(torch.arange(48).float()).reshape(4, 12)


def family(count, dim, *, master=True, frozen=False):
    shape = (4 // count, 12) if dim == 0 else (4, 12 // count)
    source = shaped(shape, model_spec(tuple(reversed(range(count))), dim))
    elements = shape[0] * shape[1]
    bindings = dict(source.bindings)
    bindings["optimizer_type"] = "example.Adam"
    rows = []
    for row in bindings["optimizer_bindings"]:
        if row["role"] in {"parameter", "compute_copy"} and not master:
            continue
        initial = (
            {"kind": "parameter", "name": "linear.weight"}
            if row["role"] == "parameter"
            else {"kind": "constant", "value": 0}
        )
        rows.append({**row, "initialization": initial})
    moment = next(row for row in rows if row["name"].endswith("exp_avg"))
    rows.append(
        {
            **moment,
            "name": "optimizer.linear.weight.exp_avg_sq",
            "object_id": "variance",
        }
    )
    bindings["optimizer_bindings"] = rows
    aliases = (*source.alias_groups, AliasGroupSpec("variance", "rank0", elements * 4))
    objects = (*source.objects, ObjectSpec("variance", "variance", 0, elements * 4))
    if master:
        bindings["model_tensor_geometry"] = {
            name: {**geometry, "dtype": "torch.bfloat16"}
            for name, geometry in bindings["model_tensor_geometry"].items()
        }
        bindings["optimizer_bindings"] = [
            {**row, "dtype": "torch.bfloat16"} if row["role"] == "compute_copy" else row
            for row in rows
        ]
        aliases = tuple(
            replace(alias, size_bytes=elements * 2)
            if alias.alias_group_id == "0"
            else alias
            for alias in aliases
        )
        objects = tuple(
            replace(obj, size_bytes=elements * 2) if obj.alias_group_id == "0" else obj
            for obj in objects
        )
    if frozen:
        names = ("base.weight", "base.tied")
        bindings["registrations"] = [
            {**row, "trainable": True} for row in bindings["registrations"]
        ] + [
            dict(name=name, object_id=names[0], parameter=True, trainable=False)
            for name in names
        ]
        bindings["model_tensor_geometry"] = {
            **bindings["model_tensor_geometry"],
            names[0]: {
                "shape": shape,
                "stride": (shape[1], 1),
                "dtype": "torch.float32",
            },
        }
        ownership = resolve_parameters(
            source.parallelism, (ParameterGeometry(names, shape, False, elements * 4),)
        )[0]
        bindings["parameter_ownership"] = [
            *bindings["parameter_ownership"],
            ownership.to_dict(),
        ]
        aliases = (*aliases, AliasGroupSpec("base", "rank0", elements * 4))
        objects = (*objects, ObjectSpec(names[0], "base", 0, elements * 4))
    source = replace(source, bindings=bindings, alias_groups=aliases, objects=objects)
    return [PlanState(plan(source), {}, logical_rank=rank) for rank in range(count)]


class HostRank:
    def __init__(self, state, *, initialize=False):
        self.state = state
        self.plan, self.plans, self.logical_rank = (
            state.plan,
            state.plans,
            state.logical_rank,
        )
        self.values = {
            chunk.name: torch.zeros(
                chunk.storage_elements
                if chunk.region_elements is None
                else chunk.region_elements,
                dtype=chunk.dtype,
            )
            for chunk in state.tensor_chunks()
        }
        if initialize:
            for item in state.logical_tensors():
                name = item.tensor.identity.name
                logical = (
                    FIRST
                    if name.endswith("exp_avg")
                    else SECOND
                    if name.endswith("exp_avg_sq")
                    else torch.tensor(17, dtype=torch.int64)
                    if name.endswith(".step")
                    else WEIGHT
                )
                local = logical[
                    tuple(
                        slice(start, start + size)
                        for start, size in zip(
                            item.view.origin, item.view.local_shape, strict=True
                        )
                    )
                ]
                self.values[item.archive.name].as_strided(
                    item.view.local_shape, item.view.stride, item.view.storage_offset
                ).copy_(local)
            self.values["optimizer_group.0.lr"].fill_(0.125)

    def tensor_chunks(self):
        return self.state.tensor_chunks()

    def export_tensors(self, callback, *, chunk_bytes, names=None):
        for chunk in self.tensor_chunks():
            if names is not None and chunk.name not in names:
                continue
            value = self.values[chunk.name]
            step = chunk_bytes // chunk.dtype.itemsize
            for start in range(0, value.numel(), step):
                callback(
                    replace(chunk, offset=chunk.region_offset + start),
                    value[start : start + step],
                )

    def initialize(self, callback, *, chunk_bytes, restore, replicated_copies=None):
        assert restore and not replicated_copies
        self.export_tensors(callback, chunk_bytes=chunk_bytes)


def save_sources(tmp_path, states, weights):
    sources = []
    for state in states:
        path = tmp_path / f"source-{state.logical_rank}.ssp"
        save_checkpoint(
            HostRank(state, initialize=True),
            path,
            metadata={"updates": 17},
            weights=weights,
            chunk_bytes=32,
        )
        sources.append((state, path))
    return sources


def temporary(rank, request, destination):
    assert request.name == "optimizer_group.0.lr"
    destination.fill_(0.5)


def adam(weight, first, second, gradient, step):
    parameter = torch.nn.Parameter(weight.clone())
    optimizer = torch.optim.Adam(
        [parameter], lr=0.03, betas=(0.8, 0.95), eps=1e-7, foreach=False
    )
    optimizer.state[parameter] = {
        "step": torch.tensor(float(step)),
        "exp_avg": first.clone(),
        "exp_avg_sq": second.clone(),
    }
    parameter.grad = gradient.clone()
    optimizer.step()
    return parameter.detach(), optimizer.state[parameter]


@pytest.mark.parametrize("source_count,target_count", [(1, 4), (4, 1), (2, 4), (4, 2)])
@pytest.mark.parametrize(
    "weights,master", [("master", True), ("compute", True), ("master", False)]
)
def test_converted_generation_loads_and_takes_the_next_adam_update(
    tmp_path, source_count, target_count, weights, master
):
    sources = save_sources(tmp_path, family(source_count, 0, master=master), weights)
    targets = family(target_count, 1, master=master)
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for _, path in sources]
    expected_weight = (
        WEIGHT if weights == "master" or not master else WEIGHT.bfloat16().float()
    )
    expected, expected_state = adam(expected_weight, FIRST, SECOND, GRADIENT, 17)
    with store(tmp_path) as artifacts:
        info = replace(
            boundary(artifacts, count=target_count, weights=weights),
            completed_updates=17,
        )
        with convert_state(
            sources,
            targets,
            tmp_path,
            scratch_bytes=128,
            max_disk_bytes=4096,
            initialize_temporary=temporary,
        ) as converted:
            assert converted.source_metadata == {"updates": 17}
            assert (
                converted.logical_identity
                == describe_family([state for state, _ in sources]).identity
            )
            ranks = [
                RankSave(
                    rank.logical_rank,
                    checkpoint_size(
                        rank, metadata=info.rank_metadata(), weights=weights
                    ),
                    state_identity(rank),
                    rank.plan.program.digest,
                    {},
                )
                for rank in converted.ranks
            ]
            result = publish_checkpoint(
                artifacts,
                info,
                ranks,
                lambda rank, sink: stream_rank(converted.ranks[rank.rank], info, sink),
                name="converted",
            )
        with artifacts.open(result) as stream:
            manifest = decode(stream.read())
        for rank in manifest.ranks:
            path = tmp_path / f"target-{rank.rank}.ssp"
            with artifacts.open(rank.archive) as stream:
                path.write_bytes(stream.read())
            executor = HostRank(targets[rank.rank])
            assert (
                load_checkpoint(executor, path, chunk_bytes=17) == info.rank_metadata()
            )
            column = targets[rank.rank].logical_rank
            column = (target_count - column - 1) * (12 // target_count)
            shape = (4, 12 // target_count)
            selected = slice(column, column + shape[1])
            weight_name = "master.linear.weight" if master else "linear.weight"
            weight = executor.values[weight_name].reshape(shape)
            torch.testing.assert_close(
                weight, expected_weight[:, selected], rtol=0, atol=0
            )
            first = executor.values["optimizer.linear.weight.exp_avg"].reshape(shape)
            second = executor.values["optimizer.linear.weight.exp_avg_sq"].reshape(
                shape
            )
            step = executor.values["optimizer.linear.weight.step"].item()
            actual, state = adam(weight, first, second, GRADIENT[:, selected], step)
            torch.testing.assert_close(
                actual, expected[:, selected], rtol=1e-6, atol=1e-7
            )
            torch.testing.assert_close(
                state["exp_avg"], expected_state["exp_avg"][:, selected], rtol=0, atol=0
            )
            torch.testing.assert_close(
                state["exp_avg_sq"],
                expected_state["exp_avg_sq"][:, selected],
                rtol=0,
                atol=0,
            )
            assert state["step"].item() == 18
            assert executor.values["optimizer_group.0.lr"].item() == 0.5
    assert not list(tmp_path.glob(".checkpoint-conversion-*"))
    assert before == [
        hashlib.sha256(path.read_bytes()).hexdigest() for _, path in sources
    ]


@pytest.mark.parametrize(
    "change,error",
    [
        ("optimizer", "interpretation differs"),
        ("trainable", "interpretation differs"),
        ("inventory", "inventory differs"),
        ("ties", "structural ties differ"),
    ],
)
def test_incompatible_state_fails_before_any_source_file_or_staging_is_opened(
    tmp_path, change, error
):
    sources = family(1, 0)
    target = family(1, 1)[0].plan.program
    bindings = dict(target.bindings)
    if change == "optimizer":
        bindings["optimizer_type"] = "different.Adam"
    elif change == "trainable":
        bindings["parameter_ownership"] = [
            {**owner, "trainable": False} for owner in bindings["parameter_ownership"]
        ]
    elif change == "inventory":
        bindings["registrations"] = bindings["registrations"][:1]
    else:
        alias = AliasGroupSpec("untied", "rank0", 96)
        target = replace(
            target,
            alias_groups=(*target.alias_groups, alias),
            objects=tuple(
                replace(obj, alias_group_id="untied")
                if obj.object_id == "head.weight"
                else obj
                for obj in target.objects
            ),
        )
    target = PlanState(plan(replace(target, bindings=bindings)), {}, logical_rank=0)
    with (
        pytest.raises(ValueError, match=error),
        convert_state(
            [(sources[0], tmp_path / "not-opened.ssp")],
            [target],
            tmp_path,
            scratch_bytes=128,
            max_disk_bytes=4096,
            initialize_temporary=temporary,
        ),
    ):
        pytest.fail("incompatible conversion yielded state")
    assert not list(tmp_path.iterdir())


def test_incomplete_or_misordered_rank_sets_are_rejected():
    states = family(2, 0)
    with pytest.raises(ValueError, match="rank set differs"):
        describe_family(states[:1])
    with pytest.raises(ValueError, match="once in order"):
        describe_family(list(reversed(states)))
    require_compatible(describe_family(family(2, 0)), describe_family(family(4, 1)))


def tied_views(weight_stride, head_stride):
    source = shaped((4, 4), Parallelism())
    geometry = source.bindings["model_tensor_geometry"]
    return describe_family(
        [
            PlanState(
                plan(
                    replace(
                        source,
                        bindings={
                            **source.bindings,
                            "model_tensor_geometry": {
                                name: {
                                    **value,
                                    "stride": weight_stride
                                    if name == "linear.weight"
                                    else head_stride,
                                }
                                for name, value in geometry.items()
                            },
                        },
                    )
                ),
                {},
                logical_rank=0,
            )
        ]
    )


def test_tied_relationship_is_checked_even_when_current_values_could_be_equal():
    transposed = tied_views((4, 1), (1, 4))
    identical = tied_views((4, 1), (4, 1))
    with pytest.raises(ValueError, match="tied view relationship differs"):
        require_compatible(transposed, identical)
    # Reordering physical storage preserves the same logical transpose relation.
    reordered = tied_views((1, 4), (4, 1))
    require_compatible(transposed, reordered)
    assert transposed.identity == reordered.identity


def test_unproved_internal_overlap_needs_an_explicit_conversion_mapping():
    with pytest.raises(ValueError, match="overlapping state views"):
        tied_views((4, 1), (0, 1))


def test_encoder_cannot_silently_change_the_converted_weight_selection(tmp_path):
    sources = save_sources(tmp_path, family(2, 0), "compute")
    destination = tmp_path / "existing.ssp"
    destination.write_bytes(b"keep existing publication")
    with (
        convert_state(
            sources,
            family(4, 1),
            tmp_path,
            scratch_bytes=128,
            max_disk_bytes=4096,
            initialize_temporary=temporary,
        ) as converted,
        pytest.raises(ValueError, match="export differs from its description"),
    ):
        save_checkpoint(
            converted.ranks[0], destination, weights="master", chunk_bytes=32
        )
    assert destination.read_bytes() == b"keep existing publication"
    assert not list(tmp_path.glob(".checkpoint-conversion-*"))


@pytest.mark.parametrize("failure", ["missing", "temporary", "last_rank"])
def test_failed_conversion_or_publication_leaves_no_complete_generation(
    tmp_path, failure
):
    sources = save_sources(tmp_path, family(2, 0), "master")
    targets = family(4, 1)

    def initialize(*args):
        if failure == "temporary":
            raise RuntimeError("temporary preparation failed")
        temporary(*args)

    with store(tmp_path) as artifacts:
        info = replace(boundary(artifacts, count=4), completed_updates=17)
        with (
            pytest.raises((ValueError, RuntimeError)),
            convert_state(
                sources,
                targets,
                tmp_path,
                scratch_bytes=128,
                max_disk_bytes=4096,
                initialize_temporary=None if failure == "missing" else initialize,
            ) as converted,
        ):
            ranks = [
                RankSave(
                    rank.logical_rank,
                    checkpoint_size(rank, metadata=info.rank_metadata()),
                    state_identity(rank),
                    rank.plan.program.digest,
                    {},
                )
                for rank in converted.ranks
            ]

            def write(rank, sink):
                def corrupt(manifest):
                    manifest["state"] = "0" * 64

                return stream_rank(
                    converted.ranks[rank.rank],
                    info,
                    sink,
                    fault=corrupt if rank.rank == 3 else None,
                )

            publish_checkpoint(artifacts, info, ranks, write, name="failed")
        assert artifacts.resolve_alias(info.session_id, info.run_id, "failed") is None
    assert not list(tmp_path.glob(".checkpoint-conversion-*"))

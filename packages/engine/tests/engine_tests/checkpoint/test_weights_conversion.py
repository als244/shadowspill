"""Weights-only layout changes retain model values without importing training state."""

import hashlib
import io
from dataclasses import replace

import pytest
import torch

from engine_tests.checkpoint.test_conversion import WEIGHT, HostRank, family
from engine_tests.checkpoint.test_exports import decode
from engine_tests.checkpoint.test_state import plan
from engine_tests.execution.test_shard_initialization import fixture
from shadowspill.checkpoint import write_checkpoint, write_weights
from shadowspill.checkpoint.compatibility import describe_family, require_compatible
from shadowspill.checkpoint.staging import staging_bytes
from shadowspill.checkpoint.state import PlanState
from shadowspill.checkpoint.weights_conversion import convert_weights, weights_tensors
from shadowspill.ir import AliasGroupSpec, TransferPhase, TransferSpec


def saved(tmp_path, states, *, weights="master", frozen_state_id=None):
    result = []
    for state in states:
        path = tmp_path / f"source-{state.logical_rank}.ssp"
        with path.open("wb") as stream:
            write_weights(
                HostRank(state, initialize=True),
                stream,
                weights=weights,
                frozen_state_id=frozen_state_id,
                chunk_bytes=17,
                metadata={"origin_update": 17},
            )
        result.append((state, path))
    return result


def exported(rank):
    values = {}

    def collect(request, value):
        if request.name not in values:
            length = (
                request.storage_elements
                if request.region_elements is None
                else request.region_elements
            )
            values[request.name] = torch.empty(length, dtype=request.dtype)
        start = request.offset - request.region_offset
        values[request.name][start : start + value.numel()].copy_(value)

    names = {
        item.archive.name
        for item in rank.state.logical_tensors(roles=("parameter", "buffer", "master"))
        if item.archive is not None
    }
    rank.export_tensors(collect, chunk_bytes=17, names=names)
    return values


def rewrite(states, change):
    result = []
    for state in states:
        program = state.plan.program
        bindings = dict(program.bindings)
        if change == "optimizer":
            bindings["optimizer_type"] = "different.FreshOptimizer"
        elif change == "trainable":
            bindings["parameter_ownership"] = [
                {**owner, "trainable": False}
                for owner in bindings["parameter_ownership"]
            ]
            bindings["registrations"] = [
                {**row, "trainable": False} for row in bindings["registrations"]
            ]
        elif change == "inventory":
            bindings["registrations"] = bindings["registrations"][:1]
        elif change == "ties":
            program = replace(
                program,
                alias_groups=(
                    *program.alias_groups,
                    AliasGroupSpec("untied", "rank0", 96),
                ),
                objects=tuple(
                    replace(obj, alias_group_id="untied")
                    if obj.object_id == "head.weight"
                    else obj
                    for obj in program.objects
                ),
            )
        elif change == "buffer":
            bindings["registrations"] = [
                {**row, "parameter": False, "persistent": True}
                if row["name"].startswith("base.")
                else row
                for row in bindings["registrations"]
            ]
        else:
            raise AssertionError(change)
        result.append(
            PlanState(
                plan(replace(program, bindings=bindings)),
                {},
                logical_rank=state.logical_rank,
            )
        )
    return result


@pytest.mark.parametrize("counts", [(1, 4), (4, 1), (2, 3)])
@pytest.mark.parametrize(
    "source_master,target_master",
    [(True, True), (True, False), (False, True), (False, False)],
)
@pytest.mark.parametrize("weights", ["master", "compute"])
def test_bounded_weight_transfer_preserves_selected_precision_and_every_target_copy(
    tmp_path, monkeypatch, counts, source_master, target_master, weights
):
    sources = saved(
        tmp_path, family(counts[0], 0, master=source_master), weights=weights
    )
    targets = rewrite(family(counts[1], 1, master=target_master), "optimizer")
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for _, path in sources]
    allocations = []
    empty = torch.empty

    def allocate(*args, **kwargs):
        value = empty(*args, **kwargs)
        allocations.append(value.nbytes)
        return value

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", allocate)
        with convert_weights(
            sources,
            targets,
            tmp_path,
            scratch_bytes=128,
            max_disk_bytes=staging_bytes(weights_tensors(targets)),
        ) as converted:
            assert max(allocations) <= 128
            patch.undo()  # The test collector below retains complete tiny tensors.
            selected = (
                WEIGHT.bfloat16().float()
                if source_master and weights == "compute"
                else WEIGHT
            )
            for rank in converted.ranks:
                values = exported(rank)
                assert not any(
                    name.startswith(("optimizer", "gradient")) for name in values
                )
                for entry in rank.state.logical_tensors(roles=("parameter", "master")):
                    expected = selected[
                        tuple(
                            slice(start, start + size)
                            for start, size in zip(
                                entry.view.origin, entry.view.local_shape, strict=True
                            )
                        )
                    ]
                    torch.testing.assert_close(
                        values[entry.archive.name].reshape(entry.view.local_shape),
                        expected.to(entry.archive.dtype),
                        rtol=0,
                        atol=0,
                    )
                # Ordinary model export still works; publishing this as a full
                # checkpoint cannot conjure optimizer state from its metadata.
                sink = io.BytesIO()
                write_weights(rank, sink, chunk_bytes=17)
                manifest, _ = decode(sink.getvalue())
                assert manifest["format"] == "shadowspill.weights"
            assert converted.source_metadata["origin_update"] == 17
    assert not list(tmp_path.glob(".checkpoint-conversion-*"))
    assert before == [
        hashlib.sha256(path.read_bytes()).hexdigest() for _, path in sources
    ]


@pytest.mark.parametrize("change", ["optimizer", "trainable"])
def test_fresh_weights_allow_new_training_settings_without_loosening_resume(
    tmp_path, change
):
    states = family(1, 0)
    targets = rewrite(family(1, 1), change)
    with pytest.raises(ValueError, match="interpretation differs"):
        require_compatible(describe_family(states), describe_family(targets))
    with convert_weights(
        saved(tmp_path, states),
        targets,
        tmp_path,
        scratch_bytes=128,
        max_disk_bytes=4096,
    ) as converted:
        assert set(exported(converted.ranks[0])) == {
            "linear.weight",
            "master.linear.weight",
        }


@pytest.mark.parametrize(
    "change,match", [("inventory", "inventory differs"), ("ties", "ties differ")]
)
def test_model_incompatibility_fails_before_any_source_or_staging_opens(
    tmp_path, change, match
):
    targets = rewrite(family(1, 1), change)
    with (
        pytest.raises(ValueError, match=match),
        convert_weights(
            [(family(1, 0)[0], tmp_path / "absent.ssp")],
            targets,
            tmp_path,
            scratch_bytes=128,
            max_disk_bytes=4096,
        ),
    ):
        pytest.fail("incompatible model became readable")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("frozen_state_id", [None, "exact-base"])
def test_full_and_partial_frozen_weights_preserve_the_exact_omission(
    tmp_path, frozen_state_id
):
    sources = saved(
        tmp_path, family(2, 0, frozen=True), frozen_state_id=frozen_state_id
    )
    targets = family(4, 1, frozen=True)
    chunks = weights_tensors(targets, frozen_state_id)
    with convert_weights(
        sources,
        targets,
        tmp_path,
        scratch_bytes=128,
        max_disk_bytes=staging_bytes(chunks),
        frozen_state_id=frozen_state_id,
    ) as converted:
        for rank in converted.ranks:
            values = exported(rank)
            assert ("base.weight" in values) == (frozen_state_id is None)
            sink = io.BytesIO()
            write_weights(rank, sink, frozen_state_id=frozen_state_id, chunk_bytes=17)
            manifest, _ = decode(sink.getvalue())
            assert manifest["frozen_state_id"] == frozen_state_id


def test_persistent_buffers_are_transferred_even_with_frozen_provenance(tmp_path):
    sources = saved(
        tmp_path, rewrite(family(2, 0, frozen=True), "buffer"), frozen_state_id="base"
    )
    targets = rewrite(family(4, 1, frozen=True), "buffer")
    with convert_weights(
        sources,
        targets,
        tmp_path,
        scratch_bytes=128,
        max_disk_bytes=4096,
        frozen_state_id="base",
    ) as converted:
        for rank in converted.ranks:
            values = exported(rank)
            for item in rank.state.logical_tensors(roles=("buffer",)):
                column = item.view.origin[1]
                torch.testing.assert_close(
                    values[item.archive.name].reshape(4, 3),
                    WEIGHT[:, column : column + 3],
                    rtol=0,
                    atol=0,
                )


def test_omitted_base_cannot_change_when_target_training_selection_changes(tmp_path):
    targets = rewrite(family(1, 0, frozen=True), "trainable")
    with (
        pytest.raises(ValueError, match="omitted frozen-base inventory"),
        convert_weights(
            [(family(1, 0, frozen=True)[0], tmp_path / "absent.ssp")],
            targets,
            tmp_path,
            scratch_bytes=128,
            max_disk_bytes=4096,
            frozen_state_id="base",
        ),
    ):
        pytest.fail("different frozen omission was accepted")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "fault",
    ["budget", "owner_failure", "checksum", "checkpoint", "selection", "metadata"],
)
def test_failures_never_publish_or_leave_unowned_staging(tmp_path, fault):
    states = family(2, 0)
    sources = saved(tmp_path, states)
    target = family(1, 1)
    if fault in {"checkpoint", "selection", "metadata"}:
        with sources[1][1].open("wb") as stream:
            if fault == "checkpoint":
                write_checkpoint(
                    HostRank(states[1], initialize=True), stream, chunk_bytes=17
                )
            else:
                write_weights(
                    HostRank(states[1], initialize=True),
                    stream,
                    chunk_bytes=17,
                    weights="compute" if fault == "selection" else "master",
                    metadata={"origin_update": 99 if fault == "metadata" else 17},
                )
    if fault == "checksum":
        with sources[1][1].open("r+b") as stream:
            stream.seek(512)
            value = stream.read(1)[0]
            stream.seek(512)
            stream.write(bytes([value ^ 255]))
    with (
        pytest.raises((ValueError, RuntimeError)),
        convert_weights(
            sources,
            target,
            tmp_path,
            scratch_bytes=128,
            max_disk_bytes=1 if fault == "budget" else 4096,
        ),
    ):
        if fault == "owner_failure":
            raise RuntimeError("publisher failed")
        pytest.fail("invalid weights were published")
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "source-0.ssp",
        "source-1.ssp",
    ]


def test_rank_specific_metadata_requires_exact_preflight(tmp_path):
    sources = saved(tmp_path, family(2, 0))
    metadata = [{"origin_update": 17}, {"origin_update": 19}]
    with sources[1][1].open("wb") as stream:
        write_weights(
            HostRank(sources[1][0], initialize=True),
            stream,
            metadata=metadata[1],
            chunk_bytes=17,
        )
    with convert_weights(
        sources,
        family(1, 1),
        tmp_path,
        scratch_bytes=128,
        max_disk_bytes=4096,
        expected_metadata=metadata,
    ) as converted:
        assert len(converted.ranks) == 1


def test_retained_physical_copies_are_not_collapsed_through_compute_precision(tmp_path):
    states = family(1, 0)
    source = states[0].plan.program
    source = replace(
        source,
        bindings={
            **source.bindings,
            "optimizer_bindings": [
                {**row, "initialization": {"kind": "external"}}
                if row["role"] == "parameter"
                else row
                for row in source.bindings["optimizer_bindings"]
            ],
        },
    )
    states = [PlanState(plan(source), {}, logical_rank=0)]
    sources = saved(tmp_path, states)
    with convert_weights(
        sources, family(2, 1), tmp_path, scratch_bytes=128, max_disk_bytes=4096
    ) as converted:
        for rank in converted.ranks:
            values = exported(rank)
            column = (1 - rank.logical_rank) * 6
            torch.testing.assert_close(
                values["master.linear.weight"].reshape(4, 6),
                WEIGHT[:, column : column + 6],
                rtol=0,
                atol=0,
            )
        with pytest.raises(ValueError, match=r"omitted|missing|differs"):
            write_checkpoint(converted.ranks[0], io.BytesIO(), chunk_bytes=17)


def flat_states(elements, replicated):
    source = fixture(elements=elements)
    bindings = {
        **source.bindings,
        "registrations": [
            {**row, "parameter": True, "trainable": True}
            for row in source.bindings["registrations"]
        ],
        "parameter_ownership": [
            {
                **source.bindings["parameter_ownership"][0],
                "global_shape": (elements,),
                "trainable": True,
            }
        ],
    }
    if replicated:
        bindings["weight_shards"] = {}
    source = replace(source, bindings=bindings)
    target = plan(source)
    if not replicated:
        size = 2 * ((elements + 1) // 2)
        target._costed_program = replace(
            source,
            alias_groups=tuple(
                replace(
                    row,
                    transfer=TransferSpec(
                        size,
                        (TransferPhase("copy", bytes=size),),
                        (TransferPhase("copy", bytes=size),),
                    ),
                )
                if row.alias_group_id == "weight"
                else row
                for row in source.alias_groups
            ),
        )
    return [PlanState(target, {}, logical_rank=rank) for rank in range(4)]


@pytest.mark.parametrize("elements", [1, 15])
@pytest.mark.parametrize("replicated", [False, True])
@pytest.mark.parametrize("weights", ["master", "compute"])
def test_flat_shards_empty_owners_and_replicas_follow_declared_groups(
    tmp_path, elements, replicated, weights
):
    sources = []
    whole = torch.arange(elements, dtype=torch.float32) / 7 + 0.12345
    for state in flat_states(elements, replicated):
        worker = HostRank(state)
        for chunk in state.tensor_chunks():
            value = worker.values[chunk.name]
            value.copy_(
                whole[chunk.region_offset : chunk.region_offset + value.numel()]
            )
        path = tmp_path / f"rank-{state.logical_rank}.ssp"
        with path.open("wb") as stream:
            write_weights(worker, stream, chunk_bytes=17, weights=weights)
        sources.append((state, path))
    target = flat_states(elements, not replicated)
    with convert_weights(
        sources, target, tmp_path, scratch_bytes=128, max_disk_bytes=4096
    ) as converted:
        expected = whole if weights == "master" else whole.bfloat16().float()
        for rank in converted.ranks:
            values = exported(rank)
            for key, chunk in weights_tensors([rank.state]).items():
                actual = values.get(key[1], torch.empty(0, dtype=chunk.dtype))
                torch.testing.assert_close(
                    actual,
                    expected[
                        chunk.region_offset : chunk.region_offset + actual.numel()
                    ].to(chunk.dtype),
                    rtol=0,
                    atol=0,
                )

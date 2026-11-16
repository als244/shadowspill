"""Owned weights conversion uses the existing isolated CPU publication process."""

import hashlib
import io
from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch

from engine_tests.checkpoint.test_conversion import WEIGHT, HostRank
from engine_tests.checkpoint.test_weights_load import HostExecutor
from engine_tests.server.test_artifacts import reference
from engine_tests.server.test_checkpoint_conversion import configured, execute
from shadowspill.artifacts._values import plain
from shadowspill.checkpoint import weights_size, write_weights
from shadowspill.checkpoint.identity import state_identity
from shadowspill.checkpoint.preparation import plan_states
from shadowspill.checkpoint.weights_load import load_weights
from shadowspill.server import checkpoint_conversion as module
from shadowspill.server.checkpoints import publish_checkpoint
from shadowspill.server.weights_initialization import ResolvedWeights, resolve_weights
from shadowspill.training.preparation.bundle import TrainingBundle
from shadowspill.training.worker.checkpoint_records import RankSave, RankSaved
from shadowspill.training.worker.conversion_records import ConversionCommand
from shadowspill_client import Optimizer, Weights
from shadowspill_client.protocol.codec import encode
from shadowspill_client.protocol.requests import ArtifactRef


@contextmanager
def exported(
    tmp_path, *, counts=(2, 4), weights="master", partial=False, bad_metadata=False
):
    with configured(
        tmp_path,
        source_count=counts[0],
        target_count=counts[1],
        weights=weights,
        frozen=partial,
    ) as (artifacts, original, targets, kwargs):
        info = replace(original.manifest.info, kind="weights", request_id="weights")
        with artifacts.open(info.plans) as stream:
            sources = plan_states(TrainingBundle.load(stream).plans)
        ranks = [
            RankSave(
                state.logical_rank,
                weights_size(
                    state,
                    metadata=info.rank_metadata(),
                    weights=weights,
                    frozen_state_id=info.frozen_state_id,
                ),
                state_identity(state),
                state.plan.program.digest,
                {},
                kind="weights",
            )
            for state in sources
        ]

        def write(rank, sink):
            # Tiny test payloads use the real exporter; production conversion
            # below runs in a separate process with admitted streaming budgets.
            body = io.BytesIO()
            manifest = write_weights(
                HostRank(sources[rank.rank], initialize=True),
                body,
                metadata=info.rank_metadata(),
                weights=weights,
                frozen_state_id=info.frozen_state_id,
                chunk_bytes=17,
            )
            raw = body.getvalue()
            for offset in range(0, len(raw), 64 << 10):
                sink.write(memoryview(raw)[offset : offset + (64 << 10)])
            return RankSaved(hashlib.sha256(raw).hexdigest(), manifest)

        reference_weights = publish_checkpoint(artifacts, info, ranks, write)
        config = replace(
            kwargs["config"],
            model=replace(
                kwargs["config"].model,
                initialize=Weights.artifact(reference_weights.id),
            ),
            optimizer=Optimizer("fresh/adam@1", defaults={"lr": 0.04}),
        )
        updated = {}
        for rank, plans in targets.items():
            members = {}
            for name, plan in plans.items():
                program = plan.program
                rows = [
                    {
                        **row,
                        **({"device": "cpu"} if row["object_id"] is None else {}),
                        **(
                            {"initialization": {"kind": "constant", "value": 0.04}}
                            if row["role"] == "hyperparameter"
                            else {}
                        ),
                    }
                    for row in program.bindings["optimizer_bindings"]
                ]
                program = replace(
                    program,
                    bindings={
                        **program.bindings,
                        "optimizer_type": "different.FreshAdam",
                        "optimizer_bindings": rows,
                    },
                )
                members[name] = replace(
                    plan,
                    profiled_program=replace(plan.profiled_program, program=program),
                )
            updated[rank] = members
        body = encode(config)
        config_ref = reference(body, "fresh-config")
        artifacts.put(config_ref, lambda _: io.BytesIO(body))
        target = TrainingBundle(config, {}, updated, {})
        path = tmp_path / "fresh-target.ssp"
        target.save(path)
        plans_ref = ArtifactRef(
            "fresh-plans",
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_size,
        )
        artifacts.put(plans_ref, lambda _: path.open("rb"))
        source = resolve_weights(artifacts, reference_weights, config)
        if bad_metadata:
            source = replace(
                source,
                manifest=replace(
                    source.manifest, info=replace(info, completed_updates=18)
                ),
            )
        yield (
            artifacts,
            source,
            updated,
            {
                **kwargs,
                "plans": plans_ref,
                "configuration": config_ref,
                "config": config,
            },
        )


@pytest.mark.parametrize("counts", [(1, 4), (4, 1), (2, 4)])
@pytest.mark.parametrize("weights", ["master", "compute"])
def test_cpu_weights_preparation_preserves_precision_and_initializes_fresh_state(
    tmp_path, monkeypatch, counts, weights
):
    with exported(tmp_path, counts=counts, weights=weights) as (
        artifacts,
        source,
        target,
        kwargs,
    ):
        result = execute(artifacts, source, kwargs)
        assert isinstance(result, ResolvedWeights)
        info = result.manifest.info
        assert info.kind == "weights" and info.conversion is None
        assert info.completed_updates == 17 and info.world_size == counts[1]
        assert (
            info.plans == kwargs["plans"]
            and info.configuration == kwargs["configuration"]
        )
        selected = WEIGHT if weights == "master" else WEIGHT.bfloat16().float()
        for state, row in zip(plan_states(target), result.manifest.ranks, strict=True):
            restored = HostExecutor(state)
            with artifacts.local_path(row.archive) as path:
                assert (
                    plain(load_weights(restored, path, chunk_bytes=17))
                    == info.rank_metadata()
                )
            values = restored.snapshot()
            for item in state.logical_tensors(roles=("parameter", "master")):
                expected = selected[
                    tuple(
                        slice(start, start + size)
                        for start, size in zip(
                            item.view.origin, item.view.local_shape, strict=True
                        )
                    )
                ]
                torch.testing.assert_close(
                    values[item.archive.name].reshape(item.view.local_shape),
                    expected.to(item.archive.dtype),
                    rtol=0,
                    atol=0,
                )
            assert restored.controls.host["optimizer.linear.weight.step"].item() == 0
            assert restored.controls.host[
                "optimizer_group.0.lr"
            ].item() == pytest.approx(0.04)
            assert not restored.values["moment"].count_nonzero()
            assert not restored.values["variance"].count_nonzero()
        send = module.write_message

        def forbid_materialization(stream, message):
            assert (
                not isinstance(message, ConversionCommand) or message.action != "write"
            )
            return send(stream, message)

        monkeypatch.setattr(module, "write_message", forbid_materialization)
        before = artifacts.used
        assert execute(artifacts, source, kwargs) == result
        assert artifacts.used == before


def test_partial_weights_conversion_preserves_exact_base_dependency(tmp_path):
    with exported(tmp_path, partial=True) as (artifacts, source, target, kwargs):
        result = execute(artifacts, source, kwargs)
        assert result.manifest.info.frozen_base == source.manifest.info.frozen_base
        for state, row in zip(plan_states(target), result.manifest.ranks, strict=True):
            restored = HostExecutor(state)
            region = next(
                item
                for item in state.logical_tensors()
                if item.tensor.identity.name == "base.weight"
            )
            expected = WEIGHT[
                tuple(
                    slice(start, start + size)
                    for start, size in zip(
                        region.view.origin, region.view.local_shape, strict=True
                    )
                )
            ].contiguous()

            def base(request, destination, expected=expected):
                assert request.name == "base.weight"
                destination.copy_(
                    expected.flatten()[
                        request.offset : request.offset + destination.numel()
                    ]
                )

            with artifacts.local_path(row.archive) as path:
                load_weights(
                    restored,
                    path,
                    chunk_bytes=17,
                    frozen_state_id=result.manifest.info.frozen_state_id,
                    fill_tensor=base,
                )
            torch.testing.assert_close(
                restored.snapshot()["base.weight"], expected.flatten(), rtol=0, atol=0
            )


@pytest.mark.parametrize(
    "failure", ["disk", "files", "host", "publication", "metadata", "cancel"]
)
def test_weights_conversion_failure_never_publishes_a_partial_generation(
    tmp_path, monkeypatch, failure
):
    with exported(tmp_path, bad_metadata=failure == "metadata") as (
        artifacts,
        source,
        _,
        kwargs,
    ):
        if failure == "disk":
            kwargs["max_disk_bytes"] = 1
        elif failure == "files":
            kwargs["max_files"] = 1
        elif failure == "host":
            kwargs["host_bytes"] = 1
        elif failure == "publication":
            original = module.publish_checkpoint

            def fail(artifacts, info, ranks, write):
                def last(rank, writer):
                    if rank.rank == 3:
                        raise OSError("final weights rank unavailable")
                    return write(rank, writer)

                return original(artifacts, info, ranks, last)

            monkeypatch.setattr(module, "publish_checkpoint", fail)
        before = artifacts.used
        with pytest.raises((ValueError, OSError, RuntimeError)):
            execute(artifacts, source, kwargs, cancel=failure == "cancel")
        assert artifacts.used == before and kwargs["memory"].used == (0, 0)

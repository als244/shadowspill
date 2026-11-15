"""Selected export skips spill transfers and CPU controls before touching bytes."""

from dataclasses import replace
from threading import Lock
from types import SimpleNamespace

import pytest
import torch

from engine_tests.checkpoint.test_staging import descriptor
from engine_tests.execution.test_shard_initialization import fixture, host_storage
from shadowspill.checkpoint.conversion import ConvertedRank
from shadowspill.checkpoint.staging import stage_tensors, staging_bytes
from shadowspill.execution.controls import Controls
from shadowspill.execution.executor import Executor
from shadowspill.execution.tensors import visit_tensors


def executor(rank=0):
    program = fixture()
    bindings = dict(program.bindings)
    bindings["registrations"] = [
        {**row, "parameter": True} for row in bindings["registrations"]
    ]
    bindings["optimizer_bindings"] = [
        *bindings["optimizer_bindings"],
        *[
            dict(
                name=name,
                role=role,
                object_id=None,
                device="cpu",
                dtype="torch.float32",
                shape=(),
                stride=(),
            )
            for name, role in (("counter", "state"), ("lr", "hyperparameter"))
        ],
    ]
    program = replace(program, bindings=bindings)
    primary, values = host_storage(program, rank)
    primary_controls = Controls(primary)
    for tensor in primary_controls.host.values():
        tensor.fill_(3)
    extra, _ = host_storage(program, rank)
    extra.shared_aliases = dict.fromkeys(extra.sizes)
    extra_controls = Controls(
        extra, shared_host={"counter": primary_controls.host["counter"]}
    )
    extra_controls.host["lr"].fill_(9)
    result = object.__new__(Executor)
    result._paths = [SimpleNamespace(storage=primary, controls=primary_controls)]
    result._named_paths = {
        "eval": SimpleNamespace(storage=extra, controls=extra_controls)
    }
    result._initialized = True
    result.synchronize = lambda: None
    return result, primary, extra, values


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_selection_preserves_owned_offsets_and_never_reads_unselected_storage(rank):
    worker, primary, extra, values = executor(rank)
    result = []
    worker.export_tensors(
        lambda chunk, value: result.append((chunk, value.clone())),
        chunk_bytes=12,
        names={"master.layer.weight", "plans.eval.lr"},
    )
    assert {chunk.name for chunk, _ in result} == {
        "master.layer.weight",
        "plans.eval.lr",
    }
    assert {row[0] for row in primary.spill.transfers} == {"master"}
    assert not extra.spill.transfers
    assert all(row[2] <= 12 and row[3] == "read" for row in primary.spill.transfers)
    owned = next(
        chunk for chunk in worker.tensor_chunks() if chunk.name == "master.layer.weight"
    )
    assert result[0][0].offset == owned.region_offset
    selected = torch.cat(
        [value for chunk, value in result if chunk.name == "master.layer.weight"]
    )
    torch.testing.assert_close(
        selected, values["master"][: owned.region_elements], rtol=0, atol=0
    )
    assert result[-1][1].item() == 9


@pytest.mark.parametrize("names", [set(), {"missing"}, "master.layer.weight", {7}])
def test_empty_or_invalid_selection_performs_no_tensor_io(names):
    worker, primary, extra, _ = executor()
    if names:
        with pytest.raises((ValueError, TypeError)):
            worker.export_tensors(
                lambda *_: pytest.fail("exported invalid selection"), names=names
            )
    else:
        worker.export_tensors(
            lambda *_: pytest.fail("exported empty selection"), names=names
        )
    assert not primary.spill.transfers and not extra.spill.transfers


def test_default_export_keeps_all_state_and_initialization_cannot_be_filtered():
    worker, primary, _, _ = executor()
    observed = set()
    worker.export_tensors(lambda chunk, _: observed.add(chunk.name), chunk_bytes=12)
    assert observed == {chunk.name for chunk in worker.tensor_chunks()}
    assert {row[0] for row in primary.spill.transfers} == {"weight", "master", "moment"}
    with pytest.raises(ValueError, match="only to export"):
        visit_tensors(primary, lambda *_: None, 12, initialize=True, names=set())


def test_converted_archive_selection_never_opens_unselected_files(
    tmp_path, monkeypatch
):
    chunks = {(0, name): descriptor(name, count=6) for name in ("weight", "optimizer")}
    with stage_tensors(
        chunks, tmp_path, scratch_bytes=25, max_disk_bytes=staging_bytes(chunks)
    ) as staged:
        for key in chunks:
            staged.write(key, 0, torch.arange(6).float())
        staged.seal()
        paths = staged._path

        def path(key, suffix):
            assert key == (0, "weight")
            return paths(key, suffix)

        monkeypatch.setattr(staged, "_path", path)
        rank = ConvertedRank(
            SimpleNamespace(
                logical_rank=0, tensor_chunks=lambda: iter(chunks.values())
            ),
            staged,
            Lock(),
        )
        collected = []
        rank.export_tensors(
            lambda chunk, value: collected.append(value.clone()),
            chunk_bytes=12,
            names={"weight"},
        )
        torch.testing.assert_close(
            torch.cat(collected), torch.arange(6).float(), rtol=0, atol=0
        )
        with pytest.raises(ValueError, match="unknown export tensors"):
            rank.export_tensors(lambda *_: None, chunk_bytes=12, names={"unknown"})

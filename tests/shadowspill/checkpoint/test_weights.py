"""Authoritative copies, physical layouts and real owned-shard descriptors."""

import json
import tarfile
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from shadowspill import load_checkpoint, save_checkpoint
from shadowspill.checkpoint.weights import select_weights
from shadowspill.execution.tensors import tensor_descriptors, visit_tensors
from tests.shadowspill.checkpoint.archive_helpers import write_contents
from tests.shadowspill.execution.test_checkpoint import MemoryExecutor
from tests.shadowspill.execution.test_shard_initialization import fixture, host_storage


def mixed_executor():
    executor = MemoryExecutor()
    master = (torch.arange(15, dtype=torch.float32) / 7).reshape(3, 5).t()
    executor.values = {
        "weight": master.bfloat16(),
        "master.weight": master.clone(),
        "moment": torch.full_like(master, 0.25),
    }
    executor.plan.program.bindings = {
        "registrations": [
            {"name": "weight", "object_id": "w", "component_path": ()},
            {"name": "tied.weight", "object_id": "w", "component_path": ()},
        ],
        "optimizer_bindings": [
            {
                "name": "weight",
                "role": "parameter",
                "parameter_name": "weight",
                "object_id": "m",
                "initialization": {"kind": "parameter", "name": "weight"},
            },
            {
                "name": "compute.weight",
                "role": "compute_copy",
                "parameter_name": "weight",
                "object_id": "w",
            },
        ],
    }
    executor.plan.program.objects = [
        SimpleNamespace(object_id=name, alias_group_id=name) for name in ("w", "m")
    ]
    return executor


def manifest(path):
    with tarfile.open(path) as archive:
        return json.loads(archive.extractfile("manifest.json").read())


def rewrite(path, alter):
    with tarfile.open(path) as archive:
        contents = {
            name: archive.extractfile(name).read() for name in archive.getnames()
        }
    value = json.loads(contents["manifest.json"])
    alter(value, contents)
    contents["manifest.json"] = json.dumps(value).encode()
    write_contents(path, contents)


@pytest.mark.parametrize("weights", ["master", "compute"])
def test_selected_copy_roundtrip_strided_storage_and_tied_registration(
    tmp_path, weights
):
    executor = mixed_executor()
    expected = {name: value.clone() for name, value in executor.values.items()}
    if weights == "compute":
        expected["master.weight"] = expected["weight"].float()
        assert not torch.equal(
            expected["master.weight"], executor.values["master.weight"]
        )
    path = tmp_path / "one-copy.ssp"
    save_checkpoint(executor, path, weights=weights, chunk_bytes=11)
    info = manifest(path)
    selected, derived = (
        ("master.weight", "weight")
        if weights == "master"
        else ("weight", "master.weight")
    )
    assert info["derived_tensors"] == {derived: selected}
    assert info["retained_weight_copies"] == {}
    assert {entry["name"] for entry in info["tensors"]} == {selected, "moment"}
    assert sum(entry["bytes"] for entry in info["tensors"]) == sum(
        executor.values[name].nbytes for name in (selected, "moment")
    )
    for value in executor.values.values():
        value.zero_()
    load_checkpoint(executor, path, chunk_bytes=7)
    torch.testing.assert_close(executor.values, expected, rtol=0, atol=0)


@pytest.mark.parametrize("weights", ["unknown", None, {}])
def test_invalid_selection_is_rejected_before_creating_archive(tmp_path, weights):
    with pytest.raises(ValueError, match="weights must"):
        save_checkpoint(mixed_executor(), tmp_path / "invalid.ssp", weights=weights)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("alter", ["source", "missing", "policy"])
def test_reconstruction_manifest_must_match_receiving_bindings(tmp_path, alter):
    executor = mixed_executor()
    path = tmp_path / "one-copy.ssp"
    save_checkpoint(executor, path)

    def change(info, _contents):
        if alter == "source":
            info["derived_tensors"]["weight"] = "moment"
        elif alter == "missing":
            del info["derived_tensors"]
        else:
            info["weights"] = "compute"

    rewrite(path, change)
    with pytest.raises(ValueError, match="reconstruction differs"):
        load_checkpoint(executor, path)
    assert executor.initialized


def test_corrupted_selected_master_cannot_publish_derived_state(tmp_path):
    executor = mixed_executor()
    path = tmp_path / "one-copy.ssp"
    save_checkpoint(executor, path)

    def change(info, contents):
        entry = next(
            item for item in info["tensors"] if item["name"] == "master.weight"
        )
        data = contents[entry["member"]]
        contents[entry["member"]] = bytes([data[0] ^ 1]) + data[1:]

    rewrite(path, change)
    with pytest.raises(ValueError, match="checksum differs"):
        load_checkpoint(executor, path, chunk_bytes=9)
    assert not executor.initialized


def test_encoded_or_different_storage_layouts_remain_explicit(tmp_path):
    executor = mixed_executor()
    executor.plan.program.bindings["registrations"][0]["component_path"] = ("payload",)
    selection = select_weights(executor, "master")
    assert not selection.derived
    assert (
        selection.retained["weight"]
        == "no ordinary scalar-cast reconstruction contract"
    )
    executor = mixed_executor()
    executor.values["master.weight"] = executor.values["master.weight"].contiguous()
    path = tmp_path / "layouts.ssp"
    save_checkpoint(executor, path)
    info = manifest(path)
    assert not info["derived_tensors"]
    assert (
        info["retained_weight_copies"]["weight"]
        == "physical weight layouts cannot use a storage cast"
    )
    assert len(info["tensors"]) == 3


def test_nonlogical_backing_elements_remain_independent(tmp_path):
    executor = mixed_executor()
    master = torch.arange(30, dtype=torch.float32)
    compute = torch.full((30,), -5, dtype=torch.bfloat16)
    master_view = master.as_strided((3, 5), (10, 1))
    compute_view = compute.as_strided((3, 5), (10, 1))
    compute_view.copy_(master_view)
    executor.values = {"weight": compute_view, "master.weight": master_view}
    path = tmp_path / "backing.ssp"
    save_checkpoint(executor, path)
    expected_compute, expected_master = compute.clone(), master.clone()
    compute.zero_()
    master.zero_()
    load_checkpoint(executor, path)
    torch.testing.assert_close(compute, expected_compute, rtol=0, atol=0)
    torch.testing.assert_close(master, expected_master, rtol=0, atol=0)
    assert not manifest(path)["derived_tensors"]


class OwnedExecutor:
    """Use real exported descriptors/leases while keeping I/O tests on the CPU."""

    def __init__(self, program, rank):
        self.storage, self.values = host_storage(program, rank)
        self.plan, self.plans = self.storage.plan, {}
        self.logical_rank, self.initialized = rank, True

    def tensor_chunks(self):
        return tensor_descriptors(self.storage)

    def export_tensors(self, callback, *, chunk_bytes):
        visit_tensors(self.storage, callback, chunk_bytes, initialize=False)

    def initialize(self, callback, *, chunk_bytes, restore, replicated_copies=None):
        assert not replicated_copies
        self.initialized = False
        visit_tensors(
            self.storage, callback, chunk_bytes, initialize=True, restore=restore
        )
        self.initialized = True


@pytest.mark.parametrize(
    "replicated,weights", [(False, "master"), (False, "compute"), (True, "compute")]
)
@pytest.mark.parametrize("rank", [0, 1])
def test_owned_tail_shards_and_replicated_compute_copies(
    tmp_path, rank, weights, replicated
):
    program = fixture()
    if replicated:
        program = replace(program, bindings={**program.bindings, "weight_shards": {}})
    executor = OwnedExecutor(program, rank)
    regions = executor.storage.regions
    for name, tensor in executor.values.items():
        region = regions[name]
        tensor[: region.length].copy_(
            torch.arange(region.offset, region.offset + region.length) + 0.03125
        )
    expected = {name: value.clone() for name, value in executor.values.items()}
    if weights == "compute":
        source, target = regions["weight"], regions["master"]
        start = target.offset - source.offset
        expected["master"][: target.length].copy_(
            expected["weight"][start : start + target.length]
        )
    path = tmp_path / "shard.ssp"
    save_checkpoint(executor, path, weights=weights, chunk_bytes=11)
    info = manifest(path)
    assert len(info["derived_tensors"]) == 1
    assert not info["gathered_tensors"]
    for value in executor.values.values():
        value.fill_(-7)
    load_checkpoint(executor, path, chunk_bytes=13)
    for name, value in executor.values.items():
        length = regions[name].length
        torch.testing.assert_close(
            value[:length], expected[name][:length], rtol=0, atol=0
        )
        assert not value[length:].count_nonzero()


def test_empty_owned_regions_do_not_require_payloads(tmp_path):
    executor = OwnedExecutor(fixture(elements=1), rank=0)
    assert all(not region.length for region in executor.storage.regions.values())
    path = tmp_path / "empty.ssp"
    save_checkpoint(executor, path)
    load_checkpoint(executor, path, chunk_bytes=8)
    assert all(not value.count_nonzero() for value in executor.values.values())

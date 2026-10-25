"""Partial archives retain mutations/buffers and require an explicit frozen base."""

import json
import zipfile
from types import SimpleNamespace

import pytest
import torch

from shadowspill import load_checkpoint, save_checkpoint
from tests.shadowspill.execution.test_checkpoint import MemoryExecutor


def executor():
    result = MemoryExecutor()
    result.values = {
        name: torch.arange(6, dtype=torch.float32) + index
        for index, name in enumerate(
            ("base", "delta", "moving", "calls", "master.delta", "optimizer.delta")
        )
    }
    result.plan.program.bindings = {
        "registrations": [
            dict(
                name=name,
                object_id=name,
                parameter=name != "calls",
                trainable=name == "delta",
            )
            for name in ("base", "delta", "moving", "calls")
        ]
        + [dict(name="tied_base", object_id="base", parameter=True, trainable=False)]
    }
    result.plan.program.objects = [
        SimpleNamespace(object_id=name, alias_group_id=name) for name in result.values
    ]
    result.plan.program.tasks = [
        SimpleNamespace(outputs=(), mutations=[SimpleNamespace(object_id="moving")])
    ]
    return result


def test_partial_restore_fills_only_frozen_base_and_preserves_all_other_state(tmp_path):
    value = executor()
    expected = {name: tensor.clone() for name, tensor in value.values.items()}
    path = tmp_path / "adapters.ssp"
    save_checkpoint(value, path, chunk_bytes=7, frozen_state_id="base-v1")
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["omitted_parameters"] == ["base", "tied_base"]
    assert {item["name"] for item in manifest["tensors"]} == expected.keys() - {"base"}
    for tensor in value.values.values():
        tensor.zero_()
    names = []

    def fill(chunk, destination):
        names.append(chunk.name)
        assert destination.nbytes <= 13
        destination.copy_(
            expected[chunk.name][chunk.offset : chunk.offset + destination.numel()]
        )

    load_checkpoint(
        value, path, chunk_bytes=13, frozen_state_id="base-v1", fill_tensor=fill
    )
    assert set(names) == {"base"}
    torch.testing.assert_close(value.values, expected, rtol=0, atol=0)


@pytest.mark.parametrize("identity", [None, "wrong-base"])
def test_identity_mismatch_fails_before_loading_any_tensors(tmp_path, identity):
    value = executor()
    path = tmp_path / "adapters.ssp"
    save_checkpoint(value, path, frozen_state_id="base-v1")
    with pytest.raises(ValueError, match="frozen_state_id"):
        load_checkpoint(value, path, frozen_state_id=identity)
    assert value.initialized and not value.values["base"].eq(0).all()


def test_partial_restore_never_assumes_current_base_is_correct(tmp_path):
    value = executor()
    path = tmp_path / "adapters.ssp"
    save_checkpoint(value, path, frozen_state_id="base-v1")
    with pytest.raises(ValueError, match="requires fill_tensor"):
        load_checkpoint(value, path, frozen_state_id="base-v1")
    assert value.initialized


def test_trainable_alias_prevents_omitting_the_shared_storage(tmp_path):
    value = executor()
    value.plan.program.bindings["registrations"][-1]["trainable"] = True
    path = tmp_path / "adapters.ssp"
    save_checkpoint(value, path, frozen_state_id="base-v1")
    with zipfile.ZipFile(path) as archive:
        assert json.loads(archive.read("manifest.json"))["omitted_parameters"] == []


def test_old_program_cannot_claim_missing_trainability_is_frozen(tmp_path):
    value = executor()
    del value.plan.program.bindings["registrations"][0]["trainable"]
    with pytest.raises(ValueError, match="trainability metadata"):
        save_checkpoint(value, tmp_path / "adapters.ssp", frozen_state_id="base-v1")


def test_missing_mutable_tensor_is_not_delegated_to_base_loader(tmp_path):
    value = executor()
    path = tmp_path / "adapters.ssp"
    save_checkpoint(value, path, frozen_state_id="base-v1")
    with zipfile.ZipFile(path) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(contents["manifest.json"])
    removed = next(item for item in manifest["tensors"] if item["name"] == "moving")
    manifest["tensors"].remove(removed)
    del contents[removed["member"]]
    contents["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="geometry differs: moving"):
        load_checkpoint(
            value, path, frozen_state_id="base-v1", fill_tensor=lambda r, t: t.zero_()
        )
    assert not value.initialized

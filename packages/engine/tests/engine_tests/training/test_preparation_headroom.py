"""Configured worker reserve reaches profiling and cannot change on saved reuse."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine_tests.training.test_allocation_coverage import certify, compiler_fixture
from engine_tests.training.test_preparation_templates import inventory
from engine_tests.training.test_training_bundle import make_bundle
from shadowspill.training.preparation import profile
from shadowspill.training.preparation.family_identity import validate_profile
from shadowspill_client import Preparation


@pytest.mark.parametrize("headroom", [0, 512 << 20, 768 << 20])
def test_exact_worker_reserve_reaches_the_existing_profiler(monkeypatch, headroom):
    inv = inventory(prepare=Preparation(external_headroom_bytes=headroom))
    key, template = next(iter(inv.templates.items()))
    program = certify(template, compiler_fixture(template))
    calls = []
    measured = SimpleNamespace(metadata={})

    def measure(source, **kwargs):
        assert source is program
        calls.append(kwargs)
        return measured

    monkeypatch.setattr(profile, "measure_program", measure)
    monkeypatch.setattr(profile, "apply_profiles", lambda source, result: result)
    assert profile.profile_template(inv, key, program) is measured
    assert measured.metadata["preparation_template"] == key
    assert calls[0]["external_headroom_bytes"] == headroom
    assert (
        calls[0]["options"].to_dict()
        == inv.layouts.setup.config.preparation.profiling.to_dict()
    )


@pytest.mark.parametrize("headroom", [None, 512 << 20, 768 << 20])
def test_saved_profile_cannot_omit_or_change_its_worker_reserve(monkeypatch, headroom):
    bundle, _ = make_bundle(monkeypatch)
    inv = inventory(prepare=Preparation(external_headroom_bytes=0))
    key, source = next(iter(bundle.sources.items()))
    validate_profile(inv, key, source)
    metadata = dict(source.metadata)
    if headroom is None:
        del metadata["external_headroom_bytes"]
    else:
        metadata["external_headroom_bytes"] = headroom
    with pytest.raises(ValueError, match="profiling policy"):
        validate_profile(
            inv, key, replace(source, metadata=metadata, fixed_device_bytes=1024 << 20)
        )

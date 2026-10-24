from dataclasses import FrozenInstanceError, replace

import pytest

from shadowspill import Profiles, TaskProfile, apply_profiles

from ._example import profiles, program


def test_program_has_structure_without_fake_timings():
    value = program().to_dict()
    assert "profiles" not in value
    assert "duration_seconds" not in str(value)
    assert value["task_definitions"][0]["workspace_bytes"] is None
    with pytest.raises(FrozenInstanceError):
        program().name = "changed"


def test_graph_validation_is_shared_with_existing_ir():
    p = program()
    with pytest.raises(ValueError, match="unknown profile"):
        replace(p, tasks=(replace(p.tasks[0], profile_id="missing"),))


def test_missing_memory_is_not_zero():
    with pytest.raises(ValueError) as error:
        apply_profiles(program(), Profiles(tasks={"update_impl": TaskProfile(1e-6)}))
    assert "workspace_bytes" in str(error.value)
    assert "fixed_device_bytes" in str(error.value)


def test_static_memory_requires_only_duration():
    p = program(workspace=16, overhead=20)
    result = apply_profiles(
        p,
        Profiles(tasks={"update_impl": TaskProfile(1e-6), "boundary": TaskProfile(0)}),
    )
    assert result.estimated_memory == ()
    assert result.task_profiles["update_impl"].workspace_bytes == 16
    with pytest.raises(ValueError, match="smaller than compiler"):
        apply_profiles(p, Profiles(tasks={"update_impl": TaskProfile(1e-6, 0)}))


def test_missing_durations_are_not_estimated_implicitly():
    with pytest.raises(ValueError, match="duration_seconds"):
        apply_profiles(program(workspace=0, overhead=0), Profiles())


def test_profile_snapshot_and_metadata_are_independent():
    inputs = profiles()
    inputs.setup["gpu"] = {"name": "H100"}
    result = apply_profiles(program(), inputs)
    inputs.tasks["update_impl"] = TaskProfile(100, 100)
    inputs.setup["gpu"]["name"] = "different"
    result.profiles.tasks.clear()
    assert result.task_profiles["update_impl"].duration_seconds == 1e-6
    assert result.setup["gpu"]["name"] == "H100"
    with pytest.raises(TypeError):
        result.setup["gpu"]["name"] = "mutate"
    with pytest.raises(TypeError):
        result.task_profiles["update_impl"] = TaskProfile(2)


def test_merge_overwrites_values_preserving_only_unchanged_evidence():
    original = Profiles(
        tasks={
            "t": TaskProfile(
                1.0,
                128,
                metadata={"duration_kind": "measured", "memory_kind": "measured"},
            )
        }
    )
    timed = original.merge(Profiles(tasks={"t": TaskProfile(2.0)}))
    assert timed.tasks["t"].workspace_bytes == 128
    assert timed.tasks["t"].metadata["memory_kind"] == "measured"
    assert timed.tasks["t"].metadata["duration_kind"] == "estimated"
    resized = timed.merge(Profiles(tasks={"t": TaskProfile(workspace_bytes=0)}))
    assert resized.tasks["t"].workspace_bytes == 0
    assert resized.tasks["t"].metadata["memory_kind"] == "estimated"
    assert original.tasks["t"].duration_seconds == 1.0


def test_incompatible_measurements_rejected():
    p = replace(program(), setup={"gpu": "H100"})
    with pytest.raises(ValueError, match="setup"):
        apply_profiles(p, profiles())
    data = profiles()
    data.tasks["update_impl"] = TaskProfile(1e-6, 16, compatibility_digest="different")
    with pytest.raises(ValueError, match="implementation/layout"):
        apply_profiles(program(), data)
    with pytest.raises(ValueError, match="setups"):
        profiles().merge(Profiles(setup={"gpu": "other"}))


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_invalid_numeric_profiles_rejected(value):
    with pytest.raises(ValueError):
        TaskProfile(duration_seconds=value)
    with pytest.raises(ValueError):
        TaskProfile(workspace_bytes=value)


@pytest.mark.parametrize("value", [-1, True, 21])
def test_worker_headroom_is_a_valid_part_of_fixed_overhead(value):
    data = profiles()  # Twenty bytes of total fixed overhead.
    data.metadata["external_headroom_bytes"] = value
    with pytest.raises(ValueError):
        apply_profiles(program(), data)

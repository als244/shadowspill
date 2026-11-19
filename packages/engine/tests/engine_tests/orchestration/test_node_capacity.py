"""Node daemon capacity uses actual local observations and explicit step assignment."""

import os

import pytest

from shadowspill.orchestration.launch import node_capacity
from shadowspill.orchestration.launch.admission import ResourcesUnavailable
from shadowspill.orchestration.launch.device_inventory import (
    DeviceInventory,
    DeviceObservation,
)
from shadowspill.orchestration.launch.host_memory import HostMemory, MemoryLimit
from shadowspill.orchestration.launch.node_capacity import (
    assigned_capacity,
    discover_capacity,
)
from shadowspill.orchestration.launch.slurm_job import SlurmJob, SlurmNode

UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
DEVICES = DeviceInventory("driver", (DeviceObservation(UUID, "assigned", 100, 75),))
MEMORY = HostMemory(1000, 800, (MemoryLimit("job/step", 700, 100),))
JOB = SlurmJob(123, os.getuid(), {"node-a": SlurmNode("node-a", 6, 500)})


def environment():
    return {
        "SLURM_JOB_ID": "123",
        "SLURM_STEP_ID": "2",
        "SLURM_GPUS_ON_NODE": "1",
        "SLURMD_NODENAME": "node-a",
        "CUDA_VISIBLE_DEVICES": "0",
        # Slurm's global GPU index does not identify CUDA ordinal zero.
        "SLURM_STEP_GPUS": "7",
        "SLURM_MEM_PER_CPU": "100",
        "SLURM_CPUS_ON_NODE": "6",
    }


def test_local_capacity_keeps_visible_uuid_and_all_host_allowances():
    capacity = assigned_capacity(
        "local", DEVICES, MEMORY, reserved_host_bytes=50, environment={}
    )
    assert capacity.host_bytes == 550  # cgroup availability 600, less fixed 50.
    assert capacity.devices[0].id == capacity.devices[0].visibility == UUID
    assert capacity.devices[0].capacity_bytes == 100  # Not the free-memory snapshot.
    assert capacity.devices[0].name == "assigned" and capacity.driver == "driver"
    assert capacity.cpu_count == len(os.sched_getaffinity(0))


def test_slurm_memory_cpu_and_global_gpu_indices_do_not_replace_local_facts():
    capacity = assigned_capacity(
        "node-a",
        DEVICES,
        MEMORY,
        reserved_host_bytes=50,
        environment=environment(),
        job=JOB,
    )
    assert capacity.host_bytes == 450  # Actual scheduler allocation 500, not 6 * 100.
    assert capacity.devices[0].visibility == UUID
    assert capacity.cpu_count == min(6, len(os.sched_getaffinity(0)))
    low = HostMemory(1000, 800, (MemoryLimit("job/step", 400, 100),))
    assert (
        assigned_capacity(
            "node-a",
            DEVICES,
            low,
            reserved_host_bytes=50,
            environment=environment(),
            job=JOB,
        ).host_bytes
        == 250
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("SLURM_JOB_ID", "124"),
        ("SLURM_JOB_ID", ""),
        ("SLURM_JOBID", "124"),
        ("SLURM_STEP_ID", "batch"),
        ("SLURM_STEP_ID", "extern"),
        ("SLURM_STEP_ID", ""),
        ("SLURM_GPUS_ON_NODE", "2"),
        ("SLURM_GPUS_ON_NODE", ""),
        ("SLURMD_NODENAME", "other"),
        ("CUDA_VISIBLE_DEVICES", ""),
    ],
)
def test_missing_or_mismatching_step_evidence_fails_instead_of_using_whole_node(
    field, value
):
    env = {**environment(), field: value}
    with pytest.raises(ValueError):
        assigned_capacity(
            "node-a", DEVICES, MEMORY, reserved_host_bytes=50, environment=env, job=JOB
        )


@pytest.mark.parametrize("job", [None, SlurmJob(123, os.getuid() + 1, JOB.nodes)])
def test_existing_step_requires_current_same_owner_allocation_facts(job):
    with pytest.raises(ValueError, match="matching"):
        assigned_capacity(
            "node-a",
            DEVICES,
            MEMORY,
            reserved_host_bytes=0,
            environment=environment(),
            job=job,
        )


def test_scheduler_facts_cannot_be_applied_to_a_local_or_controller_process():
    with pytest.raises(ValueError, match="assigned local step"):
        assigned_capacity(
            "node-a", DEVICES, MEMORY, reserved_host_bytes=0, environment={}, job=JOB
        )


@pytest.mark.parametrize(
    "env", [{"SLURMD_NODENAME": "node-a"}, {"SLURM_GPUS_ON_NODE": "1"}]
)
def test_partial_scheduler_context_does_not_fall_back_to_local_inventory(env):
    with pytest.raises(ValueError, match="SLURM_JOB_ID"):
        assigned_capacity(
            "node-a", DEVICES, MEMORY, reserved_host_bytes=0, environment=env
        )


def test_fixed_overhead_must_fit_and_unavailable_memory_never_becomes_capacity():
    with pytest.raises(ValueError, match="cannot fit"):
        assigned_capacity(
            "local", DEVICES, MEMORY, reserved_host_bytes=701, environment={}
        )
    with pytest.raises(ResourcesUnavailable, match="availability"):
        assigned_capacity(
            "local", DEVICES, MEMORY, reserved_host_bytes=601, environment={}
        )
    exhausted = HostMemory(1000, 800, (MemoryLimit("job/step", 400, 500),))
    assert (
        assigned_capacity(
            "local", DEVICES, exhausted, reserved_host_bytes=0, environment={}
        ).host_bytes
        == 0
    )


def test_discovery_uses_same_explicit_environment_and_checks_step_before_probe(
    tmp_path, monkeypatch
):
    events = []
    env = environment()

    def query(job_id, **options):
        assert job_id == JOB.job_id
        assert options == {"environment": env, "root": tmp_path, "timeout": 7}
        events.append("scheduler")
        return JOB

    def memory():
        events.append("memory")
        return MEMORY

    def probe(**options):
        assert options == {"environment": env, "root": tmp_path, "timeout": 7}
        events.append("devices")
        return DEVICES

    monkeypatch.setattr(node_capacity, "query_job", query)
    monkeypatch.setattr(node_capacity, "host_memory", memory)
    monkeypatch.setattr(node_capacity, "probe_devices", probe)
    assert (
        discover_capacity(
            environment=env, root=tmp_path, reserved_host_bytes=50, timeout=7
        ).host_bytes
        == 450
    )
    assert events == ["scheduler", "memory", "devices"]
    events.clear()
    env["SLURM_STEP_ID"] = "batch"
    with pytest.raises(ValueError, match="SLURM_STEP_ID"):
        discover_capacity(environment=env, root=tmp_path, reserved_host_bytes=50)
    assert not events

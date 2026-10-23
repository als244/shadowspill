"""Stage tracing must preserve storage and verify actual profiler capture."""

import sqlite3

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from benchmarking.stage_traces.report import verify
from benchmarking.stage_traces.sweep import describe_queue, visible_devices
from shadowspill.pytorch.capture.fake import fake_device_model
from shadowspill.pytorch.representations import materialize_meta_state
from workloads.models.glm53_flash.stage_trace import workload
from workloads.models.shared.glm.checkpoint import physical_state


def test_fake_glm_preserves_fp8_components_and_trainability():
    case = workload(
        sequence_length=128, batch_size=1, tiny=True, tiny_expert_storage="fp8_block"
    )

    def signature(model):
        return {
            name: (tuple(value.shape), value.dtype, value.requires_grad)
            for name, value in physical_state(model).items()
        }

    expected = signature(case.model)
    assert any(dtype == torch.float8_e4m3fn for _, dtype, _ in expected.values())
    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        materialize_meta_state(case.model)
        replica = fake_device_model(case.model, mode)
        assert signature(replica) == expected
        assert all(v.device.type == "cuda" for v in physical_state(replica).values())


def test_queue_has_one_assignment_per_stage_variant_geometry(tmp_path):
    manifests = [
        {
            "sequence_length": 8192,
            "batch_size": batch,
            "stages": [
                {
                    "name": "stage_0001",
                    "occurrences": [{"index": 1}, {"index": 2}],
                    "variants": [
                        {"name": variant, "artifact": f"/{batch}-{variant}.pt"}
                        for variant in ("save", "recompute")
                    ],
                }
            ],
        }
        for batch in (1, 2, 4, 8)
    ]
    cases = describe_queue(manifests, tmp_path)
    assert len(cases) == len({c["directory"] for c in cases}) == 8
    assert all(len(c["occurrences"]) == 2 for c in cases)


def test_visible_device_ordinals_respect_scheduler_mask(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first,GPU-second,GPU-third")
    assert visible_devices(["all"]) == ["GPU-first", "GPU-second", "GPU-third"]
    assert visible_devices(["2", "0"]) == ["GPU-third", "GPU-first"]


def test_report_requires_cuda_kernels_and_each_phase_range(tmp_path):
    path = tmp_path / "trace.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("create table NVTX_EVENTS (start integer, end integer, text text)")
        db.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (start integer)")
        db.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (1)")
        for label in ("iteration/0", "forward", "backward"):
            db.execute("insert into NVTX_EVENTS values (1, 2, ?)", (label,))
    assert verify(path, 1)["cuda_kernel_count"] == 1
    with sqlite3.connect(path) as db:
        db.execute("delete from NVTX_EVENTS where text='backward'")
    with pytest.raises(ValueError, match="phase-range counts"):
        verify(path, 1)


def test_gpu_metrics_match_physical_uuid_after_cuda_remapping(tmp_path):
    path = tmp_path / "metrics.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("create table NVTX_EVENTS (start integer, end integer, text text)")
        db.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (start integer)")
        db.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (1)")
        for label in ("iteration/0", "forward", "backward"):
            db.execute("insert into NVTX_EVENTS values (1, 2, ?)", (label,))
        db.execute("create table GPU_METRICS (typeId integer, timestamp integer)")
        db.execute(
            "create table TARGET_INFO_GPU_METRICS (typeId integer, metricName text)"
        )
        db.execute(
            "create table TARGET_INFO_GPU "
            "(id integer, uuid text, busLocation text, name text)"
        )
        # CUDA ordinal 0 is physical GPU 2, as in the real Nsight probe.
        db.execute("insert into GPU_METRICS values (281479271677954, 42)")
        db.execute(
            "insert into TARGET_INFO_GPU_METRICS values (281479271677954, 'SM Active')"
        )
        db.execute(
            "insert into TARGET_INFO_GPU values (2, 'gpu-two', '4c:00.0', 'H100')"
        )
    result = verify(path, 1, gpu_uuid="gpu-two")["gpu_metrics"]
    assert result["physical_gpu_id"] == 2
    assert result["sample_timestamps"] == 1
    with pytest.raises(ValueError, match="device mismatch"):
        verify(path, 1, gpu_uuid="gpu-zero")
    with sqlite3.connect(path) as db:
        db.execute("delete from GPU_METRICS")
    with pytest.raises(ValueError, match="expected metrics from one GPU"):
        verify(path, 1, gpu_uuid="gpu-two")

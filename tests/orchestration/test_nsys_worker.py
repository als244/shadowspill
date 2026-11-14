"""Per-rank collectors isolate metrics before any CUDA import or rendezvous."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shadowspill.orchestration.capture import nsys_worker
from shadowspill.orchestration.capture.nsys import NsysCapture
from shadowspill.orchestration.launch.processes import distributed_command


def test_rank_wrapper_goes_inside_torchrun_and_keeps_common_output(tmp_path):
    capture = NsysCapture(output_dir=tmp_path)
    command = distributed_command(
        capture.rank_command(["python", "worker.py"]), ranks=8
    )
    assert command.index("torch.distributed.run") < command.index(
        "shadowspill.orchestration.capture.nsys_worker"
    )
    assert command[-3:] == ["--", "python", "worker.py"]
    assert command[command.index("--output-dir") + 1] == str(tmp_path)
    assert json.loads(command[command.index("--config") + 1])["output_dir"] == str(
        tmp_path
    )
    with pytest.raises(ValueError, match="needs output_dir"):
        NsysCapture().rank_command(["python", "worker.py"])


@pytest.mark.parametrize(
    "mask,local,device",
    [("2,5,7,3", 2, "7"), ("GPU-abc", 2, "GPU-abc"), (None, 2, "2")],
)
def test_collector_preserves_global_and_local_rank_and_scopes_metrics(
    monkeypatch, tmp_path, mask, local, device
):
    monkeypatch.setenv("RANK", "10")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_RANK", str(local))
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)

    def launch(command, *, env, **options):
        assert env["CUDA_VISIBLE_DEVICES"] == device
        assert env["RANK"] == "10" and env["LOCAL_RANK"] == str(local)
        assert "--gpu-metrics-devices=cuda-visible" in command
        output = next(
            arg.split("=", 1)[1] for arg in command if arg.startswith("--output=")
        )
        assert Path(output).parent.name == "rank-00010"
        Path(output).with_suffix(".nsys-rep").write_bytes(b"captured")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(nsys_worker.subprocess, "run", launch)
    result = nsys_worker.collect(
        ["worker"], capture=NsysCapture(executable="/bin/true"), output_dir=tmp_path
    )
    assert result["status"] == "complete" and result["report_bytes"] == 8
    assert result["hostname"] and result["cuda_visible_devices"] == device
    assert result["original_cuda_visible_devices"] == mask
    assert result["local_rank"] == local and result["rank"] == 10
    assert result["content_verified"] is False


@pytest.mark.parametrize("returncode", [0, 17])
def test_collector_failure_or_missing_report_never_claims_success(
    monkeypatch, tmp_path, returncode
):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        nsys_worker.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=returncode),
    )
    with pytest.raises(
        RuntimeError, match="failed" if returncode else "nonempty report"
    ):
        nsys_worker.collect(
            ["worker"], capture=NsysCapture(executable="/bin/true"), output_dir=tmp_path
        )
    manifest = json.loads((tmp_path / "rank-00000/capture.json").read_text())
    assert manifest["status"] == "failed" and manifest["returncode"] == returncode
    assert manifest["finished_unix_ns"] >= manifest["started_unix_ns"]
    with pytest.raises(RuntimeError, match="incomplete"):
        nsys_worker.reports(tmp_path, ranks=1)

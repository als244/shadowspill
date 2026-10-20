"""What a row records about the code, the packages and the hardware."""

import hashlib
import re
import sys

from baselines.common.provenance import (
    ROOT,
    dispatch_record,
    environment_identity,
    hardware_identity,
    source_identity,
)


def test_source_identity_digests_every_python_file_under_the_paths():
    identity = source_identity(["baselines/common", "baselines/__init__.py"])
    files = identity["file_sha256"]
    assert "baselines/__init__.py" in files
    assert "baselines/common/geometry.py" in files
    assert all(name.endswith(".py") and "__pycache__" not in name for name in files)
    source = (ROOT / "baselines/common/geometry.py").read_bytes()
    assert files["baselines/common/geometry.py"] == hashlib.sha256(source).hexdigest()
    head = identity["git_head"]
    assert head is None or re.fullmatch(r"[0-9a-f]{40}", head)
    assert isinstance(identity["git_dirty"], bool)


def test_environment_identity_probes_the_interpreter_without_a_device():
    identity = environment_identity(sys.executable)
    major, minor = sys.version_info[:2]
    assert identity["python"].startswith(f"{major}.{minor}")
    assert "pytest" in identity["packages"]
    assert isinstance(identity["mlops_source"], dict)


def test_hardware_identity_names_the_host_and_the_device():
    identity = hardware_identity()
    assert identity["host"]
    assert identity["cpus"] >= 1
    assert identity["torch"]
    assert identity["device"]["count"] >= 0
    if identity["device"]["count"]:
        assert identity["device"]["name"]
        assert re.fullmatch(r"\d+\.\d+", identity["device"]["compute_capability"])


def test_dispatch_record_counts_what_ran_per_operation():
    record = dispatch_record({"matmul": {"cublas": 3, "triton": 1}, "softmax": {}})
    assert record == {"matmul": {"cublas": 3, "triton": 1}, "softmax": {}}

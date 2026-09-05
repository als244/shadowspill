"""The Python mirrors of the C ABI agree with the C headers.

A mirror that drifts is not a compile error and not a failed call. ctypes
writes the size it believes in, so a struct one field short of the C definition
corrupts the heap and the process dies somewhere unrelated, with no traceback
pointing anywhere near the mismatch. These checks are cheap and the failure
they prevent is not.
"""

from __future__ import annotations

import ctypes
import re
import subprocess
from pathlib import Path

import pytest

from shadowspill.pytorch.runtime_adapter import abi
from shadowspill.schema import ARTIFACT_VERSION
from shadowspill.status import ABI_VERSION, Status

ROOT = Path(__file__).resolve().parents[2]
_HEADERS = ROOT / "csrc" / "include" / "shadowspill"

#: Each ctypes structure and the C struct it mirrors.
_MIRRORS = {
    "ShadowSpillPytorchPoolConfig": abi.PoolConfig,
    "ShadowSpillPytorchRouteConfig": abi.RouteConfig,
    "ShadowSpillPytorchAdapterConfig": abi.AdapterConfig,
    "ShadowSpillPytorchAdapterStatistics": abi.AdapterStatistics,
    "ShadowSpillPytorchAdapterFailure": abi.AdapterFailure,
    "ShadowSpillRuntimeStatistics": abi.RuntimeStatistics,
    "ShadowSpillAllocationEvent": abi.AllocationEvent,
    "ShadowSpillTraceConfig": abi.TraceConfig,
    "ShadowSpillTransferRouteKey": abi.TransferRouteKey,
    "ShadowSpillTransferCalibrationConfig": abi.TransferCalibrationConfig,
    "ShadowSpillTransferProfile": abi.TransferProfile,
    "ShadowSpillTraceEvent": abi.TraceEvent,
    "ShadowSpillTraceSummary": abi.TraceSummary,
    "ShadowSpillAllocation": abi.Allocation,
    "ShadowSpillRuntimeFailure": abi.RuntimeFailure,
    "ShadowSpillObjectBinding": abi.ObjectBinding,
    "ShadowSpillObjectDescription": abi.ObjectDescription,
    "ShadowSpillObjectUpdate": abi.ObjectUpdate,
    "ShadowSpillRuntimeAction": abi.RuntimeAction,
    "ShadowSpillTaskPublicationDescription": abi.TaskPublicationDescription,
    "ShadowSpillTaskAllocationContractStep": abi.TaskAllocationContractStep,
    "ShadowSpillTaskDescription": abi.TaskDescription,
    "ShadowSpillFixedPlacementDescription": abi.FixedPlacementDescription,
    "ShadowSpillFixedDependencyDescription": abi.FixedDependencyDescription,
    "ShadowSpillFixedLayoutDescription": abi.FixedLayoutDescription,
    "ShadowSpillObjectSnapshot": abi.ObjectSnapshot,
    "ShadowSpillObjectLocationSnapshot": abi.ObjectLocationSnapshot,
    "ShadowSpillPytorchAdapterCapabilities": abi.AdapterCapabilities,
    "ShadowSpillPytorchPhysicalAdmission": abi.PhysicalAdmission,
    "ShadowSpillPytorchTaskDispatchTiming": abi.TaskDispatchTiming,
}


def _c_sizes() -> dict[str, int]:
    for directory in sorted(ROOT.glob("build/*")):
        probe = directory / "shadowspill_abi_sizes_canary"
        if not probe.is_file():
            continue
        completed = subprocess.run(
            (str(probe),), check=True, capture_output=True, text=True
        )
        return {
            name: int(size)
            for name, size in (
                line.split() for line in completed.stdout.splitlines() if line.strip()
            )
        }
    pytest.skip("the C size probe has not been built")


def test_every_ctypes_structure_matches_its_c_struct() -> None:
    sizes = _c_sizes()
    mismatched = {
        name: (sizes[name], ctypes.sizeof(mirror))
        for name, mirror in _MIRRORS.items()
        if name in sizes and sizes[name] != ctypes.sizeof(mirror)
    }
    assert not mismatched, f"ctypes mirrors disagree with C (c, python): {mismatched}"
    missing = sorted(set(_MIRRORS) - set(sizes))
    assert not missing, f"the size probe does not cover: {missing}"


def test_status_vocabulary_matches_the_header() -> None:
    header = (_HEADERS / "status.h").read_text()
    declared = {
        name: int(value)
        for name, value in re.findall(r"SHADOWSPILL_STATUS_([A-Z_]+) = (\d+)", header)
    }
    mirrored = {item.name: int(item.value) for item in Status}
    assert mirrored == declared


def test_every_status_decodes_to_a_sentence() -> None:
    decoder = (ROOT / "csrc" / "src" / "common" / "status.c").read_text()
    undecoded = [
        item.name for item in Status if f"SHADOWSPILL_STATUS_{item.name}" not in decoder
    ]
    assert not undecoded, f"statuses with no string: {undecoded}"


def test_every_failure_reason_decodes_to_a_sentence() -> None:
    header = (_HEADERS / "runtime.h").read_text()
    reasons = re.findall(r"SHADOWSPILL_FAILURE_REASON_([A-Z_]+) = (\d+)", header)
    assert reasons, "the failure reason vocabulary is empty"
    source = (ROOT / "csrc" / "src" / "runtime" / "failure_reason.c").read_text()
    undecoded = [
        name
        for name, _ in reasons
        if f"SHADOWSPILL_FAILURE_REASON_{name}" not in source
    ]
    assert not undecoded, f"reasons with no string: {undecoded}"


def test_the_abi_version_is_one_number() -> None:
    header = (_HEADERS / "shadowspill.h").read_text()
    declared = re.search(r"#define SHADOWSPILL_ABI_VERSION (\d+)U", header)
    assert declared is not None
    assert int(declared.group(1)) == ABI_VERSION


def test_the_artifact_version_is_one_number() -> None:
    header = (_HEADERS / "shadowspill.h").read_text()
    declared = re.search(r"#define SHADOWSPILL_ARTIFACT_VERSION (\d+)U", header)
    assert declared is not None
    assert int(declared.group(1)) == ARTIFACT_VERSION

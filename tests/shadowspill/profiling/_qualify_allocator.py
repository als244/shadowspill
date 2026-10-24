"""Qualification only: compare saved task observations against the runtime allocator."""

import ctypes
import sys
from functools import partial

import torch
from torch._inductor.standalone_compile import AOTCompiledArtifact

from shadowspill import Profiles, Program
from shadowspill.artifacts._values import plain
from shadowspill.libraries import resolve_library
from shadowspill.profiling.inputs import prepare_inputs
from shadowspill.pytorch.frontend import PyTorchFrontend
from shadowspill.pytorch.profiling.profiler.boundary import AllocatorBoundary
from shadowspill.pytorch.profiling.profiler.workspace import output_allocation_views
from shadowspill.runtime.abi import runtime_library
from shadowspill.runtime.bootstrap import install_runtime
from shadowspill.runtime.telemetry import (
    read_allocation_telemetry,
    start_allocation_telemetry,
    stop_allocation_telemetry,
    summarize_task_workspace,
)
from shadowspill.task.allocations import TaskAllocationContract
from shadowspill.task.storage import TaskStorageContract
from tests.integration.pytorch.runtime_helpers import two_pool_topology


def main():
    installed = install_runtime(
        resolve_library("libshadowspill_pytorch.so"),
        frontend=PyTorchFrontend(),
        device_ordinal=0,
        device_budget_bytes=2 << 30,
        external_headroom_bytes=512 << 20,
        numa_binding=False,
        **two_pool_topology(64 << 20),
    )
    plan_id = ctypes.c_uint64()
    assert (
        runtime_library().shadowspill_runtime_next_plan_id(
            installed.runtime_handle, ctypes.byref(plan_id)
        )
        == 0
    )
    boundary = AllocatorBoundary(
        installed.library,
        runtime_handle=installed.runtime_handle,
        plan_id=plan_id.value,
        device_ordinal=0,
        telemetry_capacity=131072,
    )
    program, profiles = Program.load(sys.argv[1]), Profiles.load(sys.argv[2])
    stream = boundary.stream()
    for key, payload in program.executables.items():
        function = AOTCompiledArtifact.deserialize(payload)
        arguments = prepare_inputs(program, key, torch.device("cuda", 0))
        boundary.invoke(partial(function, *arguments), stream)
        start_allocation_telemetry(installed.runtime_handle, capacity=131072)
        try:
            with boundary.scope(stream) as task_id:
                output = function(*arguments)
                views, bindings = output_allocation_views(
                    boundary, output, inputs=arguments
                )
                del output
            boundary.drain(stream, problem="allocator qualification")
        finally:
            stop_allocation_telemetry(installed.runtime_handle)
        observed = summarize_task_workspace(
            read_allocation_telemetry(installed.runtime_handle),
            task_id=task_id,
            output_allocation_views=views,
            output_input_bindings=bindings,
        )
        contract = TaskAllocationContract.capture(
            observed.allocation_contract_trace,
            TaskStorageContract.from_dict(
                plain(program.bindings["manifests"][key]["storage_contract"])
            ),
        )
        expected = profiles.tasks[key]
        assert observed.peak_charged_bytes == expected.workspace_bytes, (
            key,
            observed.peak_charged_bytes,
            expected.workspace_bytes,
        )
        assert contract.to_dict() == plain(expected.metadata["allocation_contract"]), (
            key,
            contract.to_dict(),
            plain(expected.metadata["allocation_contract"]),
        )
        assert [item.to_dict() for item in bindings] == plain(
            expected.metadata["output_input_bindings"]
        ), key
        del arguments, function
    print(f"Qualified {len(program.executables)} finalized task forms")


if __name__ == "__main__":
    with torch.no_grad():
        main()

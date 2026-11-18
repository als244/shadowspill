"""Default deployment admits full-node trace files alongside preparation."""

from contextlib import ExitStack

from shadowspill.orchestration.launch.publication import OutputFiles, OutputSpec
from shadowspill.server.deployment import ServerLimits
from shadowspill_client.buffers import BufferBudget


def test_full_node_traces_fit_beside_retained_preparation(tmp_path):
    limits = ServerLimits()
    disk = BufferBudget(limits.temporary_disk_bytes, limits.temporary_files)
    memory = BufferBudget(limits.creation.transfer_bytes, 128)
    with ExitStack() as owners:
        owners.enter_context(disk.reserve(limits.creation.preparation_disk_bytes))
        files = owners.enter_context(
            OutputFiles(root=tmp_path, disk=disk, memory=memory)
        )
        for rank in range(8):
            owners.enter_context(disk.reserve(2 * limits.creation.log_bytes + 12288))
            owners.enter_context(
                files.reserve(
                    OutputSpec(f"metrics.{rank}", 1 << 20, "application/octet-stream")
                )
            )
            # Real Qwen30B DP8/64K traces reserve 2,621,643,104 bytes per rank.
            # Round up to 3 GiB; only small permit files are actually written.
            owners.enter_context(
                files.reserve(
                    OutputSpec(
                        f"trace.{rank}", 3 << 30, "application/zip", exact_size=False
                    )
                )
            )
        assert sum(path.stat().st_size for path in files.root.iterdir()) < 64 << 10
    assert disk.used == memory.used == (0, 0)

"""Explicit zero-scratch assumptions for CPU simulation fixtures only."""

from shadowspill import CommunicationInfo
from shadowspill.communication.requests import collective_requests


def estimated_communication(
    program, *, topology, bandwidth=20e9, scale_up_bytes_per_second=100e9
):
    return CommunicationInfo(
        bandwidth,
        bandwidth,
        topology=topology,
        scale_up_bytes_per_second=scale_up_bytes_per_second,
        scale_up_latency_seconds=2e-6,
        fixed_device_bytes=0,
        metadata={"memory_kind": "estimated", "source": "analytic test assumptions"},
        measurements=tuple(
            {
                **request.to_dict(),
                "setup": program.setup,
                "topology": topology,
                "memory": {"workspace_bytes": 0},
            }
            for request in collective_requests(program, None)
        ),
    )

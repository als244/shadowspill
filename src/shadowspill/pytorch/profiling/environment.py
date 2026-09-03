"""Structural profiling environment identity."""

from __future__ import annotations

import torch

from shadowspill.pytorch.accelerator import provider_version

from .records import ProfileEnvironment


def profile_environment(
    *,
    device_ordinal: int,
    provider_id: str,
    implementation_revision: str | None = None,
) -> ProfileEnvironment:
    """Describe implementation attributes that can change measured task cost."""

    properties = torch.cuda.get_device_properties(device_ordinal)
    return ProfileEnvironment(
        torch_version=torch.__version__,
        provider_version=provider_version(),
        device_name=properties.name,
        compute_capability=(properties.major, properties.minor),
        compiler_id="shadowspill-explicit-task-compiler/v3:torch-inductor",
        provider_id=provider_id,
        implementation_revision=implementation_revision,
    )


__all__ = ["profile_environment"]

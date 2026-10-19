"""Structural profiling environment identity."""

from __future__ import annotations

import torch
from torch._inductor import config as inductor_config

from shadowspill.pytorch.accelerator import provider_version
from shadowspill.pytorch.compilation.inductor.compiler import task_compiler_patches
from shadowspill.pytorch.distributed import current as distributed_preparation
from shadowspill.pytorch.distributed._timing_identity import communication_timing_digest
from shadowspill.task.profiles import ProfileEnvironment

#: The provider a build's tasks are profiled under.
DEVICE_POOL_PROVIDER_ID = "shadowspill.device_pool"


def profile_environment(
    *,
    device_ordinal: int,
    provider_id: str,
    export_bypass_key: str | None = None,
) -> ProfileEnvironment:
    """Describe implementation attributes that can change measured task cost."""

    properties = torch.cuda.get_device_properties(device_ordinal)
    bound = distributed_preparation()
    peak_reordering = task_compiler_patches().get(
        "reorder_for_peak_memory", inductor_config.reorder_for_peak_memory
    )
    return ProfileEnvironment(
        torch_version=torch.__version__,
        provider_version=provider_version(),
        device_name=properties.name,
        compute_capability=(properties.major, properties.minor),
        # Functionalizing intermediate mutations changes Inductor's storage
        # reuse and allocation sizes even when the captured task is unchanged.
        # Its manifests and profiles cannot reuse the direct-mutation compiler's
        # cache entries. Artifact schemas remain unchanged.
        compiler_id=(
            "shadowspill-functional-task-compiler/v1:torch-inductor"
            f":emulate_precision_casts={int(inductor_config.emulate_precision_casts)}"
            f":reorder_for_peak_memory={int(peak_reordering)}"
        ),
        provider_id=provider_id,
        export_bypass_key=export_bypass_key,
        communication_timing_digest=(
            None
            if bound is None
            else communication_timing_digest(bound, device_ordinal=device_ordinal)
        ),
    )


__all__ = ["DEVICE_POOL_PROVIDER_ID", "profile_environment"]

"""glm53 catalog and training integration; no framework imports."""

from functools import partial

from shadowspill_workloads.models.registration import Family
from shadowspill_workloads.models.shared.glm.initialization.metadata import describe

FAMILY = Family(
    name="glm53",
    presets=(
        (
            "published",
            "GLM-5.3, native checkpoint storage and LoRA",
        ),
        ("tiny", "GLM-5.3's small random fixture"),
    ),
    implementations=("mlops",),
    mixture=True,
    training="shadowspill_workloads.models.glm53.training:register",
    pretrained_config=partial(describe, prefix="model."),
    checkpoint_types=("glm_moe_dsa",),
)

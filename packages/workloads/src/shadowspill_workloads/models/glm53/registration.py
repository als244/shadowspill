"""glm53 catalog and training integration; no framework imports."""

from functools import partial

from shadowspill_workloads.initialization.glm_metadata import describe
from shadowspill_workloads.models.registration import Family

FAMILY = Family(
    name="glm53",
    presets=(
        (
            "published",
            8192,
            "GLM-5.3, native checkpoint storage and LoRA",
        ),
        ("tiny", 4096, "GLM-5.3's small random fixture"),
    ),
    implementations=("mlops",),
    mixture=True,
    training="shadowspill_workloads.models.glm53.training:register",
    pretrained_config=partial(describe, prefix="model."),
    checkpoint_types=("glm_moe_dsa",),
)

"""glm53_flash catalog and training integration; no framework imports."""

from functools import partial

from shadowspill_workloads.models.registration import Family
from shadowspill_workloads.models.shared.glm.initialization.metadata import describe

FAMILY = Family(
    name="glm53_flash",
    presets=(
        (
            "published",
            "GLM-5.3-Flash, native checkpoint storage and LoRA",
        ),
        ("tiny", "GLM-5.3-Flash's small random fixture"),
    ),
    implementations=("mlops",),
    mixture=True,
    training="shadowspill_workloads.models.glm53_flash.training:register",
    pretrained_config=partial(describe, prefix="model.language_model."),
    checkpoint_types=("glm5_next", "glm5_next_text"),
)

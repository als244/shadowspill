"""glm53_flash catalog and training integration; no framework imports."""

from functools import partial

from shadowspill_workloads.initialization.glm_metadata import describe
from shadowspill_workloads.models.registration import Family

FAMILY = Family(
    name="glm53_flash",
    presets=(
        (
            "published",
            8192,
            "GLM-5.3-Flash, native checkpoint storage and LoRA",
        ),
        ("tiny", 4096, "GLM-5.3-Flash's small random fixture"),
    ),
    implementations=("mlops",),
    mixture=True,
    factory="shadowspill_workloads.recipes.text.glm.quickstart:glm53_flash",
    training="shadowspill_workloads.models.glm53_flash.training:register",
    pretrained_config=partial(describe, prefix="model.language_model."),
    checkpoint_types=("glm5_next", "glm5_next_text"),
)

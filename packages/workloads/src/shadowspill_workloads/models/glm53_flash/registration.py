"""glm53_flash catalog and training integration; no framework imports."""

from shadowspill_workloads.models.registration import Family

FAMILY = Family(
    name="glm53_flash",
    presets=(
        (
            "published",
            8192,
            "GLM-5.3-Flash, LoRA over a local checkpoint through its factory",
        ),
        ("tiny", 4096, "GLM-5.3-Flash's small random fixture"),
    ),
    implementations=("mlops",),
    mixture=True,
    factory="shadowspill_workloads.recipes.text.glm.quickstart:glm53_flash",
    checkpoint_types=("glm5_next", "glm5_next_text"),
)

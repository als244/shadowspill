"""glm53 catalog and training integration; no framework imports."""

from shadowspill_workloads.models.registration import Family

FAMILY = Family(
    name="glm53",
    presets=(
        (
            "published",
            8192,
            "GLM-5.3, LoRA over a local checkpoint through its factory",
        ),
        ("tiny", 4096, "GLM-5.3's small random fixture"),
    ),
    implementations=("mlops",),
    mixture=True,
    factory="shadowspill_workloads.recipes.text.glm.quickstart:glm53",
)

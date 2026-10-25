"""Supplied model shapes; add a row here and a preset in the family package."""

from .entry import IMPLEMENTATIONS, Entry, Geometry


def _family(
    family,
    shapes,
    *,
    implementations=IMPLEMENTATIONS,
    mixture=False,
    expert_parallel=False,
    factory=None,
):
    return tuple(
        Entry(
            name=family.replace("_", "-") + ("" if size == "published" else f"-{size}"),
            family=family,
            size=size,
            implementations=implementations,
            geometry=Geometry(1024, tokens, 65536),
            mixture=mixture,
            expert_parallel=expert_parallel,
            factory=factory,
            factory_arguments={"tiny": True} if factory and size == "tiny" else {},
            description=description,
        )
        for size, tokens, description in shapes
    )


ENTRIES: tuple[Entry, ...] = (
    *_family(
        "llama3",
        (
            ("numerical", 8_192, "the exact-scale shape the numerical gate runs"),
            ("tiny", 4_096, "a smoke-test shape"),
            ("1b", 8_192, "Llama-3.2-1B's decoder, with tied embeddings"),
            ("8b", 8_192, "Llama-3.1-8B's decoder"),
        ),
    ),
    *_family(
        "qwen3",
        (
            ("tiny", 4_096, "a smoke-test shape"),
            ("1.7b", 8_192, "Qwen3-1.7B, with tied embeddings"),
            ("8b", 8_192, "Qwen3-8B"),
            ("32b", 8_192, "Qwen3-32B"),
        ),
    ),
    *_family(
        "qwen35",
        (
            ("numerical", 16_384, "the exact-scale shape the numerical gate runs"),
            ("tiny", 4_096, "a smoke-test shape"),
            ("2b", 16_384, "Qwen3.5-2B's text decoder, with tied embeddings"),
            ("9b", 16_384, "Qwen3.5-9B's text decoder"),
            (
                "27b",
                8_192,
                "Qwen3.5-27B's text decoder, without its multi-token prediction layer",
            ),
        ),
    ),
    *_family(
        "olmoe",
        (
            ("numerical", 32_768, "the exact-scale shape the numerical gate runs"),
            ("tiny", 4_096, "a smoke-test shape"),
            ("7b-a1b", 32_768, "OLMoE-1B-7B"),
        ),
        mixture=True,
        expert_parallel=True,
    ),
    *_family(
        "qwen3moe",
        (
            ("tiny", 4_096, "a smoke-test shape"),
            ("30b-a3b", 8_192, "Qwen3-30B-A3B"),
        ),
        implementations=("mlops",),
        mixture=True,
        expert_parallel=True,
    ),
    *_family(
        "qwen35moe",
        (
            ("tiny", 4_096, "a smoke-test shape"),
            ("35b-a3b", 8_192, "Qwen3.5-35B-A3B's text decoder"),
        ),
        implementations=("mlops",),
        mixture=True,
        expert_parallel=True,
    ),
    *_family(
        "glm53",
        (
            (
                "published",
                8_192,
                "GLM-5.3, LoRA over a local checkpoint through its factory",
            ),
            ("tiny", 4_096, "GLM-5.3's small random fixture"),
        ),
        implementations=("mlops",),
        mixture=True,
        factory="workloads.models.glm53.quickstart:experiment",
    ),
    *_family(
        "glm53_flash",
        (
            (
                "published",
                8_192,
                "GLM-5.3-Flash, LoRA over a local checkpoint through its factory",
            ),
            ("tiny", 4_096, "GLM-5.3-Flash's small random fixture"),
        ),
        implementations=("mlops",),
        mixture=True,
        factory="workloads.models.glm53_flash.quickstart:experiment",
    ),
)

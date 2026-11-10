"""CPU-only checkpoint architecture identification for supplied recipes."""

CHECKPOINT_FAMILIES = {
    "llama": "llama3",
    "qwen3": "qwen3",
    "qwen3_moe": "qwen3moe",
    "qwen3_5": "qwen35",
    "qwen3_5_text": "qwen35",
    "qwen3_5_moe": "qwen35moe",
    "qwen3_5_moe_text": "qwen35moe",
    "olmoe": "olmoe",
    "glm5_next": "glm53_flash",
    "glm5_next_text": "glm53_flash",
}


def checkpoint_family(metadata):
    text = metadata.get("text_config", metadata)
    kind = text.get("model_type", metadata.get("model_type"))
    try:
        return CHECKPOINT_FAMILIES[kind]
    except KeyError as error:
        raise ValueError(f"unsupported checkpoint model_type: {kind!r}") from error

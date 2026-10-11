"""CPU preparation for still-image prompts using the checkpoint's HF processor."""

import json
from pathlib import Path

from .vision import prepare_images


def prepare_image_prompt(checkpoint, image_path, text, *, assistant_prefix=""):
    """Return rendered text, token IDs and image tensors before model capture.

    PIL preprocessing avoids requiring a video backend for still images. All
    resizing, normalization and patch packing use the checkpoint's processor
    settings. A single image is supported by this demonstration CLI.
    """
    from PIL import Image
    from transformers import AutoTokenizer
    from transformers.models.glm5_next.image_processing_pil_glm5_next import (
        Glm5NextImageProcessorPil,
    )

    directory = Path(checkpoint)
    metadata = json.loads((directory / "config.json").read_text())
    settings = json.loads((directory / "processor_config.json").read_text())[
        "image_processor"
    ]
    processor = Glm5NextImageProcessorPil(**settings)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    with Image.open(image_path) as image:
        processed = processor(images=[image.convert("RGB")], return_tensors="pt")
    image_id = metadata["image_token_id"]
    image_token = tokenizer.convert_ids_to_tokens(image_id)
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": text}],
        }
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if rendered.count(image_token) != 1:
        raise ValueError("The single-image chat template must contain one image marker")
    count = int(processed["image_grid_thw"][0].prod()) // settings["merge_size"] ** 2
    rendered = rendered.replace(image_token, image_token * count) + assistant_prefix
    tokens = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].flatten()
    images = prepare_images(
        processed["pixel_values"],
        processed["image_grid_thw"],
        tokens,
        image_token_id=image_id,
        merge_size=settings["merge_size"],
    )
    return rendered, tokens.tolist(), images

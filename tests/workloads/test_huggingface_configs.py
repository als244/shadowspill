"""Each family that names a Hugging Face configuration for its model
(``to_huggingface``, what the recipe harnesses build NVIDIA's model from)
names the same model its reader would read back, and the published model's
configuration where its checkpoint is on this machine."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pytest

from workloads.models import catalog

FAMILIES = sorted(
    {
        item.family
        for item in catalog.entries()
        if hasattr(
            import_module(f"workloads.models.{item.family}.config"), "to_huggingface"
        )
    }
)

#: Published presets and the local checkpoints of the same model.
CHECKPOINTS = {
    ("llama3", "1b"): "/mnt/storage/models/Llama-3.2-1B",
    ("qwen3", "1.7b"): "/mnt/storage/models/Qwen3-1.7B",
    ("qwen3moe", "30b-a3b"): "/mnt/storage/models/Qwen3-30B-A3B",
    ("olmoe", "7b-a1b"): "/mnt/storage/models/OLMoE-7B-A1B",
    ("qwen35", "9b"): "/mnt/storage/models/Qwen3.5-9B",
    ("qwen35moe", "35b-a3b"): "/mnt/storage/models/Qwen3.5-35B-A3B",
}

#: What the families write on purpose that a published configuration does
#: not: the rotary capacity is the catalog's, the multi-token prediction
#: layer is left out, and a vision-language checkpoint's text settings name
#: neither the text architecture nor the keys kept beside them.
DELIBERATE = {
    "max_position_embeddings",
    "mtp_num_hidden_layers",
    "architectures",
    "torch_dtype",
    "tie_word_embeddings",
    "norm_topk_prob",
}


def test_the_round_one_families_name_their_configuration() -> None:
    assert {"llama3", "qwen3", "qwen3moe", "olmoe", "qwen35", "qwen35moe"} <= set(
        FAMILIES
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_every_preset_reads_back_or_is_refused_with_the_reason(family) -> None:
    module = import_module(f"workloads.models.{family}.config")
    for name, config in module.PRESETS.items():
        try:
            value = json.loads(json.dumps(module.to_huggingface(config)))
        except ValueError as error:
            assert str(error), (family, name)
            continue
        back = module.from_huggingface(value, max_seq_len=config.max_seq_len)
        assert back == config, (family, name)


@pytest.mark.parametrize(("family", "size"), sorted(CHECKPOINTS))
def test_a_published_preset_is_the_published_configuration(family, size) -> None:
    path = Path(CHECKPOINTS[(family, size)]) / "config.json"
    if not path.is_file():
        pytest.skip(f"no {path}")
    published = json.loads(path.read_text())
    published = published.get("text_config", published)
    module = import_module(f"workloads.models.{family}.config")
    ours = module.to_huggingface(module.PRESETS[size])
    differences = {
        key: (published.get(key), value)
        for key, value in ours.items()
        if key not in DELIBERATE and published.get(key) != value
    }
    assert not differences, differences

"""qwen3 catalog and training integration; no framework imports."""

from shadowspill_workloads.definitions.text_metadata import validate_activation
from shadowspill_workloads.models.registration import Family, ParameterInitialization

FAMILY = Family(
    name="qwen3",
    presets=(
        ("tiny", "a smoke-test shape"),
        ("1.7b", "Qwen3-1.7B, with tied embeddings"),
        ("8b", "Qwen3-8B"),
        ("32b", "Qwen3-32B"),
    ),
    checkpoint_reader="Qwen3Checkpoint",
    checkpoint_types=("qwen3",),
    initialization=ParameterInitialization(),
    training="shadowspill_workloads.models.qwen3.training:register",
    validate_pretrained=validate_activation,
)

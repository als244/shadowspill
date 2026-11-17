"""llama3 catalog and training integration; no framework imports."""

from shadowspill_workloads.definitions.text_metadata import validate_activation
from shadowspill_workloads.models.registration import Family, ParameterInitialization

FAMILY = Family(
    name="llama3",
    presets=(
        ("numerical", "the exact-scale shape the numerical gate runs"),
        ("tiny", "a smoke-test shape"),
        ("1b", "Llama-3.2-1B's decoder, with tied embeddings"),
        ("8b", "Llama-3.1-8B's decoder"),
    ),
    checkpoint_reader="Llama3Checkpoint",
    checkpoint_types=("llama",),
    initialization=ParameterInitialization(),
    training="shadowspill_workloads.models.llama3.training:register",
    validate_pretrained=validate_activation,
)

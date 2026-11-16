"""qwen35 catalog and training integration; no framework imports."""

from functools import partial

from shadowspill_workloads.definitions.text_metadata import validate_decoder
from shadowspill_workloads.models.registration import Family, ParameterInitialization

FAMILY = Family(
    name="qwen35",
    presets=(
        ("numerical", 16384, "the exact-scale shape the numerical gate runs"),
        ("tiny", 4096, "a smoke-test shape"),
        ("2b", 16384, "Qwen3.5-2B's text decoder, with tied embeddings"),
        ("9b", 16384, "Qwen3.5-9B's text decoder"),
        (
            "27b",
            8192,
            "Qwen3.5-27B's text decoder, without its multi-token prediction layer",
        ),
    ),
    checkpoint_reader="Qwen35Checkpoint",
    checkpoint_types=("qwen3_5", "qwen3_5_text"),
    initialization=ParameterInitialization(),
    parameter_dtypes={
        "blocks.*.mixer.A_log": "float32",
        "blocks.*.mixer.dt_bias": "float32",
    },
    training="shadowspill_workloads.models.qwen35.training:register",
    validate_pretrained=partial(validate_decoder, attention_gate=True),
)

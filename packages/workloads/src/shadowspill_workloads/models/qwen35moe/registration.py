"""qwen35moe catalog and training integration; no framework imports."""

from functools import partial

from shadowspill_workloads.definitions.text_metadata import validate_decoder
from shadowspill_workloads.models.registration import Family, ParameterInitialization

FAMILY = Family(
    name="qwen35moe",
    presets=(
        ("tiny", "a smoke-test shape"),
        ("35b-a3b", "Qwen3.5-35B-A3B's text decoder"),
    ),
    implementations=("mlops",),
    mixture=True,
    expert_parallel=True,
    parameter_dtypes={
        "blocks.*.mixer.A_log": "float32",
        "blocks.*.mixer.dt_bias": "float32",
    },
    checkpoint_reader="Qwen35MoECheckpoint",
    checkpoint_types=("qwen3_5_moe", "qwen3_5_moe_text"),
    initialization=ParameterInitialization(
        log_uniform_decay=True,
        time_bias=1,
        normal_leaves=("w13", "w2", "weight", "w13_experts", "w2_experts"),
    ),
    training="shadowspill_workloads.models.qwen35moe.training:register",
    validate_pretrained=partial(validate_decoder, mixture=True),
)

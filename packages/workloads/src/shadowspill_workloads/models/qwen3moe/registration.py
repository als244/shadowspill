"""qwen3moe catalog and training integration; no framework imports."""

from functools import partial

from shadowspill_workloads.definitions.text_metadata import validate_decoder
from shadowspill_workloads.models.registration import Family, ParameterInitialization

FAMILY = Family(
    name="qwen3moe",
    presets=(("tiny", 4096, "a smoke-test shape"), ("30b-a3b", 8192, "Qwen3-30B-A3B")),
    implementations=("mlops",),
    mixture=True,
    expert_parallel=True,
    checkpoint_reader="Qwen3MoECheckpoint",
    checkpoint_types=("qwen3_moe",),
    initialization=ParameterInitialization(
        log_uniform_decay=True,
        time_bias=1,
        normal_leaves=("w13", "w2", "weight", "w13_experts", "w2_experts"),
    ),
    training="shadowspill_workloads.models.qwen3moe.training:register",
    validate_pretrained=partial(validate_decoder, mixture=True),
)

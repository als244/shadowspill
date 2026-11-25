"""CPU-only catalog and pretrained identification."""

from shadowspill_workloads.models.registration import Family, ParameterInitialization

from .config import describe

FAMILY = Family(
    name="inkling2_mini",
    presets=(
        ("published", "Inkling 2 Mini fixed-loop text decoder"),
        ("tiny", "Three-pass text fixture"),
    ),
    implementations=("mlops",),
    mixture=True,
    checkpoint_reader="InklingCheckpoint",
    quantization_file="hf_quant_config.json",
    checkpoint_types=("inkling2_loop", "inkling2_loop_text"),
    pretrained_config=describe,
    initialization=ParameterInitialization(
        normal_leaves=(
            "w13",
            "w2",
            "proj",
            "weight_proj",
            "down_weight",
            "gain_weight",
            "forget_weight",
        ),
        zero_leaves=(
            "bias",
            "static_w",
            "attn_query",
            "mlp_query",
            "output_query",
            "middle_cycle_cross_pass_queries",
            "middle_cycle_cross_pass_kv_queries",
        ),
        one_leaves=("q_scale", "rel_scale"),
    ),
    training="shadowspill_workloads.models.inkling2_mini.training:register",
)

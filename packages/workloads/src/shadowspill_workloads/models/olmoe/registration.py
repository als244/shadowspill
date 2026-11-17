"""olmoe catalog and training integration; no framework imports."""

from functools import partial

from shadowspill_workloads.definitions.text_metadata import validate_decoder
from shadowspill_workloads.models.registration import Family, ParameterInitialization

FAMILY = Family(
    name="olmoe",
    presets=(
        ("numerical", "the exact-scale shape the numerical gate runs"),
        ("tiny", "a smoke-test shape"),
        ("7b-a1b", "OLMoE-1B-7B"),
    ),
    mixture=True,
    expert_parallel=True,
    checkpoint_reader="OLMoECheckpoint",
    checkpoint_types=("olmoe",),
    initialization=ParameterInitialization(),
    training="shadowspill_workloads.models.olmoe.training:register",
    validate_pretrained=partial(validate_decoder, mixture=True, unclipped=True),
)

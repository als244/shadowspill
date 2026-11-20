"""Family defaults and public overrides reach full/LoRA model construction."""

from dataclasses import replace

import pytest
import torch

from shadowspill.training.components import load_registry
from shadowspill.training.preparation.setup import resolve_training
from shadowspill_workloads.models import catalog
from shadowspill_workloads.models.families import family
from workload_tests.components.test_dense_qwen_components import prepared
from workload_tests.components.test_dense_qwen_components import (
    settings as dense_settings,
)
from workload_tests.components.test_moe_components import settings as moe_settings


@pytest.mark.parametrize("name", ["qwen35", "qwen35moe"])
@pytest.mark.parametrize("lora", [False, True])
def test_public_precision_overrides_preserve_other_defaults_and_trainability(
    name, lora
):
    registry = load_registry(["shadowspill-workloads"])
    settings = (dense_settings if name == "qwen35" else moe_settings)(
        name, dtype="bfloat16", lora=lora
    )
    settings = replace(
        settings,
        precision={**settings.precision, "parameter_dtypes": {"*.A_log": "bfloat16"}},
    )
    resolved = registry.resolve_model(settings.name, settings=settings)
    assert resolved.component.parameter_dtypes == family(name).parameter_dtypes
    with torch.device("meta"):
        model = resolved.create()
    parameters = dict(model.named_parameters())
    assert parameters["blocks.0.mixer.A_log"].dtype == torch.bfloat16
    assert parameters["blocks.0.mixer.dt_bias"].dtype == torch.float32
    assert parameters["blocks.0.mixer.A_log"].requires_grad == (not lora)
    assert any(p.requires_grad for p in parameters.values())
    if lora:
        assert all("lora_" in n for n, p in parameters.items() if p.requires_grad)
    assert resolved.settings.precision == settings.precision


@pytest.mark.parametrize("name", ["qwen35", "qwen35moe"])
def test_catalog_uses_same_family_policy_and_optional_layers_can_be_absent(name):
    for interval in (1, 2):
        model = catalog.build_on_meta(
            f"{name}-tiny",
            dtype="bfloat16",
            n_layers=2,
            full_attention_interval=interval,
        )
        decay = [
            p for n, p in model.named_parameters() if n.endswith(("A_log", "dt_bias"))
        ]
        assert len(decay) == (0 if interval == 1 else 2)
        assert all(p.dtype == torch.float32 for p in decay)


def test_precision_rules_survive_saved_settings_and_change_setup_identity():
    registry, setup, _ = prepared("qwen35")
    changed = replace(
        setup.config,
        model=replace(
            setup.config.model,
            precision={
                **setup.config.model.precision,
                "parameter_dtypes": {"*.A_log": "bfloat16"},
            },
        ),
    )
    restored = resolve_training(registry, changed)
    assert restored.config == changed and restored.config != setup.config
    with torch.device("meta"):
        assert restored.model.create().blocks[0].mixer.A_log.dtype == torch.bfloat16


def test_misspelled_selector_fails_on_meta_before_capture():
    registry = load_registry(["shadowspill-workloads"])
    settings = dense_settings("qwen35")
    settings = replace(
        settings, precision={"parameter_dtypes": {"*.mixer.Alog": "float32"}}
    )
    resolved = registry.resolve_model("qwen35", settings=settings)
    with (
        torch.device("meta"),
        pytest.raises(ValueError, match="matches no registered tensor"),
    ):
        resolved.create()

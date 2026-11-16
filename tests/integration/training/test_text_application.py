"""Shared text preparation accepts model and objective declarations from callers."""

from types import SimpleNamespace

import pytest

from examples.training.text.preparation import resolve
from examples.training.text.settings import TextSettings
from shadowspill.parallelism import Parallelism
from shadowspill.training.definitions import load_registry
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.setup import resolve_training
from shadowspill_client import Model, Objective
from shadowspill_client.parallelism import data_parallel, expert_parallel


@pytest.mark.parametrize("family", ["llama3", "qwen3moe", "qwen35moe", "olmoe"])
def test_common_settings_prepare_multiple_models_with_the_same_scheduled_cap(family):
    settings = TextSettings(
        data="/client/data",
        sequence_length=4,
        sequences_per_step=4,
        microbatch_sizes=(1, 2),
        steps=2,
        resolution_options=("0", "1/2", "1"),
        tail_save_counts=(0, 2),
    )
    model = Model(
        f"shadowspill/{family}@1",
        config={"preset": "tiny", "architecture": {"max_seq_len": 4}},
    )
    tokens = Objective("shadowspill/cross_entropy@1")
    objective = (
        tokens
        if family == "llama3"
        else Objective.sum(
            {"primary": tokens, "routing": Objective("shadowspill/moe_balance@1")},
            weights={"routing": 0.25},
        )
    )
    request = settings.request(model=model, objective=objective, parallelism=None)
    checkpoint_coefficient = family in {"qwen3moe", "qwen35moe"}
    registry = load_registry(["shadowspill", "shadowspill-workloads"])
    calls = []

    def describe(**options):
        calls.append(options)
        value = resolve_training(registry, **options).preview()
        return SimpleNamespace(result=lambda: value)

    preview = resolve(
        SimpleNamespace(describe_training=describe),
        request,
        definition=model.definition,
        defaults_from_model={"routing.weight": ("architecture", "router_aux_loss_coef")}
        if checkpoint_coefficient
        else None,
        resolution_options=settings.resolution_options,
        tail_save_counts=settings.tail_save_counts,
    )
    assert len(calls) == 2
    assert request["model"] is model and request["objective"] is objective
    assert calls[1]["model"].definition == model.definition
    if family != "llama3":
        assert objective.defaults["routing.weight"] == 0.25
        expected = (
            preview.config.model.config["architecture"]["router_aux_loss_coef"]
            if checkpoint_coefficient
            else 0.25
        )
        assert preview.config.objective.defaults["routing.weight"] == expected
        balance = preview.config.objective.config["terms"]["routing"]
        assert balance["config"]["include_padding"] is False
    search = preview.config.preparation.planning.to_dict()["search"]
    assert search["algorithm"]["options"]["resolution_options"] == ["0", "1/2", "1"]
    assert search["algorithm"]["options"]["tail_save_counts"] == [0, 2]
    layouts = prepare_layouts(resolve_training(registry, preview.config))
    for rows in layouts.families.values():
        for row in rows:
            assert (
                sum(
                    buffer.fields["inputs", "tokens"][0]
                    for buffer in row.candidate.schedule.buffers
                )
                <= 16
            )


@pytest.mark.parametrize("ranks", [1, 2, 4])
def test_parallelism_accepts_custom_fields_outputs_and_expert_parameter_names(ranks):
    fields = (("inputs", "features"), ("loss_inputs", "labels"))
    outputs = {"prediction": {"mode": "leader", "axis": None}}
    wire = data_parallel(ranks, fields=fields, outputs=outputs)
    parsed = Parallelism.from_dict(wire)
    assert parsed == Parallelism.dp(ranks_per_replica=ranks, inputs=parsed.inputs)
    experts = ("layers.*.bank", "layers.*.projection")
    wire = expert_parallel(ranks, fields=fields, experts=experts, outputs=outputs)
    parsed = Parallelism.from_dict(wire)
    assert set(parsed.parameter_placements) == set(experts)
    assert wire["inputs"]["outputs"] == outputs
    assert [entry["path"] for entry in wire["inputs"]["fields"]] == [
        list(path) for path in fields
    ]

"""Metadata fills missing settings without changing explicit nested overrides."""

from dataclasses import replace

import pytest

from shadowspill.training.definitions import ModelReferenceDefinition
from shadowspill_client import Model, Weights


def resolver(field, values):
    return ModelReferenceDefinition(
        "external/metadata@1",
        lambda reference: reference.startswith("fixture://"),
        lambda requested: replace(
            requested,
            definition="external/classifier@1",
            initialize=Weights.pretrained("fixture://classifier", revision="b" * 40),
            **{field: values},
        ),
    )


@pytest.mark.parametrize("field", ["config", "precision"])
def test_metadata_can_fill_nested_defaults_and_recheck_the_resolved_pin(field):
    explicit = {
        "architecture": {"length": 32, "layout": {"tied": True}},
        "stages": ["input", "output"],
        "optional": None,
    }
    complete = {
        "architecture": {
            "length": 32,
            "width": 16,
            "layout": {"tied": True, "axis": 0},
        },
        "stages": ["input", "output"],
        "optional": None,
        "default": 17,
    }
    requested = Model.from_pretrained("fixture://classifier", **{field: explicit})
    before = requested.to_dict()
    metadata = resolver(field, complete)
    resolved = metadata.pin(requested)
    assert resolved.to_dict()[field] == complete
    assert metadata.pin(Model.from_dict(resolved.to_dict())) == resolved
    assert requested.to_dict() == before


@pytest.mark.parametrize("field", ["config", "precision"])
@pytest.mark.parametrize(
    "explicit, resolved",
    [
        ({"nested": {"length": 32}}, {"nested": {"length": 64}}),
        ({"nested": {"length": 32}}, {"nested": {"width": 16}}),
        ({"nested": {"length": 32}}, {"nested": 32}),
        ({"nested": None}, {"nested": {"default": 1}}),
        ({"nested": [1, 2]}, {"nested": [1, 2, 3]}),
        ({"nested": [1, 2]}, {"nested": [2, 1]}),
    ],
)
def test_metadata_cannot_replace_remove_or_extend_explicit_values(
    field, explicit, resolved
):
    requested = Model.from_pretrained("fixture://classifier", **{field: explicit})
    with pytest.raises(ValueError, match="changed explicit model settings"):
        resolver(field, resolved).pin(requested)

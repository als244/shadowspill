"""Explicit recipe geometry is validated before any expensive phase starts."""

import json

import pytest

from shadowspill_workloads.recipes.text.phases.candidates import batch_sizes


def test_explicit_sizes_keep_requested_order_without_building_duplicates():
    assert batch_sizes(dict(sequences_per_step=12, microbatch_sizes=[4, 3, 4])) == (
        4,
        3,
    )


@pytest.mark.parametrize("sizes", [[], 2, "2", [True], [2.0], [0], [-1], [3], [16]])
def test_invalid_sizes_are_refused_instead_of_silently_filtered(sizes):
    with pytest.raises(ValueError, match="microbatch"):
        batch_sizes(dict(sequences_per_step=8, microbatch_sizes=sizes))


@pytest.mark.parametrize("count", [0, 2.5, True])
def test_step_size_must_be_a_positive_integer(count):
    with pytest.raises(ValueError, match="sequences_per_step"):
        batch_sizes(dict(sequences_per_step=count, microbatch_sizes=[1]))


def test_saved_recipe_preserves_exact_candidates_and_rejects_shape_changes(tmp_path):
    recipe = "shadowspill_workloads.recipes.text.catalog"
    from shadowspill.training.preparation.configuration import configure

    root = tmp_path / "run"
    build = {
        "data": str(tmp_path / "data"),
        "run_dir": str(root),
        "model": "qwen3moe-tiny",
        "sequences_per_step": 8,
        "microbatch_sizes": [4, 2, 4],
    }
    configure(recipe, build, phase="build")
    path = root / "config.json"
    saved = json.loads(path.read_text())
    assert saved["microbatch_sizes"] == [4, 2]
    assert "max_sequences_per_example" not in saved
    plan = {"run_dir": str(root), "execution_gib": 4}
    configure(recipe, plan, phase="plan")
    assert json.loads(path.read_text()) == {**saved, "execution_gib": 4}
    before = path.read_bytes()
    with pytest.raises(ValueError, match="changed microbatch_sizes"):
        configure(recipe, {**plan, "microbatch_sizes": [1, 2, 4]}, phase="plan")
    assert path.read_bytes() == before

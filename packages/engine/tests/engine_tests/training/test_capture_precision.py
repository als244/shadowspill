"""Definition precision reaches the existing real gradient/optimizer captures."""

from dataclasses import asdict, replace

import pytest
import torch

from engine_tests.fixtures.definition_operations import definitions
from engine_tests.training.test_setup_resolution import arguments, installed
from shadowspill.build.optimizer import capture_update
from shadowspill.build.training_capture import capture_training_model
from shadowspill.training.definitions import ModelBuild, adamw
from shadowspill.training.preparation.capture import build_template
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.preparation.templates import capture_inventory
from shadowspill_client import BuildCandidates, Optimizer, PlanningOptions, Preparation


def create(config):
    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(4, config.num_classes, dtype=torch.bfloat16)

        def forward(self, pixels):
            return self.head(pixels.flatten(start_dim=1).bfloat16()).float()

    return Classifier()


def prepared(master, grad="float32"):
    model = replace(
        definitions()[0].definition,
        create=create,
        build=lambda config, context: ModelBuild(asdict(config), grad_dtype=grad),
    )
    registry = installed(model=model)
    adamw.register(registry)
    setup = resolve_training(
        registry,
        **{
            **arguments(),
            "optimizer": Optimizer(
                "mlops/adamw@1",
                precision={
                    "master_dtype": master,
                    "gradient_dtype": "float32",
                    "opt_state_dtype": "float32",
                },
            ),
        },
        operations={"forward_backward": ["metrics"], "forward": ["predictions"]},
        prepare=Preparation(
            candidates=BuildCandidates(microbatch_sizes=[4]),
            planning=PlanningOptions(microbatch_counts=[1]),
        ),
    )
    return registry, setup, capture_inventory(prepare_layouts(setup))


@pytest.mark.parametrize("master", [None, "float32"])
def test_bfloat16_model_fp32_gradients_and_optional_masters_reach_actual_capture(
    monkeypatch, master
):
    registry, setup, inventory = prepared(master)
    template = next(t for t in inventory.templates.values() if t.kind == "step")
    assert setup.optimizer.master_dtype == master
    assert template.signature()["model_build"]["grad_dtype"] == "float32"
    assert template.signature()["optimizer_master_dtype"] == master
    assert resolve_training(registry, setup.config).config == setup.config

    class Captured(Exception):
        pass

    def capture(model, *, objective, example_args, optimizer, **options):
        # Exercise real core captures, stopping before CUDA compilation/profiling.
        assert options["grad_dtype"] == torch.float32
        assert options.get("master_dtype") == (
            None if master is None else torch.float32
        )
        training = capture_training_model(
            model,
            objective,
            example_args,
            {},
            num_examples=1,
            example_calls=None,
            fuse_narrow_gradient_accumulation=False,
            partition="whole",
            graph_pair_policy=None,
            gradient_dtype=options["grad_dtype"],
            device_index=0,
        )
        update = capture_update(
            model,
            optimizer,
            training,
            gradient_dtype=options["grad_dtype"],
            master_dtype=options.get("master_dtype"),
            runtime_gradient_scale=options["runtime_gradient_scale"],
        )
        bindings = update.captured.bindings
        parameters = [b for b in bindings if b.role == "parameter"]
        copies = [b for b in bindings if b.role == "compute_copy"]
        gradients = [b for b in bindings if b.role == "gradient"]
        assert parameters and gradients
        assert all(
            b.tensor.dtype == (torch.bfloat16 if master is None else torch.float32)
            for b in parameters
        )
        assert bool(copies) == (master is not None)
        assert all(b.tensor.dtype == torch.bfloat16 for b in copies)
        assert all(b.tensor.dtype == torch.float32 for b in gradients)
        assert update.gradient_scale_control is not None
        setup.optimizer.definition.validate_capture(update.captured)
        raise Captured

    monkeypatch.setattr("shadowspill.build_step", capture)
    with pytest.raises(Captured):
        build_template(registry, inventory, template.id)


def test_precision_changes_capture_identity_and_invalid_choices_fail_on_cpu():
    _, _, default = prepared(None, None)
    _, _, gradients = prepared(None)
    _, _, masters = prepared("float32")
    assert default.templates.keys().isdisjoint(gradients.templates)
    step = {t.id for t in gradients.templates.values() if t.kind == "step"}
    assert step.isdisjoint(masters.templates)
    for dtype in ("int32", "float8", 1, True):
        with pytest.raises(ValueError, match="floating dtype"):
            ModelBuild({}, grad_dtype=dtype)
        with pytest.raises(ValueError, match="floating dtype"):
            replace(definitions()[2].definition, master_dtype=dtype)

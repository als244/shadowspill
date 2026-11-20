"""Component precision reaches the existing real gradient/optimizer captures."""

from contextlib import contextmanager
from dataclasses import asdict, replace

import pytest
import torch

from engine_tests.fixtures.component_operations import components
from engine_tests.training.test_setup_resolution import arguments, installed
from shadowspill.build.optimizer import capture_update
from shadowspill.build.training_capture import capture_training_model
from shadowspill.training.components import ModelBuild
from shadowspill.training.preparation.capture.build import build_template
from shadowspill.training.preparation.capture.templates import capture_inventory
from shadowspill.training.preparation.configuration.layouts import prepare_layouts
from shadowspill.training.preparation.setup import resolve_training
from shadowspill_client import BuildCandidates, Optimizer, PlanningOptions, Preparation
from shadowspill_workloads.components import adamw


def create(config):
    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(4, config.num_classes, dtype=torch.bfloat16)

        def forward(self, pixels):
            return self.head(pixels.flatten(start_dim=1).bfloat16()).float()

    return Classifier()


def prepared(
    master,
    grad="float32",
    capture_context=None,
    gradient_destination_rules=None,
    optimizer_grad="float32",
):
    model = replace(
        components()[0].component,
        create=create,
        build=lambda config, context: ModelBuild(asdict(config), grad_dtype=grad),
        capture_context=capture_context,
        gradient_destination_rules=gradient_destination_rules,
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
                    "gradient_dtype": optimizer_grad,
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


@pytest.mark.cuda
@pytest.mark.parametrize("master", [None, "bfloat16", "float32"])
@pytest.mark.parametrize("gradient", ["float32", "bfloat16"])
def test_bfloat16_model_fp32_gradients_and_optional_masters_reach_actual_capture(
    monkeypatch, master, gradient
):
    if not torch.cuda.is_available():
        pytest.skip("fake CUDA autograd capture requires a representative GPU")
    registry, setup, inventory = prepared(master, optimizer_grad=gradient)
    template = next(t for t in inventory.templates.values() if t.kind == "step")
    assert setup.optimizer.master_dtype == master
    assert template.signature()["model_build"]["grad_dtype"] == gradient
    assert template.signature()["optimizer_master_dtype"] == master
    assert resolve_training(registry, setup.config).config == setup.config

    class Captured(Exception):
        pass

    def capture(model, *, objective, example_args, optimizer, **options):
        # Exercise real core captures, stopping before CUDA compilation/profiling.
        assert options["grad_dtype"] == getattr(torch, gradient)
        assert options.get("master_dtype") == (
            None if master is None else getattr(torch, master)
        )
        training = capture_training_model(
            model,
            objective,
            example_args,
            {},
            num_examples=1,
            example_calls=None,
            fuse_narrow_gradient_accumulation=True,
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
            b.tensor.dtype == (torch.float32 if master == "float32" else torch.bfloat16)
            for b in parameters
        )
        assert bool(copies) == (master == "float32")
        assert all(b.tensor.dtype == torch.bfloat16 for b in copies)
        assert all(b.tensor.dtype == getattr(torch, gradient) for b in gradients)
        assert update.gradient_scale_control is not None
        setup.optimizer.component.validate_capture(update.captured)
        raise Captured

    monkeypatch.setattr("shadowspill.build_step", capture)
    with pytest.raises(Captured):
        build_template(registry, inventory, template.id)


def test_precision_changes_capture_identity_and_invalid_choices_fail_on_cpu():
    _, _, default = prepared(None, None, optimizer_grad="bfloat16")
    _, _, gradients = prepared(None)
    _, _, masters = prepared("float32")
    assert default.templates.keys().isdisjoint(gradients.templates)
    step = {t.id for t in gradients.templates.values() if t.kind == "step"}
    assert step.isdisjoint(masters.templates)
    for dtype in ("int32", "float8", 1, True):
        with pytest.raises(ValueError, match="floating dtype"):
            ModelBuild({}, grad_dtype=dtype)
        with pytest.raises(ValueError, match="floating dtype"):
            replace(components()[2].component, master_dtype=dtype)


@pytest.mark.parametrize("fail_in", ["create", "capture"])
def test_provider_context_is_worker_only_and_restored_after_failure(
    monkeypatch, fail_in
):
    from mlops.dispatch import weight_gradient_dtype, weight_gradients_at

    opened = []

    @contextmanager
    def context(config):
        opened.append(config.num_classes)
        with weight_gradients_at(torch.float32):
            yield

    registry, setup, inventory = prepared("float32", capture_context=context)
    assert opened == []  # Description and inventory never enter the provider.
    template = next(t for t in inventory.templates.values() if t.kind == "step")
    factory = setup.model.component.create

    def checked_create(config):
        assert weight_gradient_dtype() == torch.float32
        if fail_in == "create":
            raise RuntimeError("intentional constructor failure")
        return factory(config)

    def checked_capture(*args, **kwargs):
        assert weight_gradient_dtype() == torch.float32
        raise RuntimeError("intentional capture failure")

    monkeypatch.setattr(
        type(setup.model),
        "create",
        lambda self, **kw: checked_create(
            self.component.config(**dict(kw["build"].config))
        ),
    )
    monkeypatch.setattr("shadowspill.build_step", checked_capture)
    with weight_gradients_at(torch.float64):
        with pytest.raises(RuntimeError, match="intentional"):
            build_template(registry, inventory, template.id)
        assert weight_gradient_dtype() == torch.float64
    assert opened == [3]

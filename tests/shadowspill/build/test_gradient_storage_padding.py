"""Optimizer communication padding belongs to its own buffer, not the producer."""

from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill import Parallelism
from shadowspill.build.gradient_ops import GradientOperations, _padding
from shadowspill.build.lowering.objects import ObjectCatalog
from shadowspill.build.lowering.training.gradients import gradient_templates
from shadowspill.build.lowering.training.records import GradientBinding, TrainingObjects
from shadowspill.build.parallelism import model_parallelism
from shadowspill.ir import ObjectRole, Persistence
from shadowspill.parallelism.reductions import gradient_reduction
from shadowspill.pytorch.operators.materialize import PAD_STORAGE


@pytest.mark.parametrize(
    "gradients,optimizer",
    [("replicated", "replicated"), ("replicated", "sharded"), ("sharded", "sharded")],
)
def test_replicated_gradient_keeps_its_extent_before_optimizer_padding(
    gradients, optimizer
):
    with FakeTensorMode():
        model = torch.nn.Linear(11, 7, bias=False, device="cuda")
        _, (ownership,) = model_parallelism(
            model,
            Parallelism.dp(
                ranks_per_replica=2,
                weights="replicated",
                gradients=gradients,
                optimizer=optimizer,
            ),
        )
        reduce = gradient_reduction(ownership, destination="optimizer")
        spill = gradient_reduction(ownership, destination="gradients")
        gradient = torch.empty(7, 11, dtype=torch.float32, device="cuda")
        if gradients == "sharded":
            gradient = PAD_STORAGE(gradient, 78)
        catalog = ObjectCatalog(device_id="device0")
        weight_id = catalog.add(
            model.weight, role=ObjectRole.PARAMETER, persistence=Persistence.RUN
        )
        source = catalog.add(
            gradient, role=ObjectRole.GRADIENT, persistence=Persistence.STEP
        )
        physical_bytes = gradient.untyped_storage().nbytes()
        objects = TrainingObjects(
            catalog,
            (),
            (),
            {"weight": weight_id},
            (GradientBinding("weight", weight_id, source),),
            {weight_id: source},
            (),
        )
        padding = _padding(
            gradient,
            reduce.padded_elements,
            copy=reduce.phases[0].operation == "all_reduce",
        )
        operation = GradientOperations(
            reduce, spill, torch.empty(reduce.result_elements, device="cuda"), padding
        )
        # This unit checks lowering's ownership contract, independent of compiling
        # the padding task. The distributed saved-Plan test executes every phase.
        compiled = SimpleNamespace(
            definitions={}, inputs={}, key=lambda *_args: "padding"
        )
        (template,), _ = gradient_templates(
            {"weight": operation}, objects, compiled, for_optimizer=True
        )
    assert catalog.alias_size(catalog.alias_id(source)) == physical_bytes
    if template["padding"] is not None:
        padded = template["padding"]["output"]
        assert padded != source
        assert (
            catalog.alias_size(catalog.alias_id(padded)) == reduce.padded_elements * 4
        )
        reference = torch.arange(77, dtype=torch.float32).reshape(7, 11)
        (value,) = padding.graph_module(reference)
        torch.testing.assert_close(value[:77], reference.flatten())
        assert value.numel() == reduce.padded_elements
        assert torch.count_nonzero(value[77:]) == 0

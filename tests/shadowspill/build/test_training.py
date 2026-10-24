"""Numerical checks of portable training forms, independent of the Executor."""

import os
import subprocess
import sys
import textwrap
from collections import defaultdict
from itertools import product

import pytest
import torch
from torch import nn
from torch._inductor.standalone_compile import AOTCompiledArtifact
from torch.utils._pytree import tree_flatten

from shadowspill import Program
from shadowspill.build.training import build_forward_backward
from shadowspill.planning.training.bind import bind_training
from shadowspill.step.ordering import StepDataOrdering


class Network(nn.Module):
    def __init__(self, *, tied=False):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        if tied:
            self.layers[1].weight = self.layers[0].weight

    def forward(self, x):
        original = x
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x + original


def objective(model, x, *, labels):
    prediction = model(x)
    return (prediction - labels).square().mean(), {"mean": prediction.mean()}


def replay_training(
    program, model, examples, variant, initial_gradients=None, optimizer_values=None
):
    objects = {
        registration["object_id"]: model.get_parameter(registration["name"])
        for registration in program.bindings["registrations"]
    }
    for item in program.bindings["gradients"]:
        if initial_gradients is not None:
            objects[item["gradient_object_id"]] = initial_gradients[
                item["parameter_name"]
            ].clone()
    if optimizer_values is not None:
        for item in program.bindings["optimizer_bindings"]:
            if item["name"] in optimizer_values and item["object_id"] is not None:
                objects[item["object_id"]] = optimizer_values[item["name"]]
    for slots, (args, kwargs) in zip(
        program.bindings["root_inputs"], examples, strict=True
    ):
        leaves, _ = tree_flatten((args, kwargs))
        for slot in slots:
            source = program.bindings["root_sources"][slot["leaf_index"]]
            if source["kind"] == "USER_INPUT":
                objects[slot["object_id"]] = leaves[source["user_leaf"]]
    assert not program.bindings["fixed_inputs"]
    profiles = {item.task_id: item.profile_id for item in program.tasks}
    functions = {}
    for entry in program.bindings["entrypoints"]:
        if entry["options"].get("variant", variant) != variant:
            continue
        key = profiles[entry["task_id"]]
        if key not in functions:
            functions[key] = AOTCompiledArtifact.deserialize(program.executables[key])
        count = program.bindings["task_inputs"][key]["count"]
        arguments = [None] * count
        for slot in entry["input_slots"]:
            arguments[slot["leaf_index"]] = objects[slot["object_id"]]
        for slot in entry.get("host_inputs", ()):
            arguments[slot["leaf_index"]] = optimizer_values[slot["name"]]
        outputs, _ = tree_flatten(functions[key](*arguments))
        for slot in entry["output_slots"]:
            objects[slot["object_id"]] = outputs[slot["leaf_index"]]
        contributions = defaultdict(list)
        for slot in entry["options"].get("contribution_slots", ()):
            contributions[slot["object_id"]].append(outputs[slot["leaf_index"]])
        for object_id, values in contributions.items():
            contribution = values[0]
            for additional in values[1:]:
                contribution.add_(additional)
            destination = objects.get(object_id)
            if destination is None:
                objects[object_id] = contribution
            elif destination.data_ptr() != contribution.data_ptr():
                destination.add_(contribution)
    return (
        [
            [objects[value] for value in values]
            for values in program.bindings["public_outputs"]
        ],
        {
            item["parameter_name"]: objects[item["gradient_object_id"]]
            for item in program.bindings["gradients"]
        },
    )


@pytest.mark.cuda
@pytest.mark.parametrize("tied", [False, True])
def test_training_forms_and_orderings_match_autograd(tmp_path, tied):
    if not torch.cuda.is_available():
        pytest.skip("finalizing CUDA task kernels needs a representative GPU")
    with torch.device("meta"):
        model = Network(tied=tied)
    program = build_forward_backward(
        model,
        objective=objective,
        example_args=(torch.empty(4, 8),),
        example_kwargs={"labels": torch.empty(4, 8)},
        num_examples=2,
    )
    program.save(tmp_path / "training.ssp")
    program = Program.load(tmp_path / "training.ssp")
    actual_model = Network(tied=tied).cuda()
    examples = [
        (
            (torch.randn(4, 8, device="cuda"),),
            {"labels": torch.randn(4, 8, device="cuda")},
        )
        for _ in range(2)
    ]
    initial = {
        name: torch.randn_like(value) for name, value in actual_model.named_parameters()
    }
    expected_losses = []
    for args, kwargs in examples:
        loss, metrics = objective(actual_model, *args, **kwargs)
        expected_losses.append([loss.detach(), metrics["mean"].detach()])
        loss.backward()
    expected = {
        name: value.grad.clone() for name, value in actual_model.named_parameters()
    }
    orderings = (
        StepDataOrdering.depth_first(2),
        StepDataOrdering(1, 2),
        StepDataOrdering(1, 2, reverse_breadth=False, pair_loss=False),
    )
    for variant, ordering, accumulate in product(
        ("save", "recompute"), orderings, (False, True)
    ):
        bound = bind_training(program, ordering, accumulate=accumulate)
        with torch.no_grad():
            losses, gradients = replay_training(
                bound, actual_model, examples, variant, initial if accumulate else None
            )
        torch.testing.assert_close(losses, expected_losses)
        for name, gradient in gradients.items():
            torch.testing.assert_close(
                gradient,
                expected[name] + (initial[name] if accumulate else 0),
                msg=f"{tied=}, {variant=}, {ordering=}, {accumulate=}, {name}",
            )
        assert bound.executables == program.executables
    if not tied:
        _check_cpu_planning(tmp_path)


def _check_cpu_planning(directory, filename="training.ssp"):
    code = textwrap.dedent("""
        import importlib.abc, sys
        class NoExecutionImports(importlib.abc.MetaPathFinder):
            def find_spec(self, name, *args):
                blocked = ('torch', 'mlops', 'shadowspill.runtime',
                           'shadowspill.pytorch', 'shadowspill.compilation',
                           'shadowspill.store')
                if any(name == part or name.startswith(part + '.') for part in blocked):
                    raise AssertionError('CPU planning imported ' + name)
        sys.meta_path.insert(0, NoExecutionImports())
        from shadowspill import (Program, Profiles, TaskProfile, MemoryBudgets,
            CommunicationInfo, apply_profiles, plan_program, write_report, Plan)
        from shadowspill.step.ordering import StepDataOrdering
        from shadowspill.planner import GenericPlanningOptions, SearchOptions
        p = Program.load(sys.argv[1])
        profiles = Profiles(setup=dict(p.setup), fixed_device_bytes=0, tasks={
            d.profile_key: TaskProfile(duration_seconds=0.001, workspace_bytes=max(
                d.workspace_bytes or 0, p.bindings['compiler_memory'].get(
                    d.profile_key, {}).get('known_workspace_bytes', 0)))
            for d in p.task_definitions})
        pp = apply_profiles(p, profiles)
        signatures = []
        for index, ordering in enumerate((StepDataOrdering.depth_first(2),
                StepDataOrdering(1, 2), StepDataOrdering(1, 2, False, False))):
            plan = plan_program(pp, memory_budgets=MemoryBudgets(16384, 65536),
                communication_info=CommunicationInfo(fetch_bytes_per_second=1e9,
                                                     evict_bytes_per_second=1e9),
                ordering=ordering, search_options=SearchOptions(workers=1,
                    generic=GenericPlanningOptions(minimum_object_bytes_evict_eligible=0)))
            signatures.append(tuple(t.task_id for t in plan.program.tasks))
            assert plan.program.executables == p.executables
            assert plan.program.bindings['ordering'] == ordering.to_dict()
            plan.save(f'plan-{index}.ssp')
            restored = Plan.load(f'plan-{index}.ssp')
            assert restored.to_dict() == plan.to_dict()
            write_report(restored, f'report-{index}')
        assert len(set(signatures)) == 3
        print('All three traversals planned and reported without framework imports')
    """)
    result = subprocess.run(
        [sys.executable, "-S", "-c", code, filename],
        cwd=directory,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr

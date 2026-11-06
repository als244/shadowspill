"""Fresh-process public build/profile/plan/execute with unchanged oracle bounds."""

import copy
import hashlib
import json
import os


def write(directory, name, value):
    (directory / name).write_text(json.dumps(value, indent=2, default=dict) + "\n")


def prepare(directory, case):
    batch, kind = case["batch"], case["kind"]
    import torch

    from .model import Network, examples, objective

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    model = Network(provider="independent")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.001, foreach=False
    )
    initial = copy.deepcopy(model.state_dict())
    references = []
    for step in range(3):
        if kind == "step" or step != 1:
            optimizer.zero_grad(set_to_none=True)
        losses = []
        for args, kwargs in examples(step, batch=batch):
            loss, _ = objective(model, *args, **kwargs)
            losses.append(loss.detach().clone())
            loss.backward()
        gradients = {
            name: p.grad.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        assert all(
            g[:2].count_nonzero() and not g[2:].count_nonzero()
            for g in gradients.values()
        )
        assert all(torch.isfinite(g).all() for g in gradients.values())
        if kind == "step":
            optimizer.step()
        state = {name: p.detach().clone() for name, p in model.named_parameters()}
        if kind == "step":
            state.update(
                {
                    f"optimizer.{name}.{entry}": value.detach().clone()
                    for name, parameter in model.named_parameters()
                    for entry, value in optimizer.state[parameter].items()
                }
            )
        else:
            state.update(
                {"gradient." + name: value for name, value in gradients.items()}
            )
        references.append(dict(losses=losses, gradients=gradients, state=state))
    torch.save(
        dict(
            initial=initial,
            examples=[examples(step, batch=batch) for step in range(3)],
            expected=references,
        ),
        directory / "reference.pt",
    )
    write(
        directory, "settings.json", dict(execution_bytes=4 << 30, spill_bytes=64 << 20)
    )
    assert not torch.cuda.is_initialized()
    write(
        directory,
        "prepared-inputs.json",
        dict(
            status="CPU_INDEPENDENT_REFERENCE_PASS",
            steps=3,
            batch=batch,
            kind=kind,
            sequences=[65, 48],
            rank=32,
            dtype="bfloat16",
            shared_factor_uses=2,
            reference_sha256=hashlib.sha256(
                (directory / "reference.pt").read_bytes()
            ).hexdigest(),
            cuda_initialized=False,
            limits="No GPU build, destination execution or DP qualification.",
        ),
    )


def build(directory, case):
    import torch
    from mlops.dispatch.context import weight_gradients_at
    from mlops.integrations.shadowspill import sonicmoe_gradient_destination_rules

    from shadowspill import OperatorPreparation, build_forward_backward, build_step
    from shadowspill.pytorch import IsolateOperations

    from .model import Network, examples, objective
    from .policy import Chunks, pair_policy

    model = Network(device="meta")
    builder = build_step if case["kind"] == "step" else build_forward_backward
    arguments = {}
    if case["kind"] == "step":
        arguments["optimizer"] = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.001,
            foreach=False,
        )
    with weight_gradients_at(torch.bfloat16):
        program = builder(
            model,
            **arguments,
            objective=objective,
            example_calls=examples(batch=case["batch"]),
            partition=IsolateOperations(("aten.sin.default",), base="whole"),
            chunking=Chunks(case["variant"], batch=case["batch"]),
            graph_pair_policy=pair_policy(),
            grad_dtype=torch.bfloat16,
            fuse_narrow_gradient_accumulation=True,
            gradient_destination_rules=sonicmoe_gradient_destination_rules(),
            operator_preparation=(
                OperatorPreparation(
                    "mlops.preparation.sonic:factory",
                    {"cache_root": str(directory / "prepared-cache")},
                ),
            ),
        )
    dependencies = {
        item["operator"]
        for items in program.bindings["operator_dependencies"].values()
        for item in items
    }
    assert {
        f"mlops::lora_sonicmoe_{direction}_bwd_accumulate"
        for direction in ("up", "down")
    } <= dependencies
    assert program.operator_binaries
    assert set(program.bindings["operator_preparation"]["tasks"]) == set(
        program.executables
    )
    program.save(directory / "program.ssp")
    write(
        directory,
        "built.json",
        dict(
            tasks=len(program.tasks),
            executable_keys=len(program.executables),
            binary_blobs=len(program.operator_binaries),
            binary_bytes=sum(
                len(value) for value in program.operator_binaries.values()
            ),
            dependencies=sorted(dependencies),
        ),
    )


def profile(directory, case):

    from shadowspill import CommunicationInfo, Program, apply_profiles, measure_program

    from .guards import forbid_provider_tuning

    program = Program.load(directory / "program.ssp")
    forbid_provider_tuning(program.setup["capability"])
    profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.003)
    profiles.save(directory / "profiles.ssp")
    apply_profiles(program, profiles).save(directory / "profiled.ssp")
    CommunicationInfo(fetch_bytes_per_second=1e10, evict_bytes_per_second=1e10).save(
        directory / "communication.ssp"
    )


def plan(directory, case):

    from qualification.pipeline.plan import run

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    run(directory)


def execute(directory, case, *, resume=False):
    kind = case["kind"]
    import torch

    from qualification.runtime_evidence import (
        adapter_statistics,
        check_physical_budget,
        statistics_dict,
    )
    from shadowspill import Executor, Plan, load_checkpoint, save_checkpoint
    from shadowspill.memory import pinned_host
    from tests.shadowspill.execution._cold import forbid_preparation
    from tests.shadowspill.execution._trace_checks import check_trace

    from .guards import forbid_provider_tuning

    plan = Plan.load(directory / "plan.ssp")
    forbid_provider_tuning(plan.program.setup["capability"])
    forbid_preparation(
        "tests.workloads.prepared_sonic.model",
        "tests.workloads.prepared_sonic.policy",
        "mlops.integrations.shadowspill",
    )
    reference = torch.load(directory / "reference.pt", weights_only=True)
    from .observe import DestinationObserver

    observer = DestinationObserver()

    def initialize(request, chunk):
        value = reference["initial"][request.name].reshape(-1)
        chunk.copy_(value[request.offset : request.offset + chunk.numel()])

    steps = (1, 2) if resume else (0, 1, 2)
    prefix = "resume-" if resume else ""
    physical = []
    executor = Executor(
        plan,
        spill=pinned_host(capacity=plan.memory_budgets.spill_bytes),
        reject_overbudget=True,
    )
    try:
        executor.initialize(initialize, chunk_bytes=4096)
        assert adapter_statistics().physical_budget_sealed
        if resume:
            assert load_checkpoint(
                executor, directory / "checkpoint.ssp", chunk_bytes=4096
            ) == {"step": 1}
        for step in steps:
            calls = reference["examples"][step]
            observer.phase = f"invocation-{step}"
            pending = executor.run(
                input_loader=lambda i, calls=calls: calls[i],
                runtime_trace=True,
                accumulate_gradients=kind == "forward_backward" and step == 1,
            )
            result = pending.result()
            selected_plan = (
                plan._accumulation_plan
                if kind == "forward_backward" and step == 1
                else plan
            )
            assert selected_plan is not None
            check_trace(selected_plan, pending.diagnostics)
            assert check_physical_budget() == 0
            statistics = adapter_statistics()
            assert statistics.physical_budget_sealed
            assert (
                0
                < statistics.peak_process_physical_bytes
                <= (plan.memory_budgets.execution_bytes)
            )
            physical.append(dict(step=step, **statistics_dict(statistics)))
            actual = {}

            def collect(request, chunk, actual=actual):
                if request.name not in actual:
                    actual[request.name] = torch.empty(
                        request.storage_elements, dtype=request.dtype
                    )
                actual[request.name][
                    request.offset : request.offset + chunk.numel()
                ].copy_(chunk)

            executor.export_tensors(collect, chunk_bytes=4096)
            torch.save(
                dict(
                    losses=[loss.detach().cpu().clone() for loss in result.losses],
                    state=actual,
                ),
                directory / f"{prefix}actual-{step}.pt",
            )
            if step == 0:
                save_checkpoint(
                    executor,
                    directory / "checkpoint.ssp",
                    metadata={"step": 1},
                    chunk_bytes=4096,
                )
                executor.initialize(lambda request, chunk: chunk.zero_(), restore=True)
                assert load_checkpoint(
                    executor, directory / "checkpoint.ssp", chunk_bytes=4096
                ) == {"step": 1}
                restored = {}

                def collect_restored(request, chunk, restored=restored):
                    if request.name not in restored:
                        restored[request.name] = torch.empty(
                            request.storage_elements, dtype=request.dtype
                        )
                    restored[request.name][
                        request.offset : request.offset + chunk.numel()
                    ].copy_(chunk)

                executor.export_tensors(collect_restored, chunk_bytes=4096)
                assert restored.keys() == actual.keys()
                assert all(
                    torch.equal(restored[name], value) for name, value in actual.items()
                )
                write(
                    directory,
                    "checkpoint-inspection.json",
                    dict(status="BITWISE_ALL_EXPORTED_STATE", tensors=len(actual)),
                )
            del result, pending
    finally:
        executor.close()
    write(directory, prefix + "destination-inspection.json", observer.report(steps))
    write(
        directory,
        prefix + "execution.json",
        dict(
            status="COLD_SAVED_INVOCATIONS_COMPLETED_REQUIRES_NUMERICAL_AUDIT",
            steps=len(steps),
            spill="pinned_host",
            checkpoint_restored_after=1,
            kind=kind,
            accumulate=[kind == "forward_backward" and step == 1 for step in steps],
            physical=physical,
        ),
    )


def inspect(directory, case):
    kind = case["kind"]
    from dataclasses import asdict

    import torch

    from qualification.numerical.metrics import tensor_metrics
    from qualification.numerical.tolerances import (
        LOSS_ABSOLUTE_TOLERANCE,
        LOSS_RELATIVE_TOLERANCE,
        meets_tensor_tolerance,
    )

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    reference = torch.load(directory / "reference.pt", weights_only=True)
    comparisons, failures = [], []
    previous_moments = {}
    for step, wanted in enumerate(reference["expected"]):
        actual = torch.load(directory / f"actual-{step}.pt", weights_only=True)
        assert actual["state"].keys() == wanted["state"].keys()
        for name, expected in wanted["state"].items():
            value = actual["state"][name].reshape_as(expected)
            assert value.dtype == expected.dtype and torch.isfinite(value).all()
            half = (
                "gradient"
                if name.startswith("gradient.")
                else ("optimizer" if name.startswith("optimizer.") else "model")
            )
            metric = tensor_metrics(expected, value)
            passed = meets_tensor_tolerance(
                metric, key=f"state/{half}/{name}", step_size=0.001, steps=step + 1
            )
            if name.endswith(".step") or (
                kind == "forward_backward" and half == "model"
            ):
                passed = torch.equal(value, expected)
            row = dict(step=step, name=name, passed=passed, **asdict(metric))
            comparisons.append(row)
            if not passed:
                failures.append(row)
            if name.endswith(".exp_avg"):
                factor = name.removeprefix("optimizer.").removesuffix(".exp_avg")
                prior = previous_moments.get(name, torch.zeros_like(value))
                # Diagnostic inferred gradient, kept separate from Adam state
                # acceptance because the moment itself is rounded to BF16.
                inferred = ((value.float() - 0.9 * prior.float()) / 0.1).to(value.dtype)
                comparisons.append(
                    dict(
                        step=step,
                        name="inferred_gradient." + factor,
                        diagnostic=True,
                        **asdict(tensor_metrics(wanted["gradients"][factor], inferred)),
                    )
                )
                previous_moments[name] = value.clone()
        for microbatch, (value, expected) in enumerate(
            zip(actual["losses"], wanted["losses"], strict=True)
        ):
            passed = torch.allclose(
                value,
                expected,
                rtol=LOSS_RELATIVE_TOLERANCE,
                atol=LOSS_ABSOLUTE_TOLERANCE,
            )
            if not passed:
                failures.append(
                    dict(
                        step=step,
                        microbatch=microbatch,
                        kind="loss",
                        actual=float(value),
                        expected=float(expected),
                    )
                )
    pointers = json.loads((directory / "destination-inspection.json").read_text())
    assert pointers["status"] == "PASS_SELECTED_PRODUCTS_USE_EXISTING_C_AND_OUTPUT"
    assert all(pointers["counts"][f"invocation-{step}"] > 0 for step in range(3))
    write(
        directory,
        "numerical-inspection.json",
        dict(
            status="PASS" if not failures else "FAIL",
            comparisons=comparisons,
            failures=failures,
            thresholds="Unchanged qualification.numerical.tolerances",
            scope=(
                "Small real SonicMoE saved training fixture; "
                "not model-family/DP qualification."
            ),
        ),
    )
    assert not failures, failures

    # Resume happens in another process with unrelated empty compiler caches.
    # Every loss and exported tensor must reproduce the original invocation.
    resumed_tensors = 0
    for step in (1, 2):
        actual = torch.load(directory / f"actual-{step}.pt", weights_only=True)
        resumed = torch.load(directory / f"resume-actual-{step}.pt", weights_only=True)
        assert actual["state"].keys() == resumed["state"].keys()
        assert len(actual["losses"]) == len(resumed["losses"])
        for name, value in actual["state"].items():
            assert torch.equal(value, resumed["state"][name]), (step, name)
            resumed_tensors += 1
        assert all(
            torch.equal(a, b)
            for a, b in zip(actual["losses"], resumed["losses"], strict=True)
        )
    resumed = json.loads((directory / "resume-destination-inspection.json").read_text())
    assert resumed["status"] == "PASS_SELECTED_PRODUCTS_USE_EXISTING_C_AND_OUTPUT"
    assert all(resumed["counts"][f"invocation-{step}"] > 0 for step in (1, 2))
    write(
        directory,
        "resume-inspection.json",
        dict(
            status="BITWISE_FRESH_PROCESS_RESUME",
            tensors=resumed_tensors,
            steps=2,
        ),
    )

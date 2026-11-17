"""All-value admission rejects unknown provider behavior despite measured samples."""

import subprocess
import sys
from dataclasses import replace

import pytest

from engine_tests.fixtures.definition_operations import definitions
from engine_tests.offline._example import program
from engine_tests.training.test_model_build import prepare
from engine_tests.training.test_preparation_templates import inventory
from shadowspill.artifacts._values import plain
from shadowspill.artifacts.program import TaskDefinition
from shadowspill.training.definitions import ModelBuild
from shadowspill.training.definitions.allocations import allocation_invariants
from shadowspill.training.preparation.capture.allocation_coverage import (
    allocation_coverage,
    validate_allocation_coverage,
)


def compiler_fixture(template, *, opaque=(), unknown=()):
    """A description-only fixture; never evidence of native execution admission."""
    return replace(
        program(),
        executables={"update_impl": b"compiled fixture"},
        bindings={
            "kind": template.kind,
            "preparation_template": template.signature(),
            "profiling_scenario": template.signature()["scenario"],
            "compiler_memory": {
                "update_impl": {
                    "allocation_behavior": {
                        "version": 1,
                        "scope": (
                            "fixed tensor geometry/layout and captured non-tensor "
                            "arguments; all tensor values"
                        ),
                        "opaque_calls": list(opaque),
                        "unresolved": list(unknown),
                    }
                }
            },
        },
    )


def certify(template, captured):
    return replace(
        captured,
        bindings={
            **captured.bindings,
            "allocation_coverage": allocation_coverage(template, captured),
        },
    )


def model_contract(contract="fixture/allocate-like-input@1"):
    def describe(config, context):
        return ModelBuild(
            vars(config), allocation_invariants={"provider.call": contract}
        )

    return replace(definitions()[0].definition, build=describe)


def test_measured_bytes_cannot_supply_missing_all_value_allocation_evidence():
    template = next(iter(inventory().templates.values()))
    captured = compiler_fixture(template, opaque=["provider.data_dependent"])
    # Even a known workspace integer is not a guarantee about all input values.
    captured = replace(
        captured,
        task_definitions=tuple(
            replace(item, workspace_bytes=16) for item in captured.task_definitions
        ),
    )
    with pytest.raises(ValueError, match=r"missing all-value.*provider.data_dependent"):
        allocation_coverage(template, captured)
    with pytest.raises(ValueError, match="every compiler memory"):
        allocation_coverage(template, replace(captured, bindings={}))


@pytest.mark.parametrize(
    "operator, contract",
    [
        (name, "shadowspill/aten-dense-matmul-allocation@1")
        for name in ("extern_kernels.mm", "extern_kernels.mm_dtype")
    ]
    + [
        (name, "shadowspill/gradient-storage-allocation@1")
        for name in (
            "torch.ops.shadowspill.materialize_gradient.default",
            "torch.ops.shadowspill.pad_storage.default",
        )
    ],
)
def test_opaque_contracts_are_explicit_source_owned_and_never_cover_unknown_wrappers(
    operator,
    contract,
):
    _, prepared = prepare(model_contract())
    template = next(iter(prepared.templates.values()))
    captured = compiler_fixture(template, opaque=["provider.call", operator])
    saved = certify(template, captured)
    validate_allocation_coverage(template, saved)
    task = saved.bindings["allocation_coverage"]["tasks"]["update_impl"]
    assert task["contracts"]["provider.call"] == "fixture/allocate-like-input@1"
    assert task["contracts"][operator] == contract
    with pytest.raises(ValueError, match="missing all-value"):
        allocation_coverage(
            template, compiler_fixture(template, opaque=[operator + "_wrapper"])
        )
    with pytest.raises(ValueError, match="unresolved allocation behavior"):
        allocation_coverage(
            template,
            compiler_fixture(
                template, opaque=["provider.call"], unknown=["data-dependent extent"]
            ),
        )
    _, changed = prepare(model_contract("fixture/allocate-like-input@2"))
    assert prepared.templates.keys().isdisjoint(changed.templates)
    with pytest.raises(ValueError, match="saved all-value allocation coverage"):
        validate_allocation_coverage(next(iter(changed.templates.values())), saved)


def test_additive_forms_and_preparation_activation_require_their_own_coverage():
    template = next(iter(inventory().templates.values()))
    captured = compiler_fixture(template)
    extra = replace(
        captured,
        task_definitions=(
            *captured.task_definitions,
            TaskDefinition("additive", "additive"),
        ),
        executables={**captured.executables, "additive": b"another compiled form"},
    )
    with pytest.raises(ValueError, match="every compiler memory"):
        allocation_coverage(template, extra)
    with pytest.raises(
        ValueError, match=r"missing bound-task.*provider.factory:create"
    ):
        allocation_coverage(
            template,
            replace(
                captured,
                bindings={
                    **captured.bindings,
                    "operator_preparation": {
                        "providers": [{"factory": "provider.factory:create"}]
                    },
                },
            ),
        )


@pytest.mark.parametrize("change", ["code", "compiler", "environment", "proof"])
def test_cpu_recheck_rejects_stale_saved_allocation_proofs(change):
    template = next(iter(inventory().templates.values()))
    saved = certify(template, compiler_fixture(template))
    if change == "code":
        saved = replace(saved, executables={"update_impl": b"different code"})
    elif change == "environment":
        saved = replace(saved, setup={"changed": "toolchain"})
    else:
        bindings = plain(saved.bindings)
        if change == "compiler":
            bindings["compiler_memory"]["update_impl"]["known_workspace_bytes"] = 32
        else:
            bindings["allocation_coverage"]["tasks"].clear()
        saved = replace(saved, bindings=bindings)
    with pytest.raises(ValueError, match="saved all-value allocation coverage"):
        validate_allocation_coverage(template, saved)


@pytest.mark.parametrize(
    "values",
    [
        {"provider.*": "test@1"},
        {"provider.call": "unversioned"},
        {"provider.call": "@1"},
        {"provider.call": True},
    ],
)
def test_contracts_never_accept_wildcards_or_unversioned_claims(values):
    with pytest.raises(ValueError):
        allocation_invariants(values)


def test_objective_optimizer_and_model_contracts_are_immutable_and_conflicts_fail():
    model, objective, optimizer = definitions()
    contract = {"extern_kernels.mm": "unreviewed/replacement@1"}
    for item in (
        replace(objective.definition, allocation_invariants=contract),
        replace(optimizer.definition, allocation_invariants=contract),
        ModelBuild({}, allocation_invariants=contract),
    ):
        with pytest.raises(TypeError):
            item.allocation_invariants["added"] = "test@1"
    _, prepared = prepare(
        replace(
            model.definition,
            build=lambda config, context: ModelBuild(
                vars(config), allocation_invariants=contract
            ),
        )
    )
    template = next(iter(prepared.templates.values()))
    with pytest.raises(ValueError, match="conflicting allocation contract"):
        allocation_coverage(template, compiler_fixture(template))


def test_allocation_proof_is_inspectable_without_frameworks_or_provider_imports():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
for name in ('torch', 'triton', 'mlops', 'provider', 'shadowspill_workloads'):
    sys.modules[name] = None
from engine_tests.training.test_allocation_coverage import (
    prepare, model_contract, compiler_fixture, certify, validate_allocation_coverage,
)
_, prepared = prepare(model_contract())
template = next(iter(prepared.templates.values()))
saved = certify(template, compiler_fixture(template, opaque=['provider.call']))
validate_allocation_coverage(template, saved)
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_compiler_seed_fill_has_an_exact_contract_and_retains_output_storage():
    import torch

    from shadowspill.pytorch._rng import preserve_rng

    template = next(iter(inventory().templates.values()))
    saved = certify(
        template, compiler_fixture(template, opaque=["aten.randint.low_out"])
    )
    validate_allocation_coverage(template, saved)
    contracts = saved.bindings["allocation_coverage"]["tasks"]["update_impl"][
        "contracts"
    ]
    assert contracts == {
        "aten.randint.low_out": "shadowspill/aten-randint-out-allocation@1"
    }
    with pytest.raises(ValueError, match="missing all-value"):
        allocation_coverage(
            template, compiler_fixture(template, opaque=["aten.randint.default"])
        )
    with preserve_rng(torch.device("cpu")):
        output = torch.empty(7, dtype=torch.int64)
        pointer, size = output.data_ptr(), output.untyped_storage().nbytes()
        for seed, fill in ((0, -1), (91, 0), (2**63 + 1, 2**63 - 1)):
            torch.manual_seed(seed)
            output.fill_(fill)
            result = torch.ops.aten.randint.low_out(
                -(2**63), 2**63 - 1, [7], out=output
            )
            assert result.data_ptr() == pointer and output.shape == (7,)
            assert output.untyped_storage().nbytes() == size

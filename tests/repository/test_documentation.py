"""Keep public documentation aligned with source-level API boundaries."""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
PYTHON_API = DOCS / "python" / "api"

_MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
_PYTHON_FENCE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
_JSON_FENCE = re.compile(r"```json\n(.*?)\n```", re.DOTALL)
_SIGNATURE = re.compile(
    r"<!-- source-signature: ([^:]+):([A-Za-z0-9_.]+) -->\s*"
    r"```text\n(.*?)\n```",
    re.DOTALL,
)
_C_FUNCTION = re.compile(r"\b(shadowspill_[A-Za-z0-9_]+)\s*\(")

_MEMORY_MODULE = ROOT / "src" / "shadowspill" / "memory.py"
_PYTORCH_MODULE = ROOT / "src" / "shadowspill" / "pytorch" / "__init__.py"
_CORE_PYTHON_MODULES = (
    ROOT / "src" / "shadowspill" / "errors.py",
    ROOT / "src" / "shadowspill" / "ir" / "__init__.py",
    ROOT / "src" / "shadowspill" / "planner" / "__init__.py",
    ROOT / "src" / "shadowspill" / "simulator" / "__init__.py",
    ROOT / "src" / "shadowspill" / "runtime" / "__init__.py",
)
_PUBLIC_PYTHON_MODULES = (
    _MEMORY_MODULE,
    *_CORE_PYTHON_MODULES,
    _PYTORCH_MODULE,
)

_PUBLIC_HEADERS = ROOT / "csrc" / "include" / "shadowspill"

_PUBLIC_C_REFERENCES = {
    _PUBLIC_HEADERS / "runtime.h": DOCS / "c" / "runtime.md",
    _PUBLIC_HEADERS / "admission_replay.h": DOCS / "c" / "runtime.md",
    _PUBLIC_HEADERS / "backend.h": DOCS / "c" / "backends.md",
    _PUBLIC_HEADERS / "profiler.h": DOCS / "c" / "backends.md",
    _PUBLIC_HEADERS / "planner.h": DOCS / "c" / "planner.md",
    _PUBLIC_HEADERS / "simulator.h": DOCS / "c" / "simulator.md",
    ROOT
    / "csrc"
    / "adapter"
    / "pytorch"
    / "include"
    / "shadowspill"
    / "pytorch_adapter.h": DOCS / "c" / "pytorch-adapter.md",
}

_REQUIRED_SIGNATURES = {
    "src/shadowspill/pytorch/api.py:make_step_program",
    "src/shadowspill/pytorch/api.py:plan_forward",
    "src/shadowspill/pytorch/api.py:plan_step",
    "src/shadowspill/planner/plan.py:pressurefit_program",
    "src/shadowspill/pytorch/callables.py:PlannedForward.__call__",
    "src/shadowspill/pytorch/callables.py:PlannedForward.submit",
    "src/shadowspill/pytorch/callables.py:PlannedTrainStep.__call__",
    "src/shadowspill/pytorch/callables.py:PlannedTrainStep.submit",
    "src/shadowspill/pytorch/runtime_adapter/runtime.py:Runtime.__init__",
    "src/shadowspill/pytorch/state/model.py:export_model_state",
    "src/shadowspill/pytorch/state/model.py:import_model_state",
    "src/shadowspill/pytorch/state/model.py:release_model_state",
    "src/shadowspill/pytorch/state/optimizer.py:export_optimizer_state",
    "src/shadowspill/pytorch/state/optimizer.py:import_optimizer_state",
}


def _markdown_files() -> tuple[Path, ...]:
    ignored = {
        ".cache",
        ".git",
        ".hypothesis",
        ".mypy_cache",
        ".venv",
        "build",
        "datasets",
        "planning_caches",
        "results",
    }
    return tuple(
        path
        for path in ROOT.rglob("*.md")
        if not any(part in ignored for part in path.parts)
        and not path.is_relative_to(DOCS / "internal")
    )


def _normative_markdown_files() -> tuple[Path, ...]:
    return tuple(
        path
        for path in _markdown_files()
        if not path.is_relative_to(DOCS / "investigations")
    )


def _local_link(
    document: Path,
    raw: str,
) -> tuple[Path | None, str | None]:
    value = raw.strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]
    destination = value.split(maxsplit=1)[0]
    if (
        not destination
        or "://" in destination
        or destination.startswith(("mailto:", "plugin:"))
    ):
        return None, None
    path_value, separator, fragment = destination.partition("#")
    target = document if not path_value else (document.parent / unquote(path_value))
    return target.resolve(), unquote(fragment) if separator else None


def _heading_slug(value: str) -> str:
    plain = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    plain = re.sub(r"<[^>]+>", "", plain)
    plain = plain.replace("`", "").lower()
    plain = re.sub(r"[^\w\- ]", "", plain)
    return re.sub(r"\s+", "-", plain.strip())


def _heading_anchors(document: Path) -> set[str]:
    occurrences: defaultdict[str, int] = defaultdict(int)
    anchors: set[str] = set()
    for heading in _MARKDOWN_HEADING.findall(document.read_text()):
        base = _heading_slug(heading)
        suffix = occurrences[base]
        occurrences[base] += 1
        anchors.add(base if suffix == 0 else f"{base}-{suffix}")
    return anchors


def _all_exports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise AssertionError(f"{path}: __all__ must be a literal string list")
        return tuple(value)
    raise AssertionError(f"{path}: public module has no literal __all__")


def _documented_symbol(reference: str, name: str) -> bool:
    return re.search(rf"`{re.escape(name)}(?:`|\()", reference) is not None


def _python_page_expectations() -> dict[Path, set[str]]:
    expectations = {
        PYTHON_API / "frontend.md": set(_all_exports(_MEMORY_MODULE)),
        PYTHON_API / "artifacts.md": set(),
        PYTHON_API / "diagnostics.md": set(),
        PYTHON_API / "neutral.md": set(),
    }
    for path in _CORE_PYTHON_MODULES:
        expectations[PYTHON_API / "neutral.md"].update(_all_exports(path))

    exports = set(_all_exports(_PYTORCH_MODULE))
    tree = ast.parse(_PYTORCH_MODULE.read_text(), filename=str(_PYTORCH_MODULE))
    found: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.level != 1:
            continue
        page = {
            "diagnostics": PYTHON_API / "diagnostics.md",
            "program": PYTHON_API / "artifacts.md",
        }.get(node.module, PYTHON_API / "frontend.md")
        for alias in node.names:
            public_name = alias.asname or alias.name
            if public_name in exports:
                expectations[page].add(public_name)
                found.add(public_name)
    assert found == exports, (
        f"unclassified shadowspill.pytorch exports: {exports - found}"
    )
    return expectations


def _find_callable(path: Path, qualified_name: str) -> ast.FunctionDef:
    nodes: list[ast.stmt] = ast.parse(path.read_text(), filename=str(path)).body
    found: ast.AST | None = None
    for part in qualified_name.split("."):
        found = next(
            (
                node
                for node in nodes
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name == part
            ),
            None,
        )
        if found is None:
            raise AssertionError(f"{path}: no callable {qualified_name}")
        nodes = found.body if isinstance(found, ast.ClassDef) else []
    if not isinstance(found, ast.FunctionDef):
        raise AssertionError(f"{path}: {qualified_name} is not a function")
    return found


def _canonical_default(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    return ast.dump(node, annotate_fields=False, include_attributes=False)


def _source_parameters(path: Path, qualified_name: str) -> list[tuple[str, str | None]]:
    function = _find_callable(path, qualified_name)
    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    positional_defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(arguments.defaults)
    ) + list(arguments.defaults)
    parameters = [
        (argument.arg, _canonical_default(default))
        for argument, default in zip(positional, positional_defaults, strict=True)
        if argument.arg not in {"self", "cls"}
    ]
    parameters.extend(
        (
            argument.arg,
            _canonical_default(default),
        )
        for argument, default in zip(
            arguments.kwonlyargs, arguments.kw_defaults, strict=True
        )
    )
    return parameters


def _documented_parameters(signature: str) -> list[tuple[str, str | None]]:
    match = re.search(r"\((.*)\)", signature, re.DOTALL)
    if match is None:
        raise AssertionError(f"invalid documented signature: {signature!r}")
    parameters: list[tuple[str, str | None]] = []
    raw_parameters: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    body = match.group(1)
    for index, character in enumerate(body):
        if quote is not None:
            if character == quote and (index == 0 or body[index - 1] != "\\"):
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif character == "," and depth == 0:
            raw_parameters.append(body[start:index])
            start = index + 1
    raw_parameters.append(body[start:])
    for raw in raw_parameters:
        value = raw.strip()
        if not value or value in {"*", "/"}:
            continue
        name_value, separator, default_value = value.partition("=")
        name = name_value.partition(":")[0].strip().lstrip("*")
        default = None
        if separator:
            default_node = ast.parse(default_value.strip(), mode="eval").body
            default = _canonical_default(default_node)
        parameters.append((name, default))
    return parameters


def test_all_local_markdown_links_and_anchors_resolve() -> None:
    missing: list[str] = []
    anchor_cache: dict[Path, set[str]] = {}
    for document in _markdown_files():
        for raw in _MARKDOWN_LINK.findall(document.read_text()):
            target, fragment = _local_link(document, raw)
            if target is None:
                continue
            if not target.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
                continue
            if fragment is not None and target.suffix == ".md":
                anchors = anchor_cache.setdefault(target, _heading_anchors(target))
                if fragment not in anchors:
                    missing.append(
                        f"{document.relative_to(ROOT)} -> "
                        f"{target.relative_to(ROOT)}#{fragment}"
                    )
    assert not missing, "missing local Markdown links or anchors:\n" + "\n".join(
        missing
    )


def test_documentation_index_exposes_reading_paths() -> None:
    index = (DOCS / "README.md").read_text()
    for target in (
        "architecture/overview.md",
        "architecture/ir.md",
        "architecture/lowering.md",
        "architecture/graph-pair-construction.md",
        "architecture/pressurefit.md",
        "architecture/recomputation-selection.md",
        "architecture/physical-admission.md",
        "architecture/planning.md",
        "architecture/simulation.md",
        "architecture/memory-runtime.md",
        "python/README.md",
        "python/plan-report.md",
        "python/step-diagnostics.md",
        "python/planning-json.md",
        "python/failures.md",
        "examples/README.md",
        "examples/training-lifecycle.md",
        "examples/forward-only.md",
        "examples/reusable-planning.md",
        "examples/diagnostics.md",
        "examples/custom-partitioning.md",
        "c/README.md",
        "development/README.md",
        "investigations/README.md",
    ):
        assert f"]({target})" in index


def test_root_readme_remains_a_minimal_entrypoint() -> None:
    readme = (ROOT / "README.md").read_text()
    sections = re.findall(r"^## (.+)$", readme, re.MULTILINE)
    assert sections == [
        "Installation",
        "Minimal example",
        "Project structure",
        "Documentation",
    ]
    assert len(readme.splitlines()) <= 110


def test_architecture_overview_starts_with_purpose_and_visuals() -> None:
    overview = (DOCS / "architecture" / "overview.md").read_text()
    for required in (
        "## Why ShadowSpill exists",
        "## System at a glance",
        "## How planning responsibilities differ",
        "## Runtime interaction",
    ):
        assert required in overview
    assert overview.count("```mermaid") >= 3


def test_architecture_index_groups_one_flat_reading_path() -> None:
    index = (DOCS / "README.md").read_text()
    for heading in (
        "### Foundations",
        "### PyTorch lowering",
        "### Planning",
        "### Execution",
    ):
        assert heading in index


def test_examples_cover_complete_public_workflows() -> None:
    examples = DOCS / "examples"
    expected = {
        "training-lifecycle.md": (
            "Runtime(",
            "import_model_state(",
            "plan_step(",
            "state_dict()",
            "train_step.close()",
        ),
        "forward-only.md": ("plan_forward(", "run_forward.close()"),
        "reusable-planning.md": (
            "make_step_program(",
            "pressurefit_program(",
            "StepProgram.from_json(",
            "annotated.to_json()",
        ),
        "diagnostics.md": (
            "report.diagnostics.task(",
            "step.simulator_comparison",
        ),
        "custom-partitioning.md": ("assign_stages(", "partition=EveryNNodes("),
    }
    for name, required in expected.items():
        reference = (examples / name).read_text()
        for value in required:
            assert value in reference
        for source in _PYTHON_FENCE.findall(reference):
            tree = ast.parse(source)
            assert not any(isinstance(node, ast.Try) for node in ast.walk(tree))


def test_failure_guide_covers_public_boundaries_and_cleanup() -> None:
    reference = (DOCS / "python" / "failures.md").read_text()
    for required in (
        "RuntimeConfigurationError",
        "PlanningError",
        "CaptureError",
        "CompilationError",
        "ProfilingError",
        "AdmissionError",
        "PlanInfeasibleError",
        "PlanSearchExhaustedError",
        "ObjectiveError",
        "InputGuardError",
        "RuntimeExecutionError",
        "RuntimeFailureDiagnostics",
        "task_allocation_envelope_exceeded",
        "task_allocation_contract_mismatch",
        "## Execution rollback",
        "## Normal close order",
    ):
        assert required in reference


def test_every_public_python_export_appears_on_its_api_page() -> None:
    missing = {
        page.relative_to(ROOT).as_posix(): sorted(
            name for name in names if not _documented_symbol(page.read_text(), name)
        )
        for page, names in _python_page_expectations().items()
    }
    missing = {path: names for path, names in missing.items() if names}
    assert not missing, f"misplaced or undocumented public Python exports: {missing}"

    all_exports = {
        name for path in _PUBLIC_PYTHON_MODULES for name in _all_exports(path)
    }
    all_reference = "\n".join(path.read_text() for path in PYTHON_API.glob("*.md"))
    undocumented = sorted(
        name for name in all_exports if not _documented_symbol(all_reference, name)
    )
    assert not undocumented, f"undocumented public Python exports: {undocumented}"


def test_every_public_c_function_appears_on_its_component_page() -> None:
    missing = {
        header.relative_to(ROOT).as_posix(): sorted(
            name
            for name in set(_C_FUNCTION.findall(header.read_text()))
            if not _documented_symbol(reference.read_text(), name)
        )
        for header, reference in _PUBLIC_C_REFERENCES.items()
    }
    missing = {path: names for path, names in missing.items() if names}
    assert not missing, f"misplaced or undocumented public C functions: {missing}"


def test_documented_python_signatures_match_source() -> None:
    observed: set[str] = set()
    mismatches: list[str] = []
    for document in _normative_markdown_files():
        for relative_path, qualified_name, signature in _SIGNATURE.findall(
            document.read_text()
        ):
            identity = f"{relative_path}:{qualified_name}"
            observed.add(identity)
            source = ROOT / relative_path
            expected = _source_parameters(source, qualified_name)
            actual = _documented_parameters(signature)
            if actual != expected:
                mismatches.append(
                    f"{document.relative_to(ROOT)} {identity}: "
                    f"documented={actual!r}, source={expected!r}"
                )
    assert observed == _REQUIRED_SIGNATURES, (
        f"signature marker mismatch: missing={_REQUIRED_SIGNATURES - observed}, "
        f"unexpected={observed - _REQUIRED_SIGNATURES}"
    )
    assert not mismatches, "stale documented Python signatures:\n" + "\n".join(
        mismatches
    )


def test_normative_python_examples_compile_and_use_valid_public_keywords() -> None:
    failures: list[str] = []
    public_parameters: dict[str, set[str]] = {}
    for document in _normative_markdown_files():
        for relative_path, qualified_name, signature in _SIGNATURE.findall(
            document.read_text()
        ):
            display_name = signature.split("(", 1)[0].strip()
            public_parameters[display_name] = {
                name
                for name, _ in _source_parameters(ROOT / relative_path, qualified_name)
            }
    for document in _normative_markdown_files():
        for index, source in enumerate(_PYTHON_FENCE.findall(document.read_text())):
            try:
                compile(source, f"{document}#python-{index}", "exec")
            except SyntaxError as error:
                failures.append(f"{document.relative_to(ROOT)} block {index}: {error}")
                continue
            tree = ast.parse(source)
            for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
                if not isinstance(call.func, ast.Name):
                    continue
                allowed = public_parameters.get(call.func.id)
                if allowed is None:
                    continue
                unexpected = sorted(
                    keyword.arg
                    for keyword in call.keywords
                    if keyword.arg is not None and keyword.arg not in allowed
                )
                if unexpected:
                    failures.append(
                        f"{document.relative_to(ROOT)} block {index}: "
                        f"{call.func.id} has unknown keywords {unexpected}"
                    )
    assert not failures, "invalid documented Python examples:\n" + "\n".join(failures)


def test_pressurefit_architecture_covers_the_algorithm_contract() -> None:
    reference = (DOCS / "architecture" / "pressurefit.md").read_text()
    for required in (
        "## Inputs",
        "## Output",
        "## Mathematical formulation",
        "## Current algorithm",
        "## Pseudocode",
        "PressureFitOptions",
        "PressureFitResult",
        "AdmissionFacts",
        "PressureFitInfeasibleError",
        "PressureFitSearchExhaustedError",
    ):
        assert required in reference

    recomputation = (DOCS / "architecture" / "recomputation-selection.md").read_text()
    for required in (
        "## Inputs and output",
        "## The current selection policy",
        "## Pseudocode",
        "RecomputationGroup",
        "RecomputationOption",
        "RecomputationSelection",
    ):
        assert required in recomputation

    graph_pairs = (DOCS / "architecture" / "graph-pair-construction.md").read_text()
    for required in (
        "## Inputs and output",
        "## Saved-value accounting",
        "## Compilation and profiling",
        "## Lowering into Program alternatives",
        "TaskGraphPairs",
        "GraphPairVariant",
        "RecomputationOption",
    ):
        assert required in graph_pairs

    admission = (DOCS / "architecture" / "physical-admission.md").read_text()
    for required in (
        "## Inputs and output",
        "## Capacity accounting",
        "## Fixed placement",
        "## Causal reuse dependencies",
        "## Offset vocabulary",
        "## Runtime adoption and validation",
        "FixedPhysicalLayout",
        "TaskAllocationContract",
        "AdmissionFacts",
    ):
        assert required in admission


def test_diagnostic_and_serialization_guides_cover_runtime_schemas() -> None:
    plan_report = (DOCS / "python" / "plan-report.md").read_text()
    for required in (
        "## Tasks are keyed by execution ID",
        "## Unique stages and graph pairs",
        "## Interpreting a graph profile",
        "## PressureFit diagnostics",
        "## Physical-layout diagnostics",
        "shadowspill.plan_diagnostics/v1",
        "chosen_graph_pair_variant",
    ):
        assert required in plan_report

    step_diagnostics = (DOCS / "python" / "step-diagnostics.md").read_text()
    for required in (
        "## Start with the summary",
        "## The seven task-boundary timestamps",
        "## Host boundary breakdown",
        "## Task-by-task simulator comparison",
        "## Transfer diagnostics",
        "## Allocator diagnostics",
        "## Runtime counters and trace integrity",
        "shadowspill.step_diagnostics/v4",
    ):
        assert required in step_diagnostics

    serialization = (DOCS / "python" / "planning-json.md").read_text()
    for required in (
        "## Program format",
        "## PressureFitProgram format",
        "## StepProgram format",
        "## AnnotatedProgramPlan format",
        "## Loading and validation",
        "shadowspill.program/v3",
        "shadowspill.pressurefit_program/v1",
        "shadowspill.step_program/v1",
        "shadowspill.annotated_program_plan/v2",
        "shadowspill.fixed_physical_layout/v3",
    ):
        assert required in serialization
    for block in _JSON_FENCE.findall(serialization):
        assert isinstance(json.loads(block), dict)


def test_current_contract_docs_avoid_historical_version_language() -> None:
    forbidden = re.compile(
        r"\b(?:initial release|initial provider|legacy|previous version|"
        r"former implementation|ShadowSpill v[0-9]+)\b",
        re.IGNORECASE,
    )
    violations = {
        path.relative_to(ROOT).as_posix(): sorted(
            set(forbidden.findall(path.read_text()))
        )
        for path in _normative_markdown_files()
    }
    violations = {path: values for path, values in violations.items() if values}
    assert not violations, f"historical language in current-contract docs: {violations}"


def test_current_contract_docs_are_backend_and_topology_neutral() -> None:
    forbidden = re.compile(
        r"\b(?:CUDA|ROCm)\b|"
        r"\b(?:one|single) (?:GPU|execution(?:-device)? pool|execution device)\b|"
        r"\b(?:cross-step )?cyclic residency\b",
        re.IGNORECASE,
    )
    violations = {
        path.relative_to(ROOT).as_posix(): sorted(
            set(forbidden.findall(path.read_text()))
        )
        for path in _normative_markdown_files()
    }
    violations = {path: values for path, values in violations.items() if values}
    assert not violations, (
        f"provider or facts language in current-contract docs: {violations}"
    )


def test_investigations_are_marked_non_normative() -> None:
    reports = tuple(
        path
        for path in (DOCS / "investigations").glob("*.md")
        if path.name != "README.md"
    )
    assert reports
    for report in reports:
        assert "Historical, non-normative investigation" in report.read_text()


def test_public_headers_avoid_historical_compatibility_labels() -> None:
    violations = [
        path.relative_to(ROOT).as_posix()
        for path in _PUBLIC_C_REFERENCES
        if re.search(r"\blegacy\b", path.read_text(), re.IGNORECASE)
    ]
    assert not violations, f"historical terminology in public C headers: {violations}"


def test_superseded_public_documentation_is_removed() -> None:
    for name in (
        "architecture.md",
        "ir.md",
        "lowering_contract.md",
        "memory-budget-semantics.md",
        "planner.md",
        "artifact-store.md",
        "pytorch-allocator.md",
        "pytorch-frontend.md",
        "runtime.md",
        "simulator.md",
    ):
        assert not (DOCS / name).exists()


def test_every_timing_field_is_described_where_timings_are_explained() -> None:
    """A timing field nobody can interpret is a timing field nobody can use.

    These are the numbers an investigation reads, and several of them are
    only meaningful once you know which origin they count from and whether
    they are an instant or a span. Adding one without saying which is how the
    next reader draws the wrong conclusion from it.
    """

    source = (
        ROOT / "src/shadowspill/pytorch/diagnostics/execution.py"
    ).read_text()
    guide = (ROOT / "docs/python/step-diagnostics.md").read_text()
    tree = ast.parse(source)
    undocumented: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name not in {"TaskExecutionTiming", "StepTimingSummary"}:
            continue
        missing = [
            item.target.id
            for item in node.body
            if isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and item.target.id.endswith(("_seconds", "_ns"))
            and item.target.id not in guide
        ]
        if missing:
            undocumented[node.name] = missing
    assert not undocumented, (
        f"timing fields absent from step-diagnostics.md: {undocumented}"
    )

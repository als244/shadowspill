"""Keep public documentation aligned with source-level API boundaries."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

from shadowspill.schema import artifact_schema
from tests.repository.files import repository_files

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "packages/engine"
DOCS = ROOT / "docs"
# The engine reference: the public pages and the internals pages.
ENGINE_REFERENCE = (DOCS / "reference" / "engine", DOCS / "reference" / "internals")

_MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
_PYTHON_FENCE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
_JSON_FENCE = re.compile(r"```json\n(.*?)\n```", re.DOTALL)
_C_FUNCTION = re.compile(r"\b(shadowspill_[A-Za-z0-9_]+)\s*\(")
_REFERENCE_ENTRY = re.compile(
    r"^\.\. auto(?:class|exception|function|data|type):: (\S+)$", re.MULTILINE
)

_MEMORY_MODULE = ENGINE / "src" / "shadowspill" / "memory.py"
_PYTORCH_MODULE = ENGINE / "src" / "shadowspill" / "pytorch" / "__init__.py"
_REPORTING_MODULE = ENGINE / "src" / "shadowspill" / "reporting" / "__init__.py"
_CORE_PYTHON_MODULES = (
    ENGINE / "src" / "shadowspill" / "errors.py",
    ENGINE / "src" / "shadowspill" / "ir" / "__init__.py",
    ENGINE / "src" / "shadowspill" / "planning" / "memory_search" / "__init__.py",
    ENGINE / "src" / "shadowspill" / "simulator" / "__init__.py",
    ENGINE / "src" / "shadowspill" / "runtime" / "__init__.py",
)
_PUBLIC_PYTHON_MODULES = (
    _MEMORY_MODULE,
    *_CORE_PYTHON_MODULES,
    _PYTORCH_MODULE,
    _REPORTING_MODULE,
)

_PUBLIC_HEADERS = ENGINE / "csrc" / "include" / "shadowspill"

_PUBLIC_C_REFERENCES = {
    _PUBLIC_HEADERS / "runtime" / "vocabulary.h": DOCS / "c" / "runtime.md",
    _PUBLIC_HEADERS / "runtime" / "descriptions.h": DOCS / "c" / "runtime.md",
    _PUBLIC_HEADERS / "runtime" / "diagnostics.h": DOCS / "c" / "runtime.md",
    _PUBLIC_HEADERS / "runtime" / "lifecycle.h": DOCS / "c" / "runtime.md",
    _PUBLIC_HEADERS / "runtime" / "pools.h": DOCS / "c" / "runtime.md",
    _PUBLIC_HEADERS / "runtime" / "objects.h": DOCS / "c" / "runtime.md",
    _PUBLIC_HEADERS / "runtime" / "plan.h": DOCS / "c" / "runtime.md",
    _PUBLIC_HEADERS / "runtime" / "tasks.h": DOCS / "c" / "runtime.md",
    _PUBLIC_HEADERS / "runtime" / "telemetry.h": DOCS / "c" / "runtime.md",
    _PUBLIC_HEADERS / "admission_replay.h": DOCS / "c" / "runtime.md",
    _PUBLIC_HEADERS / "backend.h": DOCS / "c" / "backends.md",
    _PUBLIC_HEADERS / "planner.h": DOCS / "c" / "planner.md",
    _PUBLIC_HEADERS / "simulator.h": DOCS / "c" / "simulator.md",
    ENGINE
    / "csrc"
    / "adapter"
    / "pytorch"
    / "include"
    / "shadowspill"
    / "pytorch_adapter.h": DOCS / "c" / "pytorch-adapter.md",
}


def _markdown_files() -> tuple[Path, ...]:
    return tuple(
        path
        for path in repository_files("*.md")
        if not path.is_relative_to(DOCS / "internal")
    )


def _normative_markdown_files() -> tuple[Path, ...]:
    return tuple(
        path for path in _markdown_files() if not path.is_relative_to(DOCS / "internal")
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
    parts = path.relative_to(ROOT / "packages/engine/src").with_suffix("").parts
    module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
    value = importlib.import_module(module).__all__
    assert isinstance(value, list) and all(isinstance(name, str) for name in value)
    assert len(value) == len(set(value)), f"{module}: duplicate public export"
    return tuple(value)


def _documented_symbol(reference: str, name: str) -> bool:
    return re.search(rf"`{re.escape(name)}(?:`|\()", reference) is not None


def _reference_objects() -> set[int]:
    """The identities of the objects the engine reference has an entry for."""
    found: set[int] = set()
    for page in (page for root in ENGINE_REFERENCE for page in root.glob("*.rst")):
        for path in _REFERENCE_ENTRY.findall(page.read_text()):
            module, _, name = path.rpartition(".")
            found.add(id(getattr(importlib.import_module(module), name)))
    return found


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
        "architecture/README.md",
        "python/README.md",
        "python/client.md",
        "python/advanced/README.md",
        "examples/README.md",
        "examples/client-training.md",
        "c/README.md",
        "development/README.md",
        "integrations/README.md",
    ):
        assert f"]({target})" in index


def test_root_readme_remains_a_minimal_entrypoint() -> None:
    readme = (ROOT / "README.md").read_text()
    # Installation plus the three requested, realistic training examples.
    assert len(readme.splitlines()) <= 200
    assert "](docs/README.md)" in readme
    assert "](docs/examples/client-training.md#prepare-ultrachat)" in readme
    assert "](docs/python/client.md)" in readme
    assert "](docs/python/advanced/README.md)" in readme
    assert "](docs/development/repository.md#setup)" in readme


def test_architecture_overview_describes_public_phase_boundaries() -> None:
    overview = (DOCS / "architecture" / "overview.md").read_text()
    for required in (
        "build_forward",
        "build_forward_backward",
        "build_step",
        "measure_program",
        "apply_profiles",
        "measure_communication",
        "plan_program",
        "sweep_programs",
        "Executor",
        "simulation-only",
        "## Source organization",
    ):
        assert required in overview
    assert "```mermaid" in overview


def test_public_manuals_are_reachable_from_the_documentation_index() -> None:
    """Allow focused sub-indexes without leaving newly added manuals orphaned."""
    manuals = {
        path.resolve() for path in _markdown_files() if path.is_relative_to(DOCS)
    }
    pending = [DOCS / "README.md"]
    reached = set()
    while pending:
        page = pending.pop()
        if page in reached:
            continue
        reached.add(page)
        for raw in _MARKDOWN_LINK.findall(page.read_text()):
            target, _ = _local_link(page, raw)
            if target in manuals and target not in reached:
                pending.append(target)
    assert not manuals - reached, sorted(
        str(path.relative_to(ROOT)) for path in manuals - reached
    )


def test_examples_cover_complete_public_workflows() -> None:
    examples = DOCS / "examples"
    expected = {
        "forward-only.md": (
            "build_forward(",
            "measure_program(",
            "plan_program(",
            "Executor(Plan.load(",
            "executor.initialize(",
            "pending.result()",
        ),
        "reusable-planning.md": (
            "ProfiledProgram.load(",
            "plan_program(",
            "sweep_programs(",
            "plan.save(",
            "write_report(",
        ),
        "diagnostics.md": (
            "pending.diagnostics",
            "step.transfers.fetch",
        ),
        "custom-partitioning.md": (
            "assign_stages(",
            "build_step(",
            "partition=EveryNNodes(",
        ),
    }
    for name, required in expected.items():
        reference = (examples / name).read_text()
        for value in required:
            assert value in reference
        for source in _PYTHON_FENCE.findall(reference):
            tree = ast.parse(source)
            assert not any(isinstance(node, ast.Try) for node in ast.walk(tree))


def test_failure_guide_covers_public_boundaries_and_cleanup() -> None:
    reference = (DOCS / "python" / "advanced" / "failures.md").read_text()
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


def test_every_public_python_export_appears_in_the_engine_reference() -> None:
    documented = _reference_objects()
    undocumented = []
    for path in _PUBLIC_PYTHON_MODULES:
        parts = path.relative_to(ENGINE / "src").with_suffix("").parts
        module = importlib.import_module(
            ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        )
        undocumented.extend(
            f"{module.__name__}.{name}"
            for name in _all_exports(path)
            if not inspect.ismodule(value := getattr(module, name))
            and id(value) not in documented
        )
    assert not undocumented, f"exports absent from the engine reference: {undocumented}"


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


def test_normative_python_examples_compile_and_use_valid_public_keywords() -> None:
    failures: list[str] = []
    for document in _normative_markdown_files():
        parameters: dict[str, set[str]] = {}
        modules = {}
        for index, source in enumerate(_PYTHON_FENCE.findall(document.read_text())):
            try:
                compile(source, f"{document}#python-{index}", "exec")
            except SyntaxError as error:
                failures.append(f"{document.relative_to(ROOT)} block {index}: {error}")
                continue
            tree = ast.parse(source)
            # Identical short names can belong to different public modules.
            # Follow the example's imports instead of imposing another API's signature.
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "shadowspill" or alias.name.startswith(
                            "shadowspill."
                        ):
                            name = alias.asname or alias.name.split(".")[0]
                            modules[name] = importlib.import_module(
                                alias.name if alias.asname else name
                            )
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "shadowspill"
                ):
                    module = importlib.import_module(node.module)
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        if not hasattr(module, alias.name):
                            # Python's from-import can also name a submodule
                            # that the package has not imported yet.
                            try:
                                importlib.import_module(f"{node.module}.{alias.name}")
                            except ModuleNotFoundError:
                                failures.append(
                                    f"{document.relative_to(ROOT)} block {index}: "
                                    f"{node.module} has no import {alias.name}"
                                )
                                continue
                        value = getattr(module, alias.name, None)
                        if inspect.isfunction(value) or inspect.isclass(value):
                            try:
                                signature = inspect.signature(value)
                            except (TypeError, ValueError):
                                continue
                            if any(
                                p.kind == p.VAR_KEYWORD
                                for p in signature.parameters.values()
                            ):
                                parameters.pop(alias.asname or alias.name, None)
                            else:
                                parameters[alias.asname or alias.name] = set(
                                    signature.parameters
                                )
            for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
                display = ast.unparse(call.func)
                allowed = None
                if isinstance(call.func, ast.Name):
                    allowed = parameters.get(call.func.id)
                elif isinstance(call.func, ast.Attribute):
                    parts = []
                    root = call.func
                    while isinstance(root, ast.Attribute):
                        parts.append(root.attr)
                        root = root.value
                    # Returned objects and local instances need runtime type
                    # information; only resolve statically imported API paths.
                    if not isinstance(root, ast.Name):
                        continue
                    value = modules.get(root.id)
                    if value is None:
                        continue
                    try:
                        for part in reversed(parts):
                            value = getattr(value, part)
                    except AttributeError:
                        failures.append(
                            f"{document.relative_to(ROOT)} block {index}: "
                            f"unknown public callable {display}"
                        )
                        continue
                    try:
                        signature = inspect.signature(value)
                    except (TypeError, ValueError):
                        continue
                    if not any(
                        p.kind == p.VAR_KEYWORD for p in signature.parameters.values()
                    ):
                        allowed = set(signature.parameters)
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
                        f"{display} has unknown keywords {unexpected}"
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
        "SearchOptions",
        "ProgramPlanResult",
        "AdmissionFacts",
        "PlanInfeasibleError",
        "PlanSearchExhaustedError",
    ):
        assert required in reference

    recomputation = (DOCS / "architecture" / "graph-pair-selection.md").read_text()
    for required in (
        "## Inputs and output",
        "## The current selection policy",
        "## Pseudocode",
        "TaskAlternativeGroup",
        "TaskAlternativeOption",
        "TaskAlternativeChoice",
    ):
        assert required in recomputation

    graph_pairs = (DOCS / "architecture" / "graph-pair-construction.md").read_text()
    for required in (
        "## Inputs and output",
        "## Saved-value accounting",
        "## Compilation and profiling",
        "## Lowering into program alternatives",
        "TaskGraphPairs",
        "GraphPairVariant",
        "TaskAlternativeOption",
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
    plan_report = (DOCS / "python" / "advanced" / "plan-report.md").read_text()
    for required in (
        "## Structure",
        "## Transfer assumptions and budgets",
        "## Unique stages and graph pairs",
        "## Search diagnostics",
        "## Physical-layout diagnostics",
        "## Execution comparison",
        "simulation_only_reasons",
    ):
        assert required in plan_report

    step_diagnostics = (
        DOCS / "python" / "advanced" / "step-diagnostics.md"
    ).read_text()
    for required in (
        "## Structure",
        "## Summary",
        "## Tasks",
        "## Transfers",
        "## Timelines",
        "## Allocator",
        "## Runtime",
        artifact_schema("step_diagnostics"),
    ):
        assert required in step_diagnostics

    serialization = (DOCS / "python" / "advanced" / "planning-json.md").read_text()
    for required in (
        "## Program format",
        "## ShadowSpillPlanningProblem format",
        "## AnnotatedProgramPlan format",
        "## Loading and validation",
        artifact_schema("program"),
        artifact_schema("plan_program"),
        artifact_schema("annotated_program_plan"),
        artifact_schema("fixed_physical_layout"),
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
    # Concrete workloads, their integration tests and baselines describe their
    # hardware requirements. Neutral engine contracts cannot fix a backend/topology.
    violations = {
        path.relative_to(ROOT).as_posix(): sorted(
            set(forbidden.findall(path.read_text()))
        )
        for path in _normative_markdown_files()
        if not path.is_relative_to(
            ROOT / "packages/workloads/src/shadowspill_workloads"
        )
        and not path.is_relative_to(ROOT / "packages/workloads/tests/workload_tests")
        and not path.is_relative_to(DOCS / "integrations")
        and path
        not in {
            ROOT / "README.md",  # Concrete server installation prerequisites.
            DOCS / "python/advanced/saved-plans.md",  # Concrete GPU remapping.
            ROOT / "qualification/reference/training/README.md",
            ROOT / "tools/stage_traces/README.md",
            ROOT / "tools/planning/corpus/README.md",
            ROOT / "tools/measure_plan/README.md",
            ROOT / "packages/engine/src/shadowspill/orchestration/README.md",
        }
    }
    violations = {path: values for path, values in violations.items() if values}
    assert not violations, (
        f"provider or facts language in current-contract docs: {violations}"
    )


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


def test_every_step_diagnostics_field_is_described_in_its_guide() -> None:
    """A field nobody can interpret is a field nobody can use.

    These are the numbers an investigation reads, and several of them are
    only meaningful once you know which origin they count from and whether
    they are an instant or a span. Adding one without saying which is how the
    next reader draws the wrong conclusion from it, so the guide has to name
    every field of every record, not only the timings.
    """

    source = (ROOT / "packages/engine/src/shadowspill/diagnostics/step.py").read_text()
    guide = (ROOT / "docs/python/advanced/step-diagnostics.md").read_text()
    undocumented: dict[str, list[str]] = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef):
            continue
        missing = [
            item.target.id
            for item in node.body
            if isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and f"`{item.target.id}`" not in guide
        ]
        if missing:
            undocumented[node.name] = missing
    assert not undocumented, f"fields absent from step-diagnostics.md: {undocumented}"


def test_step_diagnostics_names_instants_and_durations_apart() -> None:
    """One suffix per kind, so a name says what kind of number it holds.

    An instant ends in `_at_seconds` and a duration ends in `_seconds`.
    Summing instants is meaningless, and the two are indistinguishable
    without the convention, so a field that reads as a moment -- started,
    finished, reached, queued -- has to carry the instant suffix.
    """

    source = (ROOT / "packages/engine/src/shadowspill/diagnostics/step.py").read_text()
    moments = (
        "started",
        "finished",
        "reached",
        "entered",
        "exited",
        "queued",
        "reserved",
        "dispatched",
        "observed",
        "ready",
    )
    offenders: dict[str, list[str]] = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef):
            continue
        wrong = [
            item.target.id
            for item in node.body
            if isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and item.target.id.endswith(("_seconds", "_ns"))
            and not item.target.id.endswith(("_at_seconds", "_at_ns"))
            and any(word in item.target.id.split("_") for word in moments)
        ]
        if wrong:
            offenders[node.name] = wrong
    assert not offenders, (
        f"instants named as durations: {offenders}; end an instant in _at_seconds"
    )


def test_search_diagnostic_fields_are_described_in_the_reference() -> None:
    """Document the inspectable search records used by the current Plan."""
    modules = (
        "packages/engine/src/shadowspill/planning/memory_search/diagnostics/plan_summary.py",
        "packages/engine/src/shadowspill/planning/memory_search/diagnostics/summary.py",
        "packages/engine/src/shadowspill/planning/memory_search/diagnostics/resolved_programs.py",
        "packages/engine/src/shadowspill/planning/memory_search/diagnostics/candidates.py",
        "packages/engine/src/shadowspill/planning/memory_search/diagnostics/counters.py",
    )
    reference = (ROOT / "docs/python/advanced/plan-report-fields.md").read_text()
    undocumented: dict[str, list[str]] = {}
    for module in modules:
        for node in ast.walk(ast.parse((ROOT / module).read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            missing = [
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
                and not item.target.id.isupper()
                and f"`{item.target.id}`" not in reference
            ]
            if missing:
                undocumented[node.name] = missing
    assert not undocumented, f"fields absent from plan-report-fields.md: {undocumented}"


#: The directory drawings in the docs, and the directory each is rooted at. A
#: drawing that enumerates packages is also checked for completeness; the
#: others say in their own text that they stop at one level.
_DOCUMENTED_TREES = (
    ("docs/development/repository.md", "shadowspill/", ".", False),
    (
        "docs/development/repository.md",
        "packages/engine/src/shadowspill/",
        "packages/engine/src/shadowspill",
        True,
    ),
    ("docs/development/repository.md", "tests/", "tests", False),
    ("packages/engine/csrc/README.md", "csrc/", "packages/engine/csrc", False),
    ("qualification/README.md", "qualification/", "qualification", False),
)

_TREE_BLOCK = re.compile(r"```text\n(.*?)```", re.S)
_TREE_ENTRY = re.compile(r"^((?:(?:│   |    )*)(?:├── |└── ))(\S+)")


def _drawn_paths(block: str, root: Path) -> list[tuple[Path, bool]]:
    """Every path one drawing names, and whether it is drawn as a directory."""

    drawn: list[tuple[Path, bool]] = []
    parents = [root]
    for line in block.splitlines()[1:]:
        entry = _TREE_ENTRY.match(line)
        if entry is None:
            continue
        # the branch marker is four characters wide and is not a level
        depth = (len(entry.group(1)) - 4) // 4
        name = entry.group(2)
        parent = parents[depth] if depth < len(parents) else parents[-1]
        path = parent / name.rstrip("/")
        del parents[depth + 1 :]
        parents.append(path)
        drawn.append((path, name.endswith("/")))
    return drawn


def test_documented_directory_trees_match_the_repository() -> None:
    packages = {parent for file in repository_files("*.py") for parent in file.parents}
    absent: dict[str, list[str]] = {}
    undrawn: dict[str, list[str]] = {}
    for document, heading, root, exhaustive in _DOCUMENTED_TREES:
        for block in _TREE_BLOCK.findall((ROOT / document).read_text()):
            if block.splitlines()[0].strip() != heading:
                continue
            drawn = _drawn_paths(block, ROOT / root)
            where = f"{document} :: {heading}"
            missing = [
                path.relative_to(ROOT).as_posix()
                for path, _directory in drawn
                if not path.exists()
            ]
            if missing:
                absent[where] = missing
            if not exhaustive:
                continue
            named = {path for path, _directory in drawn}
            for path, directory in [(ROOT / root, True), *drawn]:
                if not directory or not path.is_dir():
                    continue
                for child in sorted(path.iterdir()):
                    if child not in packages or child.name.startswith((".", "_")):
                        continue
                    if child not in named:
                        undrawn.setdefault(where, []).append(
                            child.relative_to(ROOT).as_posix()
                        )
    assert not absent, f"documented paths that do not exist: {absent}"
    assert not undrawn, f"packages the drawing omits: {undrawn}"

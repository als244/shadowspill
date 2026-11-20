"""Sphinx configuration for the ShadowSpill reference.

The reference is generated from the public docstrings, written in NumPy style.
Build it with ``sphinx-build -W -b html docs docs/_build/html``.
"""

import ast
import dataclasses
import enum
import inspect
import re
import sys
import textwrap
from pathlib import Path

from numpydoc.validate import Validator
from sphinx.pycode.ast import unparse
from sphinx.util.inspect import DefaultValue, signature_from_str, stringify_signature
from sphinx.util.nodes import make_refnode

ROOT = Path(__file__).resolve().parents[1]
for package in ("client", "engine"):
    sys.path.insert(0, str(ROOT / "packages" / package / "src"))

project = "ShadowSpill"
author = "Andrew Sheinberg"
root_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_parser",
]

# Only the reference is built for now; the Markdown guides join it later.
exclude_patterns = [
    "_build",
    "_templates",
    "internal",
    "architecture",
    "c",
    "development",
    "examples",
    "integrations",
    "python",
    "README.md",
]

# Signatures show names and defaults; each docstring states the types.
autodoc_typehints = "none"
autodoc_member_order = "bysource"
autodoc_preserve_defaults = True
# Every `name` reference must resolve; the build needs no network.
nitpicky = True
autodoc_class_signature = "mixed"
autodoc_default_options = {"members": True}
add_module_names = False
# `name` in a docstring links to that documented object.
default_role = "py:obj"

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = False
# Every public object states its purpose, parameters and return value.
numpydoc_validation_checks = {
    "all",
    "ES01",  # an extended summary is optional
    "EX01",  # examples are optional
    "SA01",  # "See Also" is optional
    "RT02",  # a named return value is allowed
    "GL01",  # the summary may start on the first line
}
#: Classes users never construct, by package: no constructor signature is
#: shown, and their docstrings describe the object, not its internal
#: constructor arguments.
RETURNED_ONLY = {
    "shadowspill_client": (
        "TrainingClient",
        "RequestQueries",
        "CallFuture",
        "OperationFuture",
        "TrainingFuture",
        "PreparationFuture",
        "PlanningFuture",
        "PreparationHandle",
        "ArtifactTransfers",
        "ArtifactHandle",
        "GenerationHandle",
        "CheckpointHandle",
        "WeightsHandle",
        "PlansHandle",
        "ReportHandle",
        "MetricQueries",
        "TraceHandle",
        "NsightCapture",
        "TrainingResult",
    ),
    "shadowspill": (
        "RuntimeFrontend",
        "ChunkPolicy",
        "PartitionPolicy",
        "InvocationResult",
        "StageInfo",
        "WorkerScope",
        "WorkerStartup",
        "AdmissionReplayDecision",
        "AdmissionReplayResult",
        "AdmissionReuseDependency",
        "MemoryPool",
        "RuntimeRoute",
        "TransferCapabilities",
        "TransferProfile",
        "RuntimeFailureDiagnostics",
        "ObjectRef",
        "IndexedProgram",
        "IndexedMemorySchedule",
        "IndexedExecutionPlan",
        "SharedResidencyFootprint",
        "ArtifactStore",
        "PlanningArtifact",
        "ArtifactRecorder",
        "AnnotatedProgramPlan",
        "PlanSummary",
        "GraphPairOutcome",
        "CostedAlternatives",
        "CostedGroup",
        "CostedOption",
        "TaskInterval",
        "TransferInterval",
        "DeviceMemoryPeak",
        "MemorySnapshot",
        "CapacityViolation",
        "SimulationResult",
        "Program",
        "ProfiledProgram",
        "Plan",
        "PoolMemoryProfile",
        "SweepResult",
        "SweepCase",
        "TensorChunk",
        "TrainingResult",
        "CachedTask",
        "TimingWindow",
        "PlanningInputs",
        "LocalPlans",
        "PlanningRequest",
        "TemplateUse",
        "Artifact",
        "PlanningSweep",
        "SweepRow",
        "LifetimeOperation",
        "TensorGeometry",
        "GradientDestination",
        "GradientDestinationContext",
        "FixedPhysicalLayout",
        "LeaseLifetime",
        "FixedLayoutPlacements",
        "FixedLayoutPlacement",
        "FixedLayoutReuses",
        "FixedLayoutReuse",
        "FixedLayoutTaskAllocations",
        "PoolAllocation",
        "StepDiagnostics",
        "StepTimingSummary",
        "Timelines",
    ),
}
# Private members, dunders and the returned-only classes themselves (their
# members are still checked); `_record` adds constants and enums.
numpydoc_validation_exclude = {
    r"\._[A-Za-z0-9_]*$",
    r"\.__[a-z_]+__$",
    *(
        rf"^{package}\.(?:.*\.)?{name}$"
        for package, names in RETURNED_ONLY.items()
        for name in names
    ),
}
# A private argument is neither shown nor documented, so validation does not
# ask for it.
_signature_parameters = Validator.signature_parameters.fget
Validator.signature_parameters = property(
    lambda self: tuple(p for p in _signature_parameters(self) if p[0] != "_")
)

html_theme = "furo"
html_title = "ShadowSpill"
# No theme credit or empty copyright line at the foot of each page.
html_show_sphinx = False
html_show_copyright = False
templates_path = ["_templates"]
html_static_path: list[str] = []
html_theme_options = {
    "source_repository": "https://github.com/als244/shadowspill",
    "source_branch": "master",
    "source_directory": "docs/",
}


def _skip_dataclass_fields(app, what, name, obj, skip, options):
    """Hide a dataclass field's own entry: its Parameters entry documents it."""
    owner = getattr(app.env, "temp_data", {}).get("autodoc:class")
    return True if name in _fields.get(owner, ()) else None


def _is_enum(obj):
    return isinstance(obj, type) and issubclass(obj, enum.Enum)


def _readable_signature(app, what, name, obj, options, signature, annotation):
    """Show a signature as its source writes it, without private arguments.

    An argument named with a leading ``_`` is left out. A dataclass field's
    default is its expression in the class body (``256 << 20``, not
    ``268435456``), as ``autodoc_preserve_defaults`` gives a function's; a
    default with no readable value (a private sentinel, a factory other than
    an empty container's) shows as ``...``. A returned-only class shows no
    constructor signature, and nor does an enum: its one argument is a
    member's value, and its docstring lists the members under Attributes.
    """
    package, short = name.split(".", 1)[0], name.rsplit(".", 1)[-1]
    returned_only = short in RETURNED_ONLY.get(package, ())
    if what in ("class", "exception") and (returned_only or _is_enum(obj)):
        return "", annotation
    if not signature:
        return signature, annotation
    written = _written_defaults(obj) if what in ("class", "exception") else {}
    parsed = signature_from_str(re.sub(r"=<[^<>]*>", "=...", signature))
    parameters = [
        p.replace(default=DefaultValue(written[p.name])) if p.name in written else p
        for p in parsed.parameters.values()
        if not p.name.startswith("_")
    ]
    signature = stringify_signature(
        parsed.replace(parameters=parameters),
        show_annotation=False,
        show_return_annotation=False,
    )
    return re.sub(r"=_[A-Za-z0-9_]+\b", "=...", signature), annotation


def _written_defaults(cls):
    """Each dataclass field's default as the class body writes it, by name.

    A field declared with ``field(default=...)`` gives that argument, and one
    whose factory is an empty builtin container (``dict``) gives its value
    (``{}``). A field without a default, with another factory, or whose
    default names a private object has none here.
    """
    if not dataclasses.is_dataclass(cls):
        return {}
    fields = [f for f in dataclasses.fields(cls) if f.init]
    empty = {
        f.name: repr(f.default_factory())
        for f in fields
        if f.default_factory in (dict, list, tuple, set, frozenset)
    }
    defaults = {f.name for f in fields if f.default is not dataclasses.MISSING}
    written = {}
    # A field a subclass declares again replaces its base's.
    for owner in reversed(cls.__mro__):
        if "__dataclass_fields__" not in vars(owner):
            continue
        try:
            source = textwrap.dedent(inspect.getsource(owner))
        except (OSError, TypeError):
            continue
        namespace = vars(sys.modules[owner.__module__])
        for node in ast.parse(source).body[0].body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                value = node.value
                if _is_field_call(value, namespace):
                    keywords = {k.arg: k.value for k in value.keywords}
                    value = keywords.get("default")
                private = value is None or any(
                    isinstance(n, ast.Name) and n.id.startswith("_")
                    for n in ast.walk(value)
                )
                written[node.target.id] = None if private else unparse(value, source)
    return empty | {
        name: text for name, text in written.items() if text and name in defaults
    }


def _is_field_call(node, namespace):
    func = getattr(node, "func", None)
    if isinstance(func, ast.Attribute):
        return func.attr == "field"
    return isinstance(func, ast.Name) and namespace.get(func.id) is dataclasses.field


def _enum_values(app, what, name, obj, options, lines):
    """Show each enum member its docstring lists with its value.

    numpydoc renders a member as ``**NAME** : type``; this makes it
    ``**NAME** = "value"``.
    """
    if not _is_enum(obj):
        return
    members = obj.__members__
    for index, line in enumerate(lines):
        match = re.fullmatch(r"(\s*)\*\*(\w+)\*\* : .*", line)
        if match and match[2] in members:
            value = members[match[2]].value
            literal = f'"{value}"' if isinstance(value, str) else repr(value)
            lines[index] = f"{match[1]}**{match[2]}** = ``{literal}``"


def _no_data_signature(app, what, name, obj, options, signature, annotation):
    """Keep a constant or type alias without a signature.

    numpydoc would give a callable value (an alias, an algorithm instance)
    one; an empty result ends the event without changing anything.
    """
    return () if what in ("data", "type") else None


def _record(app, what, name, obj, options, lines):
    """Note a dataclass's fields, and keep constants and enums from validation.

    A constant's ``#:`` comment is free text, where numpydoc would check its
    value's type; an enum has no constructor arguments to document.
    """
    if what in ("class", "exception") and dataclasses.is_dataclass(obj):
        _fields[name.rsplit(".", 1)[-1]] = {f.name for f in dataclasses.fields(obj)}
    if what in ("data", "type", "attribute") or _is_enum(obj):
        names = app.config.numpydoc_validation_excluder
        exact = rf"^{re.escape(name)}$"
        app.config.numpydoc_validation_excluder = re.compile(
            f"{names.pattern}|{exact}" if names else exact
        )


_fields: dict[str, set[str]] = {}


def _unique(objects, target, package=None):
    # An object documented under its public path is also registered under its
    # defining module's; both name the same entry. A name found in several
    # packages means the one in the referring object's own package.
    matches = {
        (objects[n].docname, objects[n].node_id): n
        for n in objects
        if n == target or n.endswith("." + target)
    }
    if len(matches) > 1 and package:
        matches = {k: n for k, n in matches.items() if n.split(".", 1)[0] == package}
    return next(iter(matches.values())) if len(matches) == 1 else None


def _resolve_short_name(app, env, node, contnode):
    """Link a name to the one documented object it can mean.

    A name is tried as written and then without its leading module parts, so
    ``Name``, ``module.Name`` and a full path all find an object documented
    under its public import path. A dataclass field, which has no entry of its
    own, links to its class.
    """
    if node.get("refdomain") != "py":
        return None
    parts = node["reftarget"].lstrip("~.").split(".")
    objects = env.get_domain("py").objects
    package = (node.get("py:module") or "").split(".", 1)[0] or None
    found = None
    # A field is tried first: its bare name may be another object's member.
    owner = parts[-2] if len(parts) > 1 else node.get("py:class")
    if owner and parts[-1] in _fields.get(owner.split(".")[-1], ()):
        found = _unique(
            objects, ".".join(parts[:-1]) if len(parts) > 1 else owner, package
        )
    for start in range(len(parts)):
        if found:
            break
        found = _unique(objects, ".".join(parts[start:]), package)
    # An attribute a class describes in its docstring links to that class.
    if found is None and len(parts) > 1:
        found = _unique(objects, ".".join(parts[:-1]), package)
    if found is None:
        return None
    entry = objects[found]
    return make_refnode(
        app.builder, node["refdoc"], entry.docname, entry.node_id, contnode, found
    )


def setup(app):
    app.connect("missing-reference", _resolve_short_name)
    # Before numpydoc validates a docstring or reads a signature.
    app.connect("autodoc-process-docstring", _record, priority=400)
    app.connect("autodoc-process-signature", _no_data_signature, priority=400)
    app.connect("autodoc-skip-member", _skip_dataclass_fields)
    app.connect("autodoc-process-signature", _readable_signature)
    # After numpydoc renders the docstring.
    app.connect("autodoc-process-docstring", _enum_values, priority=600)

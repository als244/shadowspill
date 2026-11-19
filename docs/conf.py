"""Sphinx configuration for the ShadowSpill reference.

The reference is generated from the public docstrings, written in NumPy style.
Build it with ``sphinx-build -W -b html docs docs/_build/html``.
"""

import dataclasses
import re
import sys
from pathlib import Path

from sphinx.util.nodes import make_refnode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "client" / "src"))

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
#: Classes the client returns and users never construct: no constructor
#: signature is shown, and their docstrings describe the object, not its
#: internal constructor arguments.
RETURNED_ONLY = (
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
)
# Private members, dunders, module constants (documented by a `#:` comment),
# and the returned-only classes themselves (their members are still checked).
numpydoc_validation_exclude = {
    r"\._[A-Za-z0-9_]*$",
    r"\.__[a-z_]+__$",
    r"\.[A-Z][A-Z0-9]*_[A-Z0-9_]+$",
    *(rf"\.{name}$" for name in RETURNED_ONLY),
}

html_theme = "furo"
html_title = "ShadowSpill"
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


def _readable_defaults(app, what, name, obj, options, signature, annotation):
    """Show a default with no readable value (a factory, a private sentinel) as ....

    A returned-only class shows no constructor signature.
    """
    if what in ("class", "exception") and name.rsplit(".", 1)[-1] in RETURNED_ONLY:
        return "", annotation
    if signature:
        signature = re.sub(r"=(?:<[^<>]*>|_[A-Za-z0-9_]+\b)", "=...", signature)
    return signature, annotation


def _record_fields(app, what, name, obj, options, lines):
    if what in ("class", "exception") and dataclasses.is_dataclass(obj):
        _fields[name.rsplit(".", 1)[-1]] = {f.name for f in dataclasses.fields(obj)}


_fields: dict[str, set[str]] = {}


def _unique(objects, target):
    # An object documented under its public path is also registered under its
    # defining module's; both name the same entry.
    matches = {
        (objects[n].docname, objects[n].node_id): n
        for n in objects
        if n == target or n.endswith("." + target)
    }
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
    found = None
    for start in range(len(parts)):
        found = _unique(objects, ".".join(parts[start:]))
        if found:
            break
    owner = parts[-2] if len(parts) > 1 else node.get("py:class")
    if found is None and owner and parts[-1] in _fields.get(owner.split(".")[-1], ()):
        found = _unique(objects, ".".join(parts[:-1]) if len(parts) > 1 else owner)
    # An attribute a class describes in its docstring links to that class.
    if found is None and len(parts) > 1:
        found = _unique(objects, ".".join(parts[:-1]))
    if found is None:
        return None
    entry = objects[found]
    return make_refnode(
        app.builder, node["refdoc"], entry.docname, entry.node_id, contnode, found
    )


def setup(app):
    app.connect("missing-reference", _resolve_short_name)
    app.connect("autodoc-process-docstring", _record_fields)
    app.connect("autodoc-skip-member", _skip_dataclass_fields)
    app.connect("autodoc-process-signature", _readable_defaults)

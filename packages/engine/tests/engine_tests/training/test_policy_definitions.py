"""Policy preview resolves only installed CPU descriptions; build uses core objects."""

import subprocess
import sys
from dataclasses import dataclass, replace

import pytest

from engine_tests.fixtures.external_definition import SOURCE
from engine_tests.training.test_preparation_resolution import preset, stacked_context
from shadowspill.training.definitions import PolicyDefinition, Registry
from shadowspill.training.preparation.configuration.policies import (
    materialize_build_options,
    resolve_build_policies,
)
from shadowspill.training.preparation.configuration.resolution import (
    resolve_preparation,
)
from shadowspill_client import BuildCandidates, BuildOptions, Policy, Preparation


@dataclass(frozen=True)
class RetentionConfig:
    save: str = "cost"
    recompute: str = "recompute"

    def __post_init__(self):
        if self.save not in ("cost", "retain", "recompute") or self.recompute not in (
            "cost",
            "retain",
            "recompute",
        ):
            raise ValueError("unknown retention choice")


def create_policy(config):
    from shadowspill.pytorch.capture.retention import RetentionPolicy
    from shadowspill.pytorch.graph_pairs.policy import GraphPairPolicy

    return GraphPairPolicy(
        save=RetentionPolicy(default=config.save),
        recompute=RetentionPolicy(default=config.recompute),
    )


def definition(create=create_policy):
    return PolicyDefinition(
        "external/retention@1", "graph_pair_policy", RetentionConfig, create
    )


def settings():
    return Preparation(
        build=BuildOptions(
            partition=Policy("shadowspill/partition@1", {"mode": "whole"})
        ),
        candidates=BuildCandidates(
            microbatch_sizes=[1],
            graph_pair_policies=[None, Policy("external/retention@1")],
        ),
    )


def test_policy_resolution_pins_source_defaults_and_never_materializes():
    registry = Registry()
    calls = []

    def forbidden(config):
        calls.append(config)
        pytest.fail("constructed a policy during inspection")

    registry.add_policy(definition(forbidden), source=SOURCE)
    resolved = resolve_preparation(preset(settings()), stacked_context())
    bound, sources = resolve_build_policies(registry, resolved)
    assert not calls
    assert bound.cases[0].build.values["graph_pair_policy"] is None
    assert bound.cases[1].build.values["graph_pair_policy"]["config"] == {
        "save": "cost",
        "recompute": "recompute",
    }
    assert sources["external/retention@1"] == SOURCE
    assert resolve_build_policies(registry, bound, sources=sources) == (bound, sources)
    with pytest.raises(ValueError, match="defaults changed"):
        resolve_build_policies(registry, resolved, sources=sources)
    with pytest.raises(ValueError, match="missing policy source"):
        resolve_build_policies(registry, bound, sources={})
    with pytest.raises(ValueError, match="unreferenced"):
        resolve_build_policies(registry, bound, sources={**sources, "unused": SOURCE})
    with pytest.raises(ValueError, match="source changed"):
        resolve_build_policies(
            registry,
            bound,
            sources={
                **sources,
                "external/retention@1": replace(SOURCE, digest="b" * 64),
            },
        )


def test_materialization_returns_the_ordinary_existing_phase_policies():
    registry = Registry()
    registry.add_policy(definition(), source=SOURCE)
    resolved = resolve_preparation(preset(settings()), stacked_context())
    bound, sources = resolve_build_policies(registry, resolved)
    first = materialize_build_options(registry, bound.cases[0].build, sources)
    assert first == {"partition": "whole", "chunking": None, "graph_pair_policy": None}
    second = materialize_build_options(registry, bound.cases[1].build, sources)
    expected = create_policy(RetentionConfig())
    assert second["graph_pair_policy"].identity() == expected.identity()
    assert second["chunking"] is None and second["partition"] == "whole"
    with pytest.raises(ValueError, match="missing policy source"):
        materialize_build_options(registry, bound.cases[0].build, {})
    with pytest.raises(ValueError, match="not fully resolved"):
        materialize_build_options(registry, resolved.cases[1].build, sources)


def test_unknown_policy_config_kind_or_request_module_names_fail_closed():
    registry = Registry()
    registry.add_policy(definition(), source=SOURCE)
    with pytest.raises(ValueError, match="not installed"):
        registry.resolve_policy("os:system", kind="partition")
    with pytest.raises(ValueError, match="cannot configure"):
        registry.resolve_policy("external/retention@1", kind="chunking")
    with pytest.raises(TypeError, match="unexpected keyword"):
        registry.resolve_policy(
            "external/retention@1", kind="graph_pair_policy", config={"typo": True}
        )
    with pytest.raises(ValueError, match="unknown retention"):
        registry.resolve_policy(
            "external/retention@1", kind="graph_pair_policy", config={"save": "guess"}
        )
    with pytest.raises(ValueError, match="already registered"):
        registry.add_policy(definition(), source=SOURCE)
    with pytest.raises(ValueError, match="mode must"):
        registry.resolve_policy(
            "shadowspill/partition@1", kind="partition", config={"mode": None}
        )


def test_policy_source_and_config_resolution_are_framework_free():
    script = """
import sys
for name in ('torch', 'triton', 'shadowspill_workloads'):
    sys.modules[name] = None
from engine_tests.training.test_policy_definitions import (
    definition, settings, stacked_context, preset, SOURCE,
)
from shadowspill.training.definitions import Registry
from shadowspill.training.preparation.configuration.resolution import (
    resolve_preparation,
)
from shadowspill.training.preparation.configuration.policies import (
    resolve_build_policies,
)
registry = Registry()
registry.add_policy(definition(), source=SOURCE)
resolved = resolve_preparation(preset(settings()), stacked_context())
bound, sources = resolve_build_policies(registry, resolved)
assert sources['external/retention@1'] == SOURCE
assert bound.cases[1].build.values['graph_pair_policy']['config']['save'] == 'cost'
"""
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )

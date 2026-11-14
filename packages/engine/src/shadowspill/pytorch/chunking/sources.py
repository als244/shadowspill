"""Bind structural chunk slices to the current occurrence's source metadata."""

from dataclasses import replace

from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.torch_deprecations import copy_graph_module

from .geometry import ChunkTask

CHUNK_SOURCE_INDEX = "shadowspill_chunk_source_index"


def bind_chunk_sources(
    template: GraphArtifact,
    source: GraphArtifact,
    current: GraphArtifact,
    domain: str | None,
) -> ChunkTask:
    """Keep sliced geometry/rewrites while rebinding unchanged source fields.

    FX copies retain node metadata values by reference. Compare against the
    template's source to distinguish inherited fields from transform overrides;
    arbitrary current source metadata need not be serialized into a cache key.
    Child tensors remain abstract slice examples, not runtime input bindings.
    """
    task = template.chunking
    assert task is not None
    before = tuple(source.graph_module.graph.nodes)
    after = tuple(current.graph_module.graph.nodes)
    assert len(before) == len(after)

    def bind(child: GraphArtifact) -> GraphArtifact:
        graph = copy_graph_module(child.graph_module)
        graph.meta = dict(current.graph_module.meta)
        for node in graph.graph.nodes:
            index = node.meta.get(CHUNK_SOURCE_INDEX)
            if index is None:
                continue
            old, new = before[index].meta, after[index].meta
            transformed = node.meta
            node.meta = {
                key: value
                for key, value in new.items()
                if key not in old or key in transformed
            }
            node.meta.update(
                {
                    key: value
                    for key, value in transformed.items()
                    if key not in old or value is not old[key]
                }
            )
            node.meta[CHUNK_SOURCE_INDEX] = index
            if domain is not None:
                domains = dict(new.get("execution_domains", {}))
                old_domains = old.get("execution_domains", {})
                transformed_domains = transformed.get("execution_domains", {})
                for name in old_domains.keys() | transformed_domains.keys():
                    if (
                        old_domains.get(name) != transformed_domains.get(name)
                        or name == domain
                    ):
                        if name in transformed_domains:
                            domains[name] = transformed_domains[name]
                        else:
                            domains.pop(name, None)
                node.meta["execution_domains"] = domains
        return replace(
            child,
            graph_module=graph,
            input_provenance=(
                *current.input_provenance,
                *child.input_provenance[current.argument_count :],
            ),
        )

    return replace(
        task,
        children=tuple((size, bind(child)) for size, child in task.children),
        accumulating_children=tuple(
            (size, bind(child)) for size, child in task.accumulating_children
        ),
    )

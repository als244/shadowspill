"""Source module paths carried by captured graph nodes."""

from torch.fx import GraphModule


def module_paths(graph: GraphModule) -> tuple[str, ...]:
    paths: dict[str, None] = {}
    for node in graph.graph.nodes:
        stack = node.meta.get("nn_module_stack", {})
        for path, _type in stack.values():
            if path:
                paths.setdefault(path, None)
    return tuple(paths)

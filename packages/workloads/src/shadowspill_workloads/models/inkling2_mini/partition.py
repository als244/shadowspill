"""Separate routed experts at global-attention visits in Inkling's repeated core."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitGlobalExperts:
    """Use the normal layer boundaries and add one before each core global MLP."""

    def assign_stages(self, graph_module, module):
        from shadowspill.pytorch.partition.policy import resolve_partition_assignments

        from .implementations.mlops.model import Model

        models = [
            (path, child)
            for path, child in module.named_modules()
            if isinstance(child, Model)
        ]
        if len(models) != 1:
            raise ValueError(
                "the Inkling expert partition requires one Inkling text decoder"
            )
        path, model = models[0]
        config = model.config
        prefix = f"{path}." if path else ""
        experts = {
            f"{prefix}layers.{i}.mlp"
            for i in range(config.core_start, config.core_end)
            if i not in config.local_layer_ids
        }
        base, _ = resolve_partition_assignments(graph_module, module, "auto")
        result = {}
        previous = None
        extra = 0
        in_experts = False
        for node, stage in base.items():
            if stage != previous:
                in_experts = False
                previous = stage
            if not in_experts and any(
                owner in experts
                for owner, _ in node.meta.get("nn_module_stack", {}).values()
            ):
                extra += 1
                in_experts = True
            result[node.name] = stage + extra
        if extra != len(experts) * config.passes:
            raise ValueError(
                "exported graph did not contain all expected Inkling expert visits"
            )
        return result


def register(registry):
    from shadowspill.training.components import PolicyComponent
    from shadowspill_workloads.components.source import component_source

    registry.add_policy(
        PolicyComponent(
            "shadowspill/inkling2_mini_split_global_experts@1",
            "partition",
            SplitGlobalExperts,
            lambda config: config,
        ),
        source=component_source(("models/inkling2_mini/partition.py",)),
    )

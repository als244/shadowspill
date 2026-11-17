"""Public provider rules with independently bounded nested execution."""


def axis_rules(batch):
    from shadowspill.integrations.mlops import sonicmoe_chunk_rules
    from shadowspill.pytorch import ChunkAxis, OperatorRule

    flat = ChunkAxis(0, before=batch)
    routing = (
        ("lora_router_logits", (flat,), ()),
        ("lora_router_logits_bwd", (flat, None), (1,)),
        ("lora_route", (flat, flat), ()),
        ("lora_route_bwd", (flat,), ()),
    )
    return (
        *(
            OperatorRule(f"mlops.{name}.default", "1", axes, reductions)
            for name, axes, reductions in routing
        ),
        *sonicmoe_chunk_rules(),
    )


class InnerChunks:
    def __init__(self, variant, batch):
        self.variant, self.batch = variant, batch

    def configure_stage(self, stage):
        from shadowspill.pytorch import AxisChunks

        axes = {
            index: domains["positions"]
            for index, domains in enumerate(stage.input_domains)
            if "positions" in domains
        }
        size = 17 if any("sonicmoe" in target for target in stage.operators) else 9
        return (
            AxisChunks(
                size, axes, (self.variant,), axis_rules(self.batch), domain="positions"
            )
            if axes
            else None
        )


class Chunks:
    def __init__(self, variant, batch=1):
        self.variant, self.batch = variant, batch

    def configure_stage(self, stage):
        from shadowspill.pytorch import AxisChunks, ChunkRegions
        from shadowspill.task.inputs import TaskInputRole

        if stage.terminal:
            return None
        axes = {
            index: 1
            for index, (value, role) in enumerate(
                zip(stage.inputs, stage.input_roles, strict=True)
            )
            if value is not None
            and len(value.shape) == 3
            and value.shape[0] == self.batch
            and role not in {TaskInputRole.PARAMETER, TaskInputRole.BUFFER}
        }
        return (
            AxisChunks(
                33,
                axes,
                (self.variant,),
                axis_rules(self.batch),
                domain="positions",
                regions=ChunkRegions(
                    "whole",
                    InnerChunks(self.variant, self.batch),
                    variants=(self.variant,),
                ),
            )
            if axes
            else None
        )


def pair_policy():
    from shadowspill.pytorch import GraphPairPolicy, RetentionPolicy, RetentionRule

    # Opaque sorted rows cannot escape a chunk. Retain ordinary inputs/choices
    # while regenerating these provider-local values within a bounded child.
    return GraphPairPolicy(
        save=RetentionPolicy(
            rules=tuple(
                RetentionRule(f"mlops.{name}.default", "recompute")
                for name in (
                    "sonicmoe_routing_metadata",
                    "lora_sonicmoe_up_fwd",
                    "lora_sonicmoe_down_fwd",
                )
            )
        )
    )

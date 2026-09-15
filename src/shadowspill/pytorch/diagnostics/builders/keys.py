"""How a stage, a pair and an entrypoint are named in a report."""

import hashlib
import json

from shadowspill.pytorch.capture.artifacts import AotGraphPair
from shadowspill.pytorch.graph_pairs import (
    DifferentiatedStage,
)
from shadowspill.pytorch.lowering.training import (
    TrainingTaskEntrypoint,
)


def _stage_key(stage: DifferentiatedStage) -> str:
    payload = {
        "roots": list(stage.differentiable_output_indices),
        "variants": [
            {
                "option_id": item.option_id,
                "memory_budget": item.memory_budget,
                "pair": _pair_key(item.pair),
            }
            for item in stage.graph_pairs.variants
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _pair_key(pair: AotGraphPair) -> dict[str, object]:
    return {
        "forward": pair.forward.compatibility_digest,
        "backward": pair.backward.compatibility_digest,
        "recomputation": pair.recomputation,
        "saved_value_count": pair.saved_value_count,
        "specialized_unit_tangent_count": pair.specialized_unit_tangent_count,
    }


def _entrypoint_key(
    entrypoint: TrainingTaskEntrypoint,
) -> tuple[int, int, str, str]:
    if (
        entrypoint.microbatch is None
        or entrypoint.stage_index is None
        or entrypoint.variant is None
    ):
        raise ValueError("entrypoint has no graph-stage identity")
    return (
        entrypoint.microbatch,
        entrypoint.stage_index,
        entrypoint.variant,
        entrypoint.phase,
    )

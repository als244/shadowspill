from __future__ import annotations

from tools.planning.geometry import DataGeometry


def test_data_geometry_groups_primary_and_derived_axes() -> None:
    geometry = DataGeometry(1024, 8192, 8)

    assert geometry.to_dict() == {
        "sequence_length": 1024,
        "tokens_per_microbatch": 8192,
        "sequences_per_microbatch": 8,
        "accumulation_rounds": 8,
        "tokens_per_optimizer_step": 65536,
    }

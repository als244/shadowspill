"""Validate captured scalar representation without importing a GPU framework."""

import struct
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

from shadowspill_client.protocol.controls import ControlSchema, Scalar


def control_values(
    schema: ControlSchema, values: Mapping[str, Scalar]
) -> Mapping[str, Scalar]:
    """Resolve exact float32/int64/bool values used by the existing capture ABI."""
    if values.keys() != schema.fields.keys():
        raise ValueError("provide exactly the declared scalar controls")
    result = {}
    for name, field in sorted(schema.fields.items()):
        try:
            value = field.validate(values[name])
            if field.dtype == "float":
                value = cast(float, struct.unpack("<f", struct.pack("<f", value))[0])
            elif field.dtype == "int" and not -(1 << 63) <= value < 1 << 63:
                raise OverflowError("integer control must fit int64")
            # Bounds crossed by rounding, underflow and non-finite values must
            # fail before the operation can reach any worker microbatch.
            result[name] = field.validate(value)
        except (ValueError, OverflowError, struct.error) as error:
            raise ValueError(
                f"control {name} cannot use its captured scalar dtype"
            ) from error
    return MappingProxyType(result)

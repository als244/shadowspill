"""Portable parameter-storage rules shared by model metadata and the engine."""

from collections.abc import Mapping
from types import MappingProxyType


def parameter_dtypes(value: object) -> Mapping[str, str]:
    """Validate portable rules without importing a framework or constructing tensors."""
    if not isinstance(value, Mapping):
        raise TypeError("parameter_dtypes must map tensor-name patterns to dtypes")
    result = {}
    for name, dtype in value.items():
        if not isinstance(name, str) or not name or name.strip() != name:
            raise ValueError("parameter_dtypes requires nonempty tensor-name patterns")
        if not isinstance(dtype, str) or dtype not in (
            "float16",
            "bfloat16",
            "float32",
            "float64",
        ):
            raise ValueError(
                f"unsupported parameter dtype for {name!r}: {dtype!r}; "
                "choose float16, bfloat16, float32 or float64"
            )
        result[name] = dtype
    return MappingProxyType(result)

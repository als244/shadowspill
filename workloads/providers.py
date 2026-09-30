"""Model implementation names; mlops resolves each operation from its inputs."""

from typing import Literal

ModelImplementation = Literal["pytorch", "mlops"]
MODEL_DTYPES = ("float16", "bfloat16", "float32")

__all__ = ["MODEL_DTYPES", "ModelImplementation"]

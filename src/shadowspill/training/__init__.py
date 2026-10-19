"""Generic training and forward execution for ordinary PyTorch models."""

from shadowspill.pytorch.distributed import Distributed, TransferSharding

from ._model import reset_parameters
from .forward import Forward
from .trainer import EvaluationResult, StepResult, Trainer

__all__ = [
    "Distributed",
    "EvaluationResult",
    "Forward",
    "StepResult",
    "Trainer",
    "TransferSharding",
    "reset_parameters",
]

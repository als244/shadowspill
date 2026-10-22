"""Generic training and forward execution for ordinary PyTorch models."""

from shadowspill.pytorch.distributed import Distributed, TransferSharding

from ._model import reset_parameters
from .forward import Forward
from .parallel import DataParallel
from .preparation import PreparedTraining
from .trainer import EvaluationResult, StepResult, Trainer

__all__ = [
    "DataParallel",
    "Distributed",
    "EvaluationResult",
    "Forward",
    "PreparedTraining",
    "StepResult",
    "Trainer",
    "TransferSharding",
    "reset_parameters",
]

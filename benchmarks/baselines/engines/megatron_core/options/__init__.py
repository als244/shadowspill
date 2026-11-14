"""Ordered engine option catalog; validation lives in settings."""

from .model import MODEL
from .parallelism import PARALLELISM
from .training import TRAINING

OPTIONS = (*PARALLELISM, *TRAINING, *MODEL)
BY_NAME = {option.name: option for option in OPTIONS}

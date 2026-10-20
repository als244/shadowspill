"""The supplied model families, one package each.

A family package holds its configuration dataclass (``config.py``), its
Hugging Face checkpoint reader where one exists (``checkpoint.py``), its
README and its implementations (``implementations/pytorch`` and
``implementations/mlops``), every one exposing ``Config`` and ``Model``. Code
two families share lives in ``shared``.
"""

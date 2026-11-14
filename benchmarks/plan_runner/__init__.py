"""Benchmark saved training Plans without building or planning again.

``benchmark`` launches a saved public recipe in fresh single-node workers.
``run`` accepts arbitrary workload hooks in an already launched worker.
Both use the public Trainer and retain its normal execution checks.
"""

__all__ = ["Context", "NsysCapture", "benchmark", "run"]


def __getattr__(name):
    if name == "NsysCapture":
        from shadowspill.orchestration.capture.nsys import NsysCapture

        return NsysCapture
    if name == "benchmark":
        from .recipe import benchmark

        return benchmark
    if name in {"Context", "run"}:
        from . import api

        return getattr(api, name)
    raise AttributeError(name)

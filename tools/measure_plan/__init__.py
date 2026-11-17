"""Measure saved training Plans through the public Executor."""

__all__ = ["Context", "NsysCapture", "run"]


def __getattr__(name):
    if name == "NsysCapture":
        from shadowspill.orchestration.capture.nsys import NsysCapture

        return NsysCapture
    if name in {"Context", "run"}:
        from . import api

        return getattr(api, name)
    raise AttributeError(name)

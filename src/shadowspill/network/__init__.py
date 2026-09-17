"""Pool memory, and later lanes, that live on another machine.

What makes this a separate package is not the subject matter but the
dependency: everything here needs a shared object the neutral runtime does not
link and does not load. A caller that never asks for a remote pool never
imports it, and a build without it still runs everything else.
"""

from __future__ import annotations

from .memory import RemotePool, remote

__all__ = ["RemotePool", "remote"]

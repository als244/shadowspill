"""The one version every ShadowSpill artifact carries.

Every serialized structure, in the artifact store or outside it (plans,
programs, schedules, profiles, diagnostics, qualification results, fixtures),
names its schema as ``shadowspill.<kind>/v<ARTIFACT_VERSION>``.  Bump the
version when any of them changes: the store then writes a fresh ``v<N>`` tree
beside the old one and replans, and inside one tree a schema that does not
match is corruption rather than a stale version.  The C library carries the
same number as ``SHADOWSPILL_ARTIFACT_VERSION`` for the digests it computes.
"""

from __future__ import annotations

from typing import Final

ARTIFACT_VERSION: Final = 1


def artifact_schema(kind: str) -> str:
    return f"shadowspill.{kind}/v{ARTIFACT_VERSION}"


__all__ = ["ARTIFACT_VERSION", "artifact_schema"]

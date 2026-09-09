"""What a store is allowed to do, in one place.

Four stores hold four different kinds of artifact -- profiles, graph pairs,
optimizer captures, plans -- and serialize them four different ways. What
they may *do* is the same for all four, and comes from one `StoreMode` on
the artifact store: read a hit, write a miss, overwrite what is there, or
refuse a miss outright.

Keeping the policy here is what stops a mode from being honoured by three
stores and quietly ignored by the fourth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: What a run does with one tree of the artifact store.
#:
#: ``contribute``  read a hit, write a miss. The default.
#: ``reuse``       read a hit, persist nothing.
#: ``require``     read a hit, refuse a miss.
#: ``refresh``     ignore hits, rebuild and replace.
type StoreMode = Literal["contribute", "reuse", "require", "refresh"]

#: The same four, as a value a CLI can offer and a config can validate
#: against. Written once so a surface cannot drift into offering three.
STORE_MODES: tuple[StoreMode, ...] = ("contribute", "reuse", "require", "refresh")


@dataclass(frozen=True, slots=True)
class StorePolicy:
    """The four gates a `StoreMode` implies."""

    #: Read a stored artifact back when one is there.
    read_enabled: bool = True
    #: Write what this run produced.
    write_enabled: bool = True
    #: Replace a stored artifact rather than keeping it.
    overwrite: bool = False
    #: Refuse a miss instead of doing the work. What makes a run prove it is
    #: reusing a store rather than silently rebuilding one.
    require_hit: bool = False

    @classmethod
    def for_mode(cls, mode: StoreMode) -> StorePolicy:
        """The policy one mode implies."""

        return cls(
            read_enabled=mode != "refresh",
            write_enabled=mode in ("contribute", "refresh"),
            overwrite=mode == "refresh",
            require_hit=mode == "require",
        )

    def refuse_miss(self, what: str, key: str) -> None:
        """Raise when a miss is not allowed. Names what would fix it.

        Called by a store on the read path, after a lookup came back empty
        and before any work is done to produce the artifact.
        """

        if not self.require_hit:
            return
        raise LookupError(
            f"{what} {key} is not in the store, and the store mode is "
            "'require', which refuses to produce one. Use 'reuse' to plan "
            "this without writing, or 'contribute' to add it."
        )


#: What a store does when no policy is named: read a hit, write a miss.
CONTRIBUTE = StorePolicy()


__all__ = ["CONTRIBUTE", "STORE_MODES", "StoreMode", "StorePolicy"]

"""What a store is allowed to do, in one place.

Several stores hold different kinds of artifact -- graph pairs, optimizer
captures, archived programs, plans -- and serialize them
differently. What they may *do* is the same for all of them, and comes from
one `StoreMode` on the artifact store: read a hit, write a miss, overwrite
what is there, or refuse a miss outright.

Keeping the policy here is what stops a mode from being honoured by some
stores and quietly ignored by another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: What a run does with one tree of the artifact store: ``"contribute"``
#: reads a hit and writes a miss (the default); ``"reuse"`` reads a hit and
#: writes nothing; ``"require"`` reads a hit and refuses a miss; and
#: ``"refresh"`` ignores hits, rebuilds and replaces them.
type StoreMode = Literal["contribute", "reuse", "require", "refresh"]

#: The four store modes, for a command line to offer or a configuration to
#: validate against.
STORE_MODES: tuple[StoreMode, ...] = ("contribute", "reuse", "require", "refresh")


@dataclass(frozen=True, slots=True)
class StorePolicy:
    """What a run may do with one tree of the artifact store.

    `StorePolicy.for_mode` builds the policy a store mode implies; every store
    reads its gates from it.

    Parameters
    ----------
    read_enabled : bool, default True
        Whether a stored artifact is read back when one is there.
    write_enabled : bool, default True
        Whether what the run produces is written.
    overwrite : bool, default False
        Whether a stored artifact is replaced rather than kept.
    require_hit : bool, default False
        Whether a miss is refused instead of doing the work, so a run proves it
        reuses a store rather than silently rebuilding one.
    """

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
        """Return the policy a store mode implies.

        Parameters
        ----------
        mode : StoreMode
            The mode.

        Returns
        -------
        StorePolicy
            Its gates.
        """

        return cls(
            read_enabled=mode != "refresh",
            write_enabled=mode in ("contribute", "refresh"),
            overwrite=mode == "refresh",
            require_hit=mode == "require",
        )

    def refuse_miss(self, what: str, key: str, *, request: str | None = None) -> None:
        """Raise if this policy refuses a miss; return otherwise.

        A store calls it after a lookup comes back empty and before any work is done
        to produce the artifact. The message names the modes that would allow the
        miss.

        Parameters
        ----------
        what : str
            The kind of artifact missed, as the message names it.
        key : str
            The key that missed.
        request : str, optional
            What the key was made of, so a miss can be read against the records the
            store holds.

        Raises
        ------
        LookupError
            If ``require_hit`` is set.
        """

        if not self.require_hit:
            return
        asked = "" if request is None else f" ({request})"
        raise LookupError(
            f"{what} {key} is not in the store{asked}, and the store mode is "
            "'require', which refuses to produce one. Use 'reuse' to plan "
            "this without writing, or 'contribute' to add it."
        )


#: What a store does when no policy is named: read a hit, write a miss.
CONTRIBUTE = StorePolicy()


__all__ = ["CONTRIBUTE", "STORE_MODES", "StoreMode", "StorePolicy"]

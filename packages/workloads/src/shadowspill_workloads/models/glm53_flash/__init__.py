"""GLM-5.3-Flash: 45 decoder layers (34 KDA, 11 NoPE sparse MLA) on mHC
residual streams, three initial dense MLPs, then 288 routed experts/top-8
plus a shared expert, with an optional vision encoder."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config

__all__ = ["Config"]


def __getattr__(name):
    if name == "Config":
        from .config import Config

        return Config
    raise AttributeError(name)

"""GLM-5.3: 78 decoder layers with positional sparse MLA and shared
attention indices, ordinary residuals, three initial dense MLPs, then
256 routed experts per token-8 plus a shared expert."""

from .config import Config

__all__ = ["Config"]

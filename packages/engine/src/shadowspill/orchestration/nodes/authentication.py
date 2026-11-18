"""Operator-issued node credentials, separate from client owner credentials."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from shadowspill.server.authentication import BearerTokens
from shadowspill_client._records import text
from shadowspill_client.protocol.requests import identifier


@dataclass(frozen=True)
class NodeAccess:
    node_id: str
    pool: str = "default"

    def __post_init__(self) -> None:
        identifier(self.node_id, "authorized node ID")
        if len(text(self.pool, "authorized pool")) > 256:
            raise ValueError("pool name exceeds its bound")


class NodeTokens:
    def __init__(self, tokens: Mapping[str, NodeAccess]) -> None:
        if not isinstance(tokens, Mapping) or any(
            not isinstance(access, NodeAccess) for access in tokens.values()
        ):
            raise TypeError("node credentials require NodeAccess policies")
        self._tokens = BearerTokens(
            {token: value.node_id for token, value in tokens.items()}
        )
        self._policies = {value.node_id: value for value in tokens.values()}
        if len(self._policies) != len(tokens):
            raise ValueError("each node identity requires one credential")

    def authorize(self, headers: Sequence[str]) -> NodeAccess:
        return self._policies[self._tokens.owner(headers)]

    def require_separate(self, clients: BearerTokens) -> None:
        clients.require_separate(self._tokens)

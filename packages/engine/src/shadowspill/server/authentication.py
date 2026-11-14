"""Deployment-owned bearer authentication; owner identities never come from URLs."""

import hashlib
import hmac
from collections.abc import Mapping, Sequence

from shadowspill_client.protocol.requests import identifier
from shadowspill_client.protocol.service import bearer_token


class BearerTokens:
    """Retain only credential digests and explicit authenticated owner IDs."""

    def __init__(self, tokens: Mapping[str, str]) -> None:
        if not isinstance(tokens, Mapping) or not 0 < len(tokens) <= 1024:
            raise ValueError("configure 1-1024 service credentials")
        self._owners = tuple(
            (
                hashlib.sha256(bearer_token(token).encode("ascii")).digest(),
                identifier(owner, "authenticated owner ID"),
            )
            for token, owner in tokens.items()
        )

    def owner(self, authorization: Sequence[str]) -> str:
        if len(authorization) != 1:
            raise PermissionError("Service authentication failed.")
        scheme, separator, token = authorization[0].partition(" ")
        if not separator or scheme.lower() != "bearer":
            raise PermissionError("Service authentication failed.")
        try:
            candidate = hashlib.sha256(bearer_token(token).encode("ascii")).digest()
        except (TypeError, ValueError):
            raise PermissionError("Service authentication failed.") from None
        owner = None
        for expected, identity in self._owners:
            if hmac.compare_digest(candidate, expected):
                owner = identity
        if owner is None:
            raise PermissionError("Service authentication failed.")
        return owner

"""Deployment-owned bearer authentication; owner identities never come from URLs."""

import hashlib
import hmac
from collections.abc import Mapping, Sequence

from shadowspill_client.protocol.requests import identifier
from shadowspill_client.protocol.service import bearer_token


class BearerTokens:
    """Client credentials for a `Server`, each with the owner ID it authenticates as.

    A request presents one in an ``Authorization: Bearer`` header and acts as
    its owner. Only the credentials' SHA-256 digests are kept, and they are
    compared in constant time.

    Parameters
    ----------
    tokens : mapping of str to str
        Each credential, mapped to its owner ID; several may share an owner.
        From 1 to 1024 entries. A credential is 32 to 512 visible ASCII
        characters. An owner ID is up to 128 letters, digits and ``_.:-``,
        starting with a letter or digit.

    Raises
    ------
    ValueError
        If ``tokens`` is not a mapping of 1 to 1024 entries, or holds an
        invalid credential or owner ID.
    """

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
        """Return the owner a request's credential authenticates as.

        Parameters
        ----------
        authorization : sequence of str
            The request's ``Authorization`` header values.

        Returns
        -------
        str
            The owner ID.

        Raises
        ------
        PermissionError
            If there is not exactly one value, or it is not ``Bearer`` (in any
            case), a space and a configured credential.
        """
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

    def require_separate(self, other: "BearerTokens") -> None:
        """Check that no credential is also one of another set's.

        `Server` checks its node credentials against its client credentials,
        so that no credential confers both.

        Parameters
        ----------
        other : BearerTokens
            The other credentials.

        Raises
        ------
        ValueError
            If a credential is in both.
        """
        if any(
            hmac.compare_digest(left, right)
            for left, _ in self._owners
            for right, _ in other._owners
        ):
            raise ValueError("client and node credentials must be distinct")
